#include "runtime/fingerprint.hpp"

#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <NvInferRuntime.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <openssl/evp.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string build_error_message(
    const RuntimeFingerprintStage stage,
    const std::string_view detail) {
  return "runtime fingerprint collection failed [stage=" +
         std::string(to_string(stage)) + "]: " + std::string(detail);
}

[[noreturn]] void fail(
    const RuntimeFingerprintStage stage,
    const std::string& detail) {
  throw RuntimeFingerprintError(stage, detail);
}

[[nodiscard]] std::string unquote_os_release_value(
    const std::string_view raw,
    const std::size_t line_number) {
  if (raw.empty()) {
    fail(
        RuntimeFingerprintStage::operating_system,
        "os-release line " + std::to_string(line_number) +
            " has an empty value");
  }
  if (raw.front() != '"' && raw.front() != '\'') {
    return std::string(raw);
  }
  const char quote = raw.front();
  if (raw.size() < 2 || raw.back() != quote) {
    fail(
        RuntimeFingerprintStage::operating_system,
        "os-release line " + std::to_string(line_number) +
            " has an unterminated quoted value");
  }
  std::string value;
  value.reserve(raw.size() - 2);
  for (std::size_t index = 1; index + 1 < raw.size(); ++index) {
    const char character = raw.at(index);
    if (character != '\\') {
      value.push_back(character);
      continue;
    }
    ++index;
    if (index + 1 >= raw.size()) {
      fail(
          RuntimeFingerprintStage::operating_system,
          "os-release line " + std::to_string(line_number) +
              " ends with an escape");
    }
    const char escaped = raw.at(index);
    if (escaped != '\\' && escaped != '"' && escaped != '\'' &&
        escaped != '$' && escaped != '`') {
      fail(
          RuntimeFingerprintStage::operating_system,
          "os-release line " + std::to_string(line_number) +
              " contains an unsupported escape");
    }
    value.push_back(escaped);
  }
  return value;
}

[[nodiscard]] std::string require_distribution_value(
    const std::map<std::string, std::string>& values,
    const std::string_view key) {
  const auto found = values.find(std::string(key));
  if (found == values.end() || found->second.empty()) {
    fail(
        RuntimeFingerprintStage::operating_system,
        "os-release is missing required field " + std::string(key));
  }
  return found->second;
}

[[nodiscard]] std::string format_cuda_runtime_version(
    const int runtime_version) {
  if (runtime_version <= 0) {
    fail(
        RuntimeFingerprintStage::cuda_runtime,
        "cudaRuntimeGetVersion returned a non-positive version");
  }
  const int major = runtime_version / 1000;
  const int minor = (runtime_version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

[[nodiscard]] std::string format_tensorrt_runtime_version() {
  const std::int32_t major = getInferLibMajorVersion();
  const std::int32_t minor = getInferLibMinorVersion();
  const std::int32_t patch = getInferLibPatchVersion();
  const std::int32_t build = getInferLibBuildVersion();
  if (major <= 0 || minor < 0 || patch < 0 || build < 0) {
    fail(
        RuntimeFingerprintStage::tensorrt_runtime,
        "TensorRT returned an invalid version component");
  }
  return std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch) + "." + std::to_string(build);
}

class NvmlSession final {
 public:
  NvmlSession() {
    const nvmlReturn_t status = nvmlInit_v2();
    if (status != NVML_SUCCESS) {
      fail(
          RuntimeFingerprintStage::nvidia_driver,
          "nvmlInit_v2 failed: " +
              std::string(nvmlErrorString(status)));
    }
    active_ = true;
  }

  NvmlSession(const NvmlSession&) = delete;
  NvmlSession& operator=(const NvmlSession&) = delete;

  ~NvmlSession() {
    if (active_) {
      static_cast<void>(nvmlShutdown());
    }
  }

  void close() {
    if (!active_) {
      return;
    }
    const nvmlReturn_t status = nvmlShutdown();
    if (status != NVML_SUCCESS) {
      fail(
          RuntimeFingerprintStage::nvidia_driver,
          "nvmlShutdown failed: " +
              std::string(nvmlErrorString(status)));
    }
    active_ = false;
  }

 private:
  bool active_{false};
};

[[nodiscard]] std::string collect_driver_version() {
  NvmlSession session;
  char value[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]{};
  const nvmlReturn_t status =
      nvmlSystemGetDriverVersion(value, sizeof(value));
  if (status != NVML_SUCCESS) {
    fail(
        RuntimeFingerprintStage::nvidia_driver,
        "nvmlSystemGetDriverVersion failed: " +
            std::string(nvmlErrorString(status)));
  }
  if (value[0] == '\0') {
    fail(
        RuntimeFingerprintStage::nvidia_driver,
        "NVML returned an empty driver version");
  }
  session.close();
  return value;
}

class FileDescriptor final {
 public:
  explicit FileDescriptor(const int value) : value_(value) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  ~FileDescriptor() {
    if (value_ >= 0) {
      static_cast<void>(::close(value_));
    }
  }

  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_;
};

struct MappedFileIdentity {
  dev_t device;
  ino_t inode;
};

[[nodiscard]] std::uint64_t parse_unsigned(
    const std::string_view value,
    const int base,
    const std::string_view field) {
  std::uint64_t parsed = 0U;
  const auto [end, error] = std::from_chars(
      value.data(), value.data() + value.size(), parsed, base);
  if (error != std::errc{} || end != value.data() + value.size()) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "malformed /proc/self/maps " + std::string(field));
  }
  return parsed;
}

[[nodiscard]] MappedFileIdentity mapped_file_identity(
    const std::uintptr_t address) {
  std::ifstream maps("/proc/self/maps");
  if (!maps) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "unable to open /proc/self/maps: " +
            std::string(std::strerror(errno)));
  }
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream fields(line);
    std::string range;
    std::string permissions;
    std::string offset;
    std::string device;
    std::string inode;
    if (!(fields >> range >> permissions >> offset >> device >> inode)) {
      fail(
          RuntimeFingerprintStage::cublas_runtime,
          "malformed entry in /proc/self/maps");
    }
    const std::size_t range_separator = range.find('-');
    if (range_separator == std::string::npos) {
      fail(
          RuntimeFingerprintStage::cublas_runtime,
          "malformed address range in /proc/self/maps");
    }
    const std::uint64_t start = parse_unsigned(
        std::string_view(range).substr(0, range_separator),
        16,
        "mapping start");
    const std::uint64_t end = parse_unsigned(
        std::string_view(range).substr(range_separator + 1),
        16,
        "mapping end");
    if (address < start || address >= end) {
      continue;
    }
    const std::size_t device_separator = device.find(':');
    if (device_separator == std::string::npos) {
      fail(
          RuntimeFingerprintStage::cublas_runtime,
          "malformed device identity in /proc/self/maps");
    }
    const std::uint64_t device_major = parse_unsigned(
        std::string_view(device).substr(0, device_separator),
        16,
        "mapping device major");
    const std::uint64_t device_minor = parse_unsigned(
        std::string_view(device).substr(device_separator + 1),
        16,
        "mapping device minor");
    const std::uint64_t inode_value =
        parse_unsigned(inode, 10, "mapping inode");
    if (inode_value == 0U ||
        device_major > std::numeric_limits<unsigned int>::max() ||
        device_minor > std::numeric_limits<unsigned int>::max() ||
        inode_value >
            static_cast<std::uint64_t>(
                std::numeric_limits<ino_t>::max())) {
      fail(
          RuntimeFingerprintStage::cublas_runtime,
          "cuBLAS symbol is not backed by a valid regular-file mapping");
    }
    return MappedFileIdentity{
        .device = ::makedev(
            static_cast<unsigned int>(device_major),
            static_cast<unsigned int>(device_minor)),
        .inode = static_cast<ino_t>(inode_value),
    };
  }
  if (!maps.eof()) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "failed while reading /proc/self/maps");
  }
  fail(
      RuntimeFingerprintStage::cublas_runtime,
      "cuBLAS provider symbol has no file-backed process mapping");
}

[[nodiscard]] bool same_stable_file(
    const struct stat& before,
    const struct stat& after) noexcept {
  return before.st_dev == after.st_dev &&
         before.st_ino == after.st_ino &&
         before.st_mode == after.st_mode &&
         before.st_size == after.st_size &&
         before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

[[nodiscard]] std::string sha256_descriptor(const int descriptor) {
  using DigestContext =
      std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  DigestContext context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "unable to initialize loaded-library SHA-256");
  }
  if (::lseek(descriptor, 0, SEEK_SET) < 0) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "unable to seek loaded library: " +
            std::string(std::strerror(errno)));
  }
  std::array<unsigned char, 64U * 1024U> buffer{};
  for (;;) {
    const ssize_t bytes_read = ::read(
        descriptor, buffer.data(), buffer.size());
    if (bytes_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail(
          RuntimeFingerprintStage::cublas_runtime,
          "unable to read loaded library: " +
              std::string(std::strerror(errno)));
    }
    if (bytes_read == 0) {
      break;
    }
    if (EVP_DigestUpdate(
            context.get(),
            buffer.data(),
            static_cast<std::size_t>(bytes_read)) != 1) {
      fail(
          RuntimeFingerprintStage::cublas_runtime,
          "unable to update loaded-library SHA-256");
    }
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0U;
  if (EVP_DigestFinal_ex(
          context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != 32U) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "unable to finalize loaded-library SHA-256");
  }
  constexpr std::string_view hexadecimal = "0123456789abcdef";
  std::string result;
  result.resize(digest_size * 2U);
  for (std::size_t index = 0; index < digest_size; ++index) {
    result.at(index * 2U) = hexadecimal.at(digest.at(index) >> 4U);
    result.at(index * 2U + 1U) =
        hexadecimal.at(digest.at(index) & 0x0FU);
  }
  return result;
}

[[nodiscard]] void* resolve_plugin_dependency_symbol(
    void* const plugin_handle,
    const char* const symbol_name) {
  if (plugin_handle == nullptr) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "authenticated plugin handle is null");
  }
  static_cast<void>(::dlerror());
  void* const symbol = ::dlsym(plugin_handle, symbol_name);
  const char* const error = ::dlerror();
  if (error != nullptr || symbol == nullptr) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "unable to resolve " + std::string(symbol_name) +
            " through the authenticated plugin dependency scope: " +
            (error == nullptr ? "null symbol" : std::string(error)));
  }
  return symbol;
}

template <typename Function>
[[nodiscard]] Function function_from_symbol(void* const symbol) {
  Function function = nullptr;
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

[[nodiscard]] std::pair<Dl_info, std::string> loaded_provider_info(
    void* const provider_symbol,
    const std::string_view expected_soname) {
  Dl_info information{};
  void* extra_information = nullptr;
  if (::dladdr1(
          provider_symbol,
          &information,
          &extra_information,
          RTLD_DL_LINKMAP) == 0 ||
      information.dli_fname == nullptr ||
      information.dli_fname[0] == '\0' ||
      extra_information == nullptr) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "dladdr did not identify the provider for " +
            std::string(expected_soname));
  }
  const auto* const link_map =
      static_cast<const struct link_map*>(extra_information);
  if (link_map->l_ld == nullptr) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded provider has no dynamic section for " +
            std::string(expected_soname));
  }
  const char* string_table = nullptr;
  std::size_t string_table_size = 0U;
  std::size_t soname_offset = 0U;
  bool found_soname = false;
  bool found_terminator = false;
  constexpr std::size_t kMaximumDynamicEntries = 16U * 1024U;
  for (std::size_t index = 0U; index < kMaximumDynamicEntries; ++index) {
    const ElfW(Dyn)& entry = link_map->l_ld[index];
    if (entry.d_tag == DT_NULL) {
      found_terminator = true;
      break;
    }
    if (entry.d_tag == DT_STRTAB) {
      string_table = reinterpret_cast<const char*>(entry.d_un.d_ptr);
    } else if (entry.d_tag == DT_STRSZ) {
      string_table_size = static_cast<std::size_t>(entry.d_un.d_val);
    } else if (entry.d_tag == DT_SONAME) {
      soname_offset = static_cast<std::size_t>(entry.d_un.d_val);
      found_soname = true;
    }
  }
  if (!found_terminator || string_table == nullptr ||
      string_table_size == 0U || !found_soname ||
      soname_offset >= string_table_size) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded provider has no bounded ELF SONAME for " +
            std::string(expected_soname));
  }
  const char* const soname_begin = string_table + soname_offset;
  const std::size_t remaining = string_table_size - soname_offset;
  const void* const terminator = std::memchr(soname_begin, '\0', remaining);
  if (terminator == nullptr) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded provider has an unterminated ELF SONAME for " +
            std::string(expected_soname));
  }
  const auto* const soname_end = static_cast<const char*>(terminator);
  std::string soname(
      soname_begin,
      static_cast<std::size_t>(soname_end - soname_begin));
  return {information, std::move(soname)};
}

[[nodiscard]] RuntimeFingerprint::LoadedLibraryIdentity
collect_loaded_library_identity(
    void* const provider_symbol,
    const std::string_view expected_soname) {
  const auto [information, actual_soname] =
      loaded_provider_info(provider_symbol, expected_soname);
  if (actual_soname != expected_soname) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded cuBLAS provider ELF SONAME differs from required SONAME " +
            std::string(expected_soname) + ": " + actual_soname);
  }
  const std::filesystem::path loader_path(information.dli_fname);
  if (loader_path.filename() != expected_soname) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded cuBLAS provider name differs from required SONAME " +
            std::string(expected_soname) + ": " +
            loader_path.filename().string());
  }
  std::error_code canonical_error;
  const std::filesystem::path canonical_path =
      std::filesystem::canonical(loader_path, canonical_error);
  if (canonical_error || canonical_path.empty()) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "unable to canonicalize loaded " + std::string(expected_soname) +
            ": " + canonical_error.message());
  }
  const std::string path_bytes = canonical_path.native();
  FileDescriptor descriptor(::open(
      path_bytes.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "unable to open loaded " + std::string(expected_soname) +
            " through its canonical nonsymlink path: " +
            std::string(std::strerror(errno)));
  }
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0 ||
      !S_ISREG(before.st_mode) || before.st_size <= 0) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded " + std::string(expected_soname) +
            " is not a nonempty regular file");
  }
  const MappedFileIdentity mapping = mapped_file_identity(
      reinterpret_cast<std::uintptr_t>(provider_symbol));
  if (before.st_dev != mapping.device || before.st_ino != mapping.inode) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded " + std::string(expected_soname) +
            " mapping does not match the resolved artifact device/inode");
  }
  const std::string digest = sha256_descriptor(descriptor.get());
  struct stat after {};
  if (::fstat(descriptor.get(), &after) != 0 ||
      !same_stable_file(before, after)) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "loaded " + std::string(expected_soname) +
            " changed while it was being authenticated");
  }
  return RuntimeFingerprint::LoadedLibraryIdentity{
      .soname = std::string(expected_soname),
      .size_bytes = static_cast<std::uint64_t>(before.st_size),
      .sha256 = digest,
  };
}

}  // namespace

std::string_view to_string(
    const RuntimeFingerprintStage stage) noexcept {
  switch (stage) {
    case RuntimeFingerprintStage::operating_system:
      return "operating_system";
    case RuntimeFingerprintStage::architecture:
      return "architecture";
    case RuntimeFingerprintStage::cuda_runtime:
      return "cuda_runtime";
    case RuntimeFingerprintStage::cublas_runtime:
      return "cublas_runtime";
    case RuntimeFingerprintStage::tensorrt_runtime:
      return "tensorrt_runtime";
    case RuntimeFingerprintStage::nvidia_driver:
      return "nvidia_driver";
    case RuntimeFingerprintStage::cuda_device:
      return "cuda_device";
  }
  return "unknown";
}

RuntimeFingerprintError::RuntimeFingerprintError(
    const RuntimeFingerprintStage stage,
    std::string detail)
    : std::runtime_error(build_error_message(stage, detail)),
      stage_(stage),
      detail_(std::move(detail)) {}

RuntimeFingerprintStage RuntimeFingerprintError::stage() const noexcept {
  return stage_;
}

const std::string& RuntimeFingerprintError::detail() const noexcept {
  return detail_;
}

std::pair<std::string, std::string>
collect_linux_distribution_fingerprint(
    const std::filesystem::path& os_release_path) {
  std::ifstream stream(os_release_path);
  if (!stream) {
    fail(
        RuntimeFingerprintStage::operating_system,
        "unable to open " + os_release_path.string() + ": " +
            std::strerror(errno));
  }
  std::map<std::string, std::string> values;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos || separator == 0) {
      fail(
          RuntimeFingerprintStage::operating_system,
          "os-release line " + std::to_string(line_number) +
              " is not KEY=VALUE");
    }
    const std::string key = line.substr(0, separator);
    for (const char character : key) {
      if (!((character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_')) {
        fail(
            RuntimeFingerprintStage::operating_system,
            "os-release line " + std::to_string(line_number) +
                " has an invalid key");
      }
    }
    const std::string value = unquote_os_release_value(
        std::string_view(line).substr(separator + 1),
        line_number);
    if (!values.emplace(key, value).second) {
      fail(
          RuntimeFingerprintStage::operating_system,
          "os-release repeats field " + key);
    }
  }
  if (!stream.eof()) {
    fail(
        RuntimeFingerprintStage::operating_system,
        "failed while reading " + os_release_path.string());
  }
  const std::string distribution_id =
      require_distribution_value(values, "ID");
  const std::string version_id =
      require_distribution_value(values, "VERSION_ID");
  if (distribution_id != "ubuntu") {
    fail(
        RuntimeFingerprintStage::operating_system,
        "only Ubuntu is accepted, got ID=" + distribution_id);
  }
  return {"linux", distribution_id + "-" + version_id};
}

RuntimeFingerprint::CublasIdentity collect_cublas_runtime_identity(
    void* const authenticated_plugin_handle) {
  void* const create_symbol = resolve_plugin_dependency_symbol(
      authenticated_plugin_handle, "cublasCreate_v2");
  void* const destroy_symbol = resolve_plugin_dependency_symbol(
      authenticated_plugin_handle, "cublasDestroy_v2");
  void* const version_symbol = resolve_plugin_dependency_symbol(
      authenticated_plugin_handle, "cublasGetVersion_v2");
  void* const lt_version_symbol = resolve_plugin_dependency_symbol(
      authenticated_plugin_handle, "cublasLtGetVersion");

  using CreateFunction = cublasStatus_t (*)(cublasHandle_t*);
  using DestroyFunction = cublasStatus_t (*)(cublasHandle_t);
  using VersionFunction = cublasStatus_t (*)(cublasHandle_t, int*);
  using LtVersionFunction = std::size_t (*)();
  const CreateFunction create =
      function_from_symbol<CreateFunction>(create_symbol);
  const DestroyFunction destroy =
      function_from_symbol<DestroyFunction>(destroy_symbol);
  const VersionFunction version =
      function_from_symbol<VersionFunction>(version_symbol);
  const LtVersionFunction lt_version =
      function_from_symbol<LtVersionFunction>(lt_version_symbol);

  cublasHandle_t handle = nullptr;
  if (create(&handle) != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "cublasCreate_v2 failed while collecting runtime identity");
  }
  int api_version = 0;
  const cublasStatus_t version_status = version(handle, &api_version);
  const cublasStatus_t destroy_status = destroy(handle);
  if (version_status != CUBLAS_STATUS_SUCCESS || api_version <= 0) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "cublasGetVersion_v2 returned an invalid version");
  }
  if (destroy_status != CUBLAS_STATUS_SUCCESS) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "cublasDestroy_v2 failed after runtime identity collection");
  }
  const std::size_t lt_api_version = lt_version();
  if (lt_api_version == 0U ||
      lt_api_version > std::numeric_limits<std::uint32_t>::max() ||
      static_cast<std::size_t>(api_version) != lt_api_version) {
    fail(
        RuntimeFingerprintStage::cublas_runtime,
        "cuBLAS and cuBLASLt API versions differ or are invalid");
  }

  return RuntimeFingerprint::CublasIdentity{
      .api_version_integer = static_cast<std::uint32_t>(api_version),
      .library = collect_loaded_library_identity(
          version_symbol, "libcublas.so.13"),
      .lt_library = collect_loaded_library_identity(
          lt_version_symbol, "libcublasLt.so.13"),
  };
}

RuntimeFingerprint collect_runtime_fingerprint(
    const std::int32_t cuda_device_index,
    const std::uint32_t plugin_abi_version,
    const RuntimeFingerprint::CublasIdentity& cublas_identity) {
  if (cuda_device_index < 0) {
    fail(
        RuntimeFingerprintStage::cuda_device,
        "CUDA device index must be non-negative");
  }
  const auto [os_name, os_version] =
      collect_linux_distribution_fingerprint("/etc/os-release");

  struct utsname system_name {};
  if (::uname(&system_name) != 0) {
    fail(
        RuntimeFingerprintStage::architecture,
        "uname failed: " + std::string(std::strerror(errno)));
  }
  const std::string architecture = system_name.machine;
  if (architecture != "aarch64" && architecture != "x86_64") {
    fail(
        RuntimeFingerprintStage::architecture,
        "unsupported machine architecture: " + architecture);
  }

  int runtime_version = 0;
  const cudaError_t runtime_status =
      cudaRuntimeGetVersion(&runtime_version);
  if (runtime_status != cudaSuccess) {
    fail(
        RuntimeFingerprintStage::cuda_runtime,
        "cudaRuntimeGetVersion failed: " +
            std::string(cudaGetErrorString(runtime_status)));
  }

  cudaDeviceProp device_properties{};
  const cudaError_t properties_status =
      cudaGetDeviceProperties(&device_properties, cuda_device_index);
  if (properties_status != cudaSuccess) {
    fail(
        RuntimeFingerprintStage::cuda_device,
        "cudaGetDeviceProperties failed: " +
            std::string(cudaGetErrorString(properties_status)));
  }
  if (device_properties.name[0] == '\0' ||
      device_properties.major < 0 || device_properties.minor < 0) {
    fail(
        RuntimeFingerprintStage::cuda_device,
        "CUDA returned invalid device properties");
  }

  Endianness endianness;
  if constexpr (std::endian::native == std::endian::little) {
    endianness = Endianness::little;
  } else if constexpr (std::endian::native == std::endian::big) {
    endianness = Endianness::big;
  } else {
    fail(
        RuntimeFingerprintStage::architecture,
        "mixed-endian hosts are unsupported");
  }

  return RuntimeFingerprint{
      .os_name = os_name,
      .os_version = os_version,
      .architecture = architecture,
      .endianness = endianness,
      .cuda_version = format_cuda_runtime_version(runtime_version),
      .cublas = cublas_identity,
      .tensorrt_version = format_tensorrt_runtime_version(),
      .driver_version = collect_driver_version(),
      .gpu_name = device_properties.name,
      .gpu_compute_capability =
          std::to_string(device_properties.major) + "." +
          std::to_string(device_properties.minor),
      .plugin_abi_version = plugin_abi_version,
  };
}

}  // namespace magpie_tts_rt

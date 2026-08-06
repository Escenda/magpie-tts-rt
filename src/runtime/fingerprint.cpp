#include "runtime/fingerprint.hpp"

#include <bit>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string_view>
#include <system_error>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <sys/utsname.h>

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

RuntimeFingerprint collect_runtime_fingerprint(
    const std::int32_t cuda_device_index,
    const std::uint32_t plugin_abi_version) {
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

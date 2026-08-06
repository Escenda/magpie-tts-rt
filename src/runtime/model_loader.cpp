#include "runtime/model_loader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include <dlfcn.h>
#include <sys/mman.h>

#include "magpie_tts_rt/magpie_tts_rt_plugin.h"
#include "runtime/fingerprint.hpp"

namespace magpie_tts_rt {
namespace {

using GetPluginApiFunction =
    mtt_plugin_status_t (*)(mtt_plugin_api_v1_t*);

inline constexpr std::array<std::string_view, MTT_PLUGIN_CREATOR_COUNT_V1>
    kRequiredPluginCreators{
        "MagpieLocalARSampling",
        "MagpieLocalAREos",
        "MagpieLayerNorm",
        "MagpieGeluTanh",
        "MagpieSoftmax",
    };

struct ProcessPluginOwner {
  std::mutex mutex;
  void* handle{nullptr};
  std::string sha256;
  std::uint32_t abi_version{0};
  struct CreatorIdentity {
    std::string name;
    std::string version;
    std::string plugin_namespace;

    bool operator==(const CreatorIdentity&) const = default;
  };
  std::array<CreatorIdentity, MTT_PLUGIN_CREATOR_COUNT_V1>
      creators;
};

// This object intentionally owns the authenticated mapping for the process
// lifetime. TensorRT's registry retains creator pointers globally, so
// releasing or replacing the mapping while TensorRT remains loaded would
// leave dangling registry entries. RTLD_NODELETE independently enforces the
// same lifetime at the dynamic-loader boundary.
[[nodiscard]] ProcessPluginOwner& process_plugin_owner() {
  static ProcessPluginOwner owner;
  return owner;
}

#if defined(MAGPIE_TTS_RT_PLUGIN_OWNER_TESTING)
std::atomic<PluginOwnerTestFault> plugin_owner_test_fault{
    PluginOwnerTestFault::none};

void inject_plugin_owner_test_fault() {
  if (plugin_owner_test_fault.exchange(
          PluginOwnerTestFault::none,
          std::memory_order_acq_rel) ==
      PluginOwnerTestFault::
          after_preparation_before_registration) {
    throw std::bad_alloc();
  }
}
#endif

[[nodiscard]] std::string plugin_error_message(
    const PluginLoadErrorCode code,
    const std::string_view detail) {
  return "authenticated TensorRT plugin load failed [code=" +
         std::string(to_string(code)) + "]: " + std::string(detail);
}

[[nodiscard]] std::string engine_error_message(
    const EngineLoadErrorCode code,
    const std::string_view engine_name,
    const std::string_view tensor_name,
    const std::string_view detail) {
  std::string message =
      "TensorRT engine contract failed [code=" +
      std::string(to_string(code)) + ", engine=" +
      std::string(engine_name);
  if (!tensor_name.empty()) {
    message += ", tensor=" + std::string(tensor_name);
  }
  message += "]: " + std::string(detail);
  return message;
}

[[noreturn]] void fail_plugin(
    const PluginLoadErrorCode code,
    const std::string& detail) {
  throw PluginLoadError(code, detail);
}

[[noreturn]] void fail_engine(
    const EngineLoadErrorCode code,
    const std::string_view engine_name,
    const std::string_view tensor_name,
    const std::string& detail) {
  throw EngineLoadError(
      code,
      std::string(engine_name),
      std::string(tensor_name),
      detail);
}

[[nodiscard]] const VerifiedBundleArtifact& require_artifact(
    const VerifiedRuntimeBundle& bundle,
    const BundleArtifactKind kind,
    const std::string_view logical_name,
    const EngineLoadErrorCode missing_code) {
  const auto found = std::find_if(
      bundle.artifacts.begin(),
      bundle.artifacts.end(),
      [&](const VerifiedBundleArtifact& artifact) {
        return artifact.kind == kind &&
               artifact.logical_name == logical_name;
      });
  if (found == bundle.artifacts.end()) {
    fail_engine(
        missing_code,
        logical_name,
        "",
        "the verified bundle does not contain the declared artifact");
  }
  return *found;
}

[[nodiscard]] const VerifiedBundleArtifact& require_plugin_artifact(
    const VerifiedRuntimeBundle& bundle) {
  const auto found = std::find_if(
      bundle.artifacts.begin(),
      bundle.artifacts.end(),
      [](const VerifiedBundleArtifact& artifact) {
        return artifact.kind == BundleArtifactKind::plugin;
      });
  if (found == bundle.artifacts.end()) {
    fail_plugin(
        PluginLoadErrorCode::missing_artifact,
        "the verified bundle contains no plugin artifact");
  }
  return *found;
}

[[nodiscard]] std::string descriptor_path(const int file_descriptor) {
  if (file_descriptor < 0) {
    fail_plugin(
        PluginLoadErrorCode::missing_artifact,
        "the verified plugin descriptor is closed");
  }
  return "/proc/self/fd/" + std::to_string(file_descriptor);
}

template <std::size_t Capacity>
[[nodiscard]] std::string fixed_string(
    const char (&value)[Capacity],
    const std::string_view field) {
  const void* terminator = std::memchr(value, '\0', Capacity);
  if (terminator == nullptr) {
    fail_plugin(
        PluginLoadErrorCode::invalid_creator_contract,
        std::string(field) + " is not NUL-terminated");
  }
  const auto length =
      static_cast<const char*>(terminator) - value;
  if (length == 0) {
    fail_plugin(
        PluginLoadErrorCode::invalid_creator_contract,
        std::string(field) + " is empty");
  }
  return std::string(value, static_cast<std::size_t>(length));
}

[[nodiscard]] ProcessPluginOwner::CreatorIdentity
authenticate_creator(
    const mtt_plugin_creator_v1_t& creator,
    const std::string_view expected_name) {
  if (creator.struct_size != sizeof(mtt_plugin_creator_v1_t) ||
      creator.abi_version != MTT_PLUGIN_ABI_VERSION_1) {
    fail_plugin(
        PluginLoadErrorCode::abi_mismatch,
        "plugin creator descriptor does not match ABI version 1");
  }
  const std::string name = fixed_string(creator.name, "creator.name");
  const std::string version =
      fixed_string(creator.version, "creator.version");
  const std::string plugin_namespace =
      fixed_string(creator.plugin_namespace, "creator.plugin_namespace");
  if (name != expected_name || version != "1" ||
      plugin_namespace != "magpie_tts_rt") {
    fail_plugin(
        PluginLoadErrorCode::invalid_creator_contract,
        "expected creator " + std::string(expected_name) +
            " version 1 in namespace magpie_tts_rt, got " + name +
            " version " + version + " in namespace " + plugin_namespace);
  }
  return ProcessPluginOwner::CreatorIdentity{
      .name = name,
      .version = version,
      .plugin_namespace = plugin_namespace,
  };
}

void require_manifest_plugin_contract(
    const VerifiedRuntimeBundle& bundle,
    const std::uint32_t abi_version) {
  if (bundle.manifest.artifacts.plugin.abi_version != abi_version ||
      bundle.manifest.runtime.plugin_abi_version != abi_version) {
    fail_plugin(
        PluginLoadErrorCode::abi_mismatch,
        "plugin ABI differs from the authenticated manifest");
  }
  if (bundle.manifest.local_ar.sampling_plugin_name !=
      kRequiredPluginCreators.front()) {
    fail_plugin(
        PluginLoadErrorCode::invalid_creator_contract,
        "local_ar.sampling_plugin_name does not select the authenticated creator");
  }
}

[[nodiscard]] GetPluginApiFunction load_get_api(void* handle) {
  static_cast<void>(::dlerror());
  void* symbol = ::dlsym(handle, "mtt_plugin_get_api_v1");
  const char* error = ::dlerror();
  if (error != nullptr || symbol == nullptr) {
    fail_plugin(
        PluginLoadErrorCode::missing_api_symbol,
        error == nullptr ? "mtt_plugin_get_api_v1 is null"
                         : std::string(error));
  }
  GetPluginApiFunction function = nullptr;
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

[[nodiscard]] mtt_plugin_api_v1_t load_plugin_api(void* handle) {
  const GetPluginApiFunction get_api = load_get_api(handle);
  mtt_plugin_api_v1_t api{};
  api.struct_size = sizeof(api);
  api.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  const mtt_plugin_status_t api_status = get_api(&api);
  if (api_status != MTT_PLUGIN_STATUS_OK ||
      api.struct_size != sizeof(api) ||
      api.abi_version != MTT_PLUGIN_ABI_VERSION_1 ||
      api.creator_count != MTT_PLUGIN_CREATOR_COUNT_V1 ||
      api.register_plugins == nullptr) {
    fail_plugin(
        PluginLoadErrorCode::abi_mismatch,
        "mtt_plugin_get_api_v1 did not return the exact ABI v1 contract");
  }
  return api;
}

[[nodiscard]] std::array<
    ProcessPluginOwner::CreatorIdentity,
    MTT_PLUGIN_CREATOR_COUNT_V1>
authenticate_creators(const mtt_plugin_api_v1_t& api) {
  std::array<
      ProcessPluginOwner::CreatorIdentity,
      MTT_PLUGIN_CREATOR_COUNT_V1>
      creators;
  for (std::size_t index = 0;
       index < kRequiredPluginCreators.size();
       ++index) {
    creators[index] = authenticate_creator(
        api.creators[index], kRequiredPluginCreators[index]);
  }
  return creators;
}

void require_matching_process_owner(
    const ProcessPluginOwner& owner,
    const VerifiedBundleArtifact& artifact,
    const VerifiedRuntimeBundle& bundle,
    const std::int32_t cuda_device_index) {
  if (owner.handle == nullptr) {
    fail_plugin(
        PluginLoadErrorCode::missing_artifact,
        "the process-global plugin owner is not initialized");
  }
  if (owner.sha256 != artifact.sha256) {
    fail_plugin(
        PluginLoadErrorCode::conflicting_artifact,
        "this process is already bound to plugin SHA-256 " +
            owner.sha256 + ", requested " + artifact.sha256);
  }
  const mtt_plugin_api_v1_t api =
      load_plugin_api(owner.handle);
  const auto creators = authenticate_creators(api);
  if (api.abi_version != owner.abi_version ||
      creators != owner.creators) {
    fail_plugin(
        PluginLoadErrorCode::invalid_creator_contract,
        "the process-global plugin owner no longer exposes its authenticated ordered creator contract");
  }
  require_manifest_plugin_contract(bundle, owner.abi_version);
  require_exact_runtime_fingerprint(
      bundle.manifest.runtime,
      collect_runtime_fingerprint(
          cuda_device_index, owner.abi_version));
}

[[nodiscard]] nvinfer1::DataType expected_data_type(
    const TensorDataType data_type) {
  switch (data_type) {
    case TensorDataType::fp32:
      return nvinfer1::DataType::kFLOAT;
    case TensorDataType::fp16:
      return nvinfer1::DataType::kHALF;
    case TensorDataType::bf16:
      return nvinfer1::DataType::kBF16;
    case TensorDataType::int64:
      return nvinfer1::DataType::kINT64;
    case TensorDataType::int32:
      return nvinfer1::DataType::kINT32;
    case TensorDataType::int8:
      return nvinfer1::DataType::kINT8;
    case TensorDataType::uint8:
      return nvinfer1::DataType::kUINT8;
    case TensorDataType::boolean:
      return nvinfer1::DataType::kBOOL;
  }
  fail_engine(
      EngineLoadErrorCode::data_type_mismatch,
      "",
      "",
      "manifest contains an unknown tensor data type");
}

[[nodiscard]] bool dimensions_equal(
    const nvinfer1::Dims& actual,
    const std::vector<std::int64_t>& expected) {
  if (actual.nbDims < 0 ||
      static_cast<std::size_t>(actual.nbDims) != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (static_cast<std::int64_t>(actual.d[index]) != expected[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string format_dimensions(
    const nvinfer1::Dims& dimensions) {
  std::ostringstream output;
  output << '[';
  for (std::int32_t index = 0; index < dimensions.nbDims; ++index) {
    if (index != 0) {
      output << ',';
    }
    output << dimensions.d[index];
  }
  output << ']';
  return output.str();
}

[[nodiscard]] std::string format_dimensions(
    const std::vector<std::int64_t>& dimensions) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << dimensions[index];
  }
  output << ']';
  return output.str();
}

[[nodiscard]] const TensorSpec* find_tensor(
    const EngineManifest& manifest,
    const std::string_view name,
    nvinfer1::TensorIOMode& expected_mode) {
  const auto input = std::find_if(
      manifest.inputs.begin(),
      manifest.inputs.end(),
      [&](const TensorSpec& tensor) { return tensor.name == name; });
  if (input != manifest.inputs.end()) {
    expected_mode = nvinfer1::TensorIOMode::kINPUT;
    return &*input;
  }
  const auto output = std::find_if(
      manifest.outputs.begin(),
      manifest.outputs.end(),
      [&](const TensorSpec& tensor) { return tensor.name == name; });
  if (output != manifest.outputs.end()) {
    expected_mode = nvinfer1::TensorIOMode::kOUTPUT;
    return &*output;
  }
  return nullptr;
}

void validate_io_contract(
    const nvinfer1::ICudaEngine& engine,
    const EngineManifest& manifest) {
  const std::size_t expected_count =
      manifest.inputs.size() + manifest.outputs.size();
  const std::int32_t actual_count = engine.getNbIOTensors();
  if (actual_count < 0 ||
      static_cast<std::size_t>(actual_count) != expected_count) {
    fail_engine(
        EngineLoadErrorCode::io_tensor_count_mismatch,
        manifest.name,
        "",
        "expected " + std::to_string(expected_count) +
            " I/O tensors, got " + std::to_string(actual_count));
  }

  std::set<std::string, std::less<>> observed_names;
  for (std::int32_t index = 0; index < actual_count; ++index) {
    const char* raw_name = engine.getIOTensorName(index);
    if (raw_name == nullptr || raw_name[0] == '\0') {
      fail_engine(
          EngineLoadErrorCode::unknown_io_tensor,
          manifest.name,
          "",
          "TensorRT returned a null or empty I/O tensor name");
    }
    const std::string name(raw_name);
    if (!observed_names.emplace(name).second) {
      fail_engine(
          EngineLoadErrorCode::unknown_io_tensor,
          manifest.name,
          name,
          "TensorRT repeated an I/O tensor name");
    }

    nvinfer1::TensorIOMode expected_mode =
        nvinfer1::TensorIOMode::kNONE;
    const TensorSpec* expected =
        find_tensor(manifest, name, expected_mode);
    if (expected == nullptr) {
      fail_engine(
          EngineLoadErrorCode::unknown_io_tensor,
          manifest.name,
          name,
          "tensor is not declared by the authenticated manifest");
    }
    if (engine.getTensorIOMode(raw_name) != expected_mode) {
      fail_engine(
          EngineLoadErrorCode::io_mode_mismatch,
          manifest.name,
          name,
          "input/output mode differs from the authenticated manifest");
    }
    if (engine.getTensorDataType(raw_name) !=
        expected_data_type(expected->dtype)) {
      fail_engine(
          EngineLoadErrorCode::data_type_mismatch,
          manifest.name,
          name,
          "data type differs from the authenticated manifest");
    }
    const nvinfer1::Dims actual_shape =
        engine.getTensorShape(raw_name);
    if (!dimensions_equal(actual_shape, expected->shape)) {
      fail_engine(
          EngineLoadErrorCode::shape_mismatch,
          manifest.name,
          name,
          "expected " + format_dimensions(expected->shape) + ", got " +
              format_dimensions(actual_shape));
    }
    const nvinfer1::TensorLocation expected_location =
        expected->location == TensorMemoryLocation::device
            ? nvinfer1::TensorLocation::kDEVICE
            : nvinfer1::TensorLocation::kHOST;
    if (engine.getTensorLocation(raw_name) != expected_location) {
      fail_engine(
          EngineLoadErrorCode::unsupported_tensor_location,
          manifest.name,
          name,
          "tensor location differs from the authenticated manifest");
    }
    if (engine.isShapeInferenceIO(raw_name) !=
        expected->shape_inference_io) {
      fail_engine(
          EngineLoadErrorCode::shape_inference_io,
          manifest.name,
          name,
          "shape-inference I/O flag differs from the authenticated manifest");
    }
  }
}

[[nodiscard]] const TensorShapeRange& require_shape_range(
    const OptimizationProfile& profile,
    const std::string_view tensor_name,
    const std::string_view engine_name) {
  const auto found = std::find_if(
      profile.input_shapes.begin(),
      profile.input_shapes.end(),
      [&](const TensorShapeRange& range) {
        return range.tensor_name == tensor_name;
      });
  if (found == profile.input_shapes.end()) {
    fail_engine(
        EngineLoadErrorCode::profile_shape_mismatch,
        engine_name,
        tensor_name,
        "the authenticated profile omits an engine input");
  }
  return *found;
}

[[nodiscard]] const TensorValueRange& require_value_range(
    const OptimizationProfile& profile,
    const std::string_view tensor_name,
    const std::string_view engine_name) {
  const auto found = std::find_if(
      profile.input_values.begin(),
      profile.input_values.end(),
      [&](const TensorValueRange& range) {
        return range.tensor_name == tensor_name;
      });
  if (found == profile.input_values.end()) {
    fail_engine(
        EngineLoadErrorCode::profile_shape_mismatch,
        engine_name,
        tensor_name,
        "the authenticated profile omits a shape-input value range");
  }
  return *found;
}

void validate_profile_contract(
    const nvinfer1::ICudaEngine& engine,
    const EngineManifest& manifest) {
  const std::int32_t actual_profiles =
      engine.getNbOptimizationProfiles();
  if (actual_profiles < 0 ||
      static_cast<std::size_t>(actual_profiles) !=
          manifest.profiles.size()) {
    fail_engine(
        EngineLoadErrorCode::profile_count_mismatch,
        manifest.name,
        "",
        "expected " + std::to_string(manifest.profiles.size()) +
            " optimization profiles, got " +
            std::to_string(actual_profiles));
  }

  for (std::int32_t profile_index = 0;
       profile_index < actual_profiles;
       ++profile_index) {
    const OptimizationProfile& profile =
        manifest.profiles.at(static_cast<std::size_t>(profile_index));
    const std::size_t dynamic_input_count =
        static_cast<std::size_t>(std::count_if(
            manifest.inputs.begin(),
            manifest.inputs.end(),
            [](const TensorSpec& input) {
              return std::find(
                         input.shape.begin(),
                         input.shape.end(),
                         -1) != input.shape.end();
            }));
    const std::size_t shape_input_count =
        static_cast<std::size_t>(std::count_if(
            manifest.inputs.begin(),
            manifest.inputs.end(),
            [](const TensorSpec& input) {
              return input.shape_inference_io;
            }));
    if (profile.input_shapes.size() != dynamic_input_count ||
        profile.input_values.size() != shape_input_count) {
      fail_engine(
          EngineLoadErrorCode::profile_shape_mismatch,
          manifest.name,
          "",
          "profile dynamic-shape or shape-input set differs from the authenticated manifest");
    }
    for (const TensorSpec& input : manifest.inputs) {
      if (std::find(input.shape.begin(), input.shape.end(), -1) ==
          input.shape.end()) {
        continue;
      }
      const TensorShapeRange& expected =
          require_shape_range(profile, input.name, manifest.name);
      const nvinfer1::Dims minimum = engine.getProfileShape(
          input.name.c_str(),
          profile_index,
          nvinfer1::OptProfileSelector::kMIN);
      const nvinfer1::Dims optimum = engine.getProfileShape(
          input.name.c_str(),
          profile_index,
          nvinfer1::OptProfileSelector::kOPT);
      const nvinfer1::Dims maximum = engine.getProfileShape(
          input.name.c_str(),
          profile_index,
          nvinfer1::OptProfileSelector::kMAX);
      if (!dimensions_equal(minimum, expected.minimum) ||
          !dimensions_equal(optimum, expected.optimum) ||
          !dimensions_equal(maximum, expected.maximum)) {
        fail_engine(
            EngineLoadErrorCode::profile_shape_mismatch,
            manifest.name,
            input.name,
            "expected min/opt/max " +
                format_dimensions(expected.minimum) + "/" +
                format_dimensions(expected.optimum) + "/" +
                format_dimensions(expected.maximum) + ", got " +
                format_dimensions(minimum) + "/" +
                format_dimensions(optimum) + "/" +
                format_dimensions(maximum));
      }
    }
    for (const TensorSpec& input : manifest.inputs) {
      if (!input.shape_inference_io) {
        continue;
      }
      const TensorValueRange& expected =
          require_value_range(profile, input.name, manifest.name);
      const std::int64_t* minimum =
          engine.getProfileTensorValuesV2(
              input.name.c_str(),
              profile_index,
              nvinfer1::OptProfileSelector::kMIN);
      const std::int64_t* optimum =
          engine.getProfileTensorValuesV2(
              input.name.c_str(),
              profile_index,
              nvinfer1::OptProfileSelector::kOPT);
      const std::int64_t* maximum =
          engine.getProfileTensorValuesV2(
              input.name.c_str(),
              profile_index,
              nvinfer1::OptProfileSelector::kMAX);
      if (minimum == nullptr || optimum == nullptr ||
          maximum == nullptr) {
        fail_engine(
            EngineLoadErrorCode::profile_shape_mismatch,
            manifest.name,
            input.name,
            "TensorRT omitted authenticated shape-input profile values");
      }
      std::uint64_t value_count = 1;
      for (const std::int64_t dimension : input.shape) {
        if (dimension <= 0 ||
            value_count >
                std::numeric_limits<std::uint64_t>::max() /
                    static_cast<std::uint64_t>(dimension)) {
          fail_engine(
              EngineLoadErrorCode::profile_shape_mismatch,
              manifest.name,
              input.name,
              "shape-input value count is not representable");
        }
        value_count *= static_cast<std::uint64_t>(dimension);
      }
      if (expected.minimum.size() != value_count ||
          expected.optimum.size() != value_count ||
          expected.maximum.size() != value_count ||
          !std::equal(
              expected.minimum.begin(),
              expected.minimum.end(),
              minimum) ||
          !std::equal(
              expected.optimum.begin(),
              expected.optimum.end(),
              optimum) ||
          !std::equal(
              expected.maximum.begin(),
              expected.maximum.end(),
              maximum)) {
        fail_engine(
            EngineLoadErrorCode::profile_shape_mismatch,
            manifest.name,
            input.name,
            "shape-input profile values differ from the authenticated manifest");
      }
    }
  }
}

class ReadOnlyMapping final {
 public:
  ReadOnlyMapping(
      const int file_descriptor,
      const std::uint64_t size_bytes,
      const std::string_view engine_name)
      : size_bytes_(size_bytes) {
    if (size_bytes == 0 ||
        size_bytes >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
      fail_engine(
          EngineLoadErrorCode::artifact_too_large,
          engine_name,
          "",
          "engine snapshot size does not fit this process");
    }
    address_ = ::mmap(
        nullptr,
        static_cast<std::size_t>(size_bytes_),
        PROT_READ,
        MAP_PRIVATE,
        file_descriptor,
        0);
    if (address_ == MAP_FAILED) {
      address_ = nullptr;
      fail_engine(
          EngineLoadErrorCode::map_failed,
          engine_name,
          "",
          "mmap failed: " + std::string(std::strerror(errno)));
    }
  }

  ReadOnlyMapping(const ReadOnlyMapping&) = delete;
  ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;

  ~ReadOnlyMapping() {
    if (address_ != nullptr) {
      static_cast<void>(::munmap(
          address_, static_cast<std::size_t>(size_bytes_)));
    }
  }

  [[nodiscard]] const void* data() const noexcept { return address_; }
  [[nodiscard]] std::size_t size() const noexcept {
    return static_cast<std::size_t>(size_bytes_);
  }

 private:
  void* address_{nullptr};
  std::uint64_t size_bytes_;
};

}  // namespace

std::string_view to_string(const PluginLoadErrorCode code) noexcept {
  switch (code) {
    case PluginLoadErrorCode::missing_artifact:
      return "missing_artifact";
    case PluginLoadErrorCode::conflicting_artifact:
      return "conflicting_artifact";
    case PluginLoadErrorCode::dynamic_loader_error:
      return "dynamic_loader_error";
    case PluginLoadErrorCode::missing_api_symbol:
      return "missing_api_symbol";
    case PluginLoadErrorCode::abi_mismatch:
      return "abi_mismatch";
    case PluginLoadErrorCode::invalid_creator_contract:
      return "invalid_creator_contract";
    case PluginLoadErrorCode::registration_failed:
      return "registration_failed";
  }
  return "unknown";
}

PluginLoadError::PluginLoadError(
    const PluginLoadErrorCode code,
    std::string detail)
    : std::runtime_error(plugin_error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

PluginLoadErrorCode PluginLoadError::code() const noexcept {
  return code_;
}

const std::string& PluginLoadError::detail() const noexcept {
  return detail_;
}

std::string_view to_string(const EngineLoadErrorCode code) noexcept {
  switch (code) {
    case EngineLoadErrorCode::missing_artifact:
      return "missing_artifact";
    case EngineLoadErrorCode::artifact_too_large:
      return "artifact_too_large";
    case EngineLoadErrorCode::map_failed:
      return "map_failed";
    case EngineLoadErrorCode::deserialize_failed:
      return "deserialize_failed";
    case EngineLoadErrorCode::io_tensor_count_mismatch:
      return "io_tensor_count_mismatch";
    case EngineLoadErrorCode::unknown_io_tensor:
      return "unknown_io_tensor";
    case EngineLoadErrorCode::io_mode_mismatch:
      return "io_mode_mismatch";
    case EngineLoadErrorCode::data_type_mismatch:
      return "data_type_mismatch";
    case EngineLoadErrorCode::shape_mismatch:
      return "shape_mismatch";
    case EngineLoadErrorCode::profile_count_mismatch:
      return "profile_count_mismatch";
    case EngineLoadErrorCode::profile_shape_mismatch:
      return "profile_shape_mismatch";
    case EngineLoadErrorCode::unsupported_tensor_location:
      return "unsupported_tensor_location";
    case EngineLoadErrorCode::shape_inference_io:
      return "shape_inference_io";
  }
  return "unknown";
}

EngineLoadError::EngineLoadError(
    const EngineLoadErrorCode code,
    std::string engine_name,
    std::string tensor_name,
    std::string detail)
    : std::runtime_error(
          engine_error_message(
              code, engine_name, tensor_name, detail)),
      code_(code),
      engine_name_(std::move(engine_name)),
      tensor_name_(std::move(tensor_name)),
      detail_(std::move(detail)) {}

EngineLoadErrorCode EngineLoadError::code() const noexcept {
  return code_;
}

const std::string& EngineLoadError::engine_name() const noexcept {
  return engine_name_;
}

const std::string& EngineLoadError::tensor_name() const noexcept {
  return tensor_name_;
}

const std::string& EngineLoadError::detail() const noexcept {
  return detail_;
}

void RuntimePluginState::authenticate_and_register(
    const VerifiedRuntimeBundle& bundle,
    const std::int32_t cuda_device_index) {
  const VerifiedBundleArtifact& artifact =
      require_plugin_artifact(bundle);
  std::scoped_lock runtime_lock(mutex_);
  ProcessPluginOwner& owner = process_plugin_owner();
  if (authenticated_) {
    if (sha256_ != artifact.sha256) {
      fail_plugin(
          PluginLoadErrorCode::conflicting_artifact,
          "this runtime is already bound to plugin SHA-256 " + sha256_ +
              ", requested " + artifact.sha256);
    }
    std::scoped_lock process_lock(owner.mutex);
    require_matching_process_owner(
        owner, artifact, bundle, cuda_device_index);
    if (abi_version_ != owner.abi_version) {
      fail_plugin(
          PluginLoadErrorCode::abi_mismatch,
          "runtime plugin ABI differs from the process-global owner");
    }
    return;
  }

  std::scoped_lock process_lock(owner.mutex);
  if (owner.handle != nullptr) {
    require_matching_process_owner(
        owner, artifact, bundle, cuda_device_index);
    authenticated_ = true;
    sha256_ = owner.sha256;
    abi_version_ = owner.abi_version;
    return;
  }

  const std::string path =
      descriptor_path(artifact.verified_file_descriptor());
  void* candidate = ::dlopen(
      path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (candidate == nullptr) {
    const char* error = ::dlerror();
    fail_plugin(
        PluginLoadErrorCode::dynamic_loader_error,
        error == nullptr ? "dlopen failed without a diagnostic"
                         : std::string(error));
  }

  try {
    // Prepare every allocating owner field before creator registration. Once
    // register_plugins() succeeds, TensorRT retains creator pointers and the
    // process-global owner commit must be impossible to interrupt.
    std::string prepared_process_sha256 = artifact.sha256;
    std::string prepared_runtime_sha256 = artifact.sha256;
    static_assert(noexcept(
        owner.sha256 = std::move(prepared_process_sha256)));
    static_assert(noexcept(
        sha256_ = std::move(prepared_runtime_sha256)));

    const mtt_plugin_api_v1_t api = load_plugin_api(candidate);
    require_manifest_plugin_contract(bundle, api.abi_version);
    auto prepared_creators = authenticate_creators(api);
    static_assert(noexcept(
        owner.creators = std::move(prepared_creators)));

    require_exact_runtime_fingerprint(
        bundle.manifest.runtime,
        collect_runtime_fingerprint(
            cuda_device_index, api.abi_version));

#if defined(MAGPIE_TTS_RT_PLUGIN_OWNER_TESTING)
    inject_plugin_owner_test_fault();
#endif
    const mtt_plugin_status_t register_status =
        api.register_plugins();
    if (register_status != MTT_PLUGIN_STATUS_OK &&
        register_status != MTT_PLUGIN_STATUS_ALREADY_REGISTERED) {
      fail_plugin(
          PluginLoadErrorCode::registration_failed,
          "explicit creator registration returned status " +
              std::to_string(register_status));
    }
    owner.sha256 = std::move(prepared_process_sha256);
    owner.abi_version = api.abi_version;
    owner.creators = std::move(prepared_creators);
    owner.handle = candidate;
    sha256_ = std::move(prepared_runtime_sha256);
    abi_version_ = api.abi_version;
    authenticated_ = true;
  } catch (...) {
    // owner.handle is assigned last in the noexcept commit above. A null owner
    // therefore proves registration did not commit and this candidate remains
    // caller-owned.
    if (owner.handle == nullptr) {
      static_cast<void>(::dlclose(candidate));
    }
    throw;
  }
}

#if defined(MAGPIE_TTS_RT_PLUGIN_OWNER_TESTING)
void set_plugin_owner_test_fault(
    const PluginOwnerTestFault fault) noexcept {
  plugin_owner_test_fault.store(
      fault, std::memory_order_release);
}
#endif

std::uint32_t RuntimePluginState::abi_version() const {
  std::scoped_lock lock(mutex_);
  if (!authenticated_) {
    fail_plugin(
        PluginLoadErrorCode::missing_artifact,
        "runtime plugin has not been authenticated");
  }
  return abi_version_;
}

std::string RuntimePluginState::sha256() const {
  std::scoped_lock lock(mutex_);
  if (!authenticated_) {
    fail_plugin(
        PluginLoadErrorCode::missing_artifact,
        "runtime plugin has not been authenticated");
  }
  return sha256_;
}

std::vector<LoadedEngine> deserialize_verified_engines(
    nvinfer1::IRuntime& runtime,
    const VerifiedRuntimeBundle& bundle) {
  std::vector<LoadedEngine> loaded;
  loaded.reserve(bundle.manifest.engines.size());
  for (const EngineManifest& manifest : bundle.manifest.engines) {
    const VerifiedBundleArtifact& artifact = require_artifact(
        bundle,
        BundleArtifactKind::engine,
        manifest.name,
        EngineLoadErrorCode::missing_artifact);
    ReadOnlyMapping mapping(
        artifact.verified_file_descriptor(),
        artifact.size_bytes,
        manifest.name);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime.deserializeCudaEngine(mapping.data(), mapping.size()));
    if (engine == nullptr) {
      fail_engine(
          EngineLoadErrorCode::deserialize_failed,
          manifest.name,
          "",
          "TensorRT returned a null engine");
    }
    validate_io_contract(*engine, manifest);
    validate_profile_contract(*engine, manifest);
    loaded.push_back(LoadedEngine{
        .role = manifest.role,
        .name = manifest.name,
        .engine = std::move(engine),
    });
  }
  return loaded;
}

const LoadedEngine& require_loaded_engine(
    const std::vector<LoadedEngine>& engines,
    const EngineRole role) {
  const auto found = std::find_if(
      engines.begin(),
      engines.end(),
      [role](const LoadedEngine& engine) {
        return engine.role == role;
      });
  if (found == engines.end()) {
    fail_engine(
        EngineLoadErrorCode::missing_artifact,
        to_string(role),
        "",
        "required engine role was not loaded");
  }
  return *found;
}

}  // namespace magpie_tts_rt

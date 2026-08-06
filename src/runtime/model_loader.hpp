#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <NvInferRuntime.h>

#include "bundle/bundle.hpp"
#include "manifest/manifest.hpp"

namespace magpie_tts_rt {

enum class PluginLoadErrorCode {
  missing_artifact,
  conflicting_artifact,
  dynamic_loader_error,
  missing_api_symbol,
  abi_mismatch,
  invalid_creator_contract,
  registration_failed,
};

[[nodiscard]] std::string_view to_string(
    PluginLoadErrorCode code) noexcept;

class PluginLoadError final : public std::runtime_error {
 public:
  PluginLoadError(PluginLoadErrorCode code, std::string detail);

  [[nodiscard]] PluginLoadErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  PluginLoadErrorCode code_;
  std::string detail_;
};

enum class EngineLoadErrorCode {
  missing_artifact,
  artifact_too_large,
  map_failed,
  deserialize_failed,
  io_tensor_count_mismatch,
  unknown_io_tensor,
  io_mode_mismatch,
  data_type_mismatch,
  shape_mismatch,
  profile_count_mismatch,
  profile_shape_mismatch,
  unsupported_tensor_location,
  shape_inference_io,
};

[[nodiscard]] std::string_view to_string(
    EngineLoadErrorCode code) noexcept;

class EngineLoadError final : public std::runtime_error {
 public:
  EngineLoadError(
      EngineLoadErrorCode code,
      std::string engine_name,
      std::string tensor_name,
      std::string detail);

  [[nodiscard]] EngineLoadErrorCode code() const noexcept;
  [[nodiscard]] const std::string& engine_name() const noexcept;
  [[nodiscard]] const std::string& tensor_name() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  EngineLoadErrorCode code_;
  std::string engine_name_;
  std::string tensor_name_;
  std::string detail_;
};

// TensorRT keeps plugin creators in its process-global registry. The first
// runtime authenticates and registers one process-global plugin owner. Later
// runtimes may bind only to the same authenticated digest, ABI, and creator
// contract; they reuse that exact mapping instead of dlopen'ing another
// verified-bundle memfd. A different digest is an explicit process conflict.
class RuntimePluginState final {
 public:
  RuntimePluginState() = default;
  ~RuntimePluginState() = default;

  RuntimePluginState(const RuntimePluginState&) = delete;
  RuntimePluginState& operator=(const RuntimePluginState&) = delete;

  // Authenticates the plugin ABI, requires an exact host fingerprint, and
  // explicitly registers both creators before any engine is deserialized.
  void authenticate_and_register(
      const VerifiedRuntimeBundle& bundle,
      std::int32_t cuda_device_index);

  [[nodiscard]] std::uint32_t abi_version() const;
  [[nodiscard]] std::string sha256() const;

 private:
  mutable std::mutex mutex_;
  bool authenticated_{false};
  std::string sha256_;
  std::uint32_t abi_version_{0};
};

#if defined(MAGPIE_TTS_RT_PLUGIN_OWNER_TESTING)
enum class PluginOwnerTestFault {
  none,
  after_preparation_before_registration,
};

void set_plugin_owner_test_fault(
    PluginOwnerTestFault fault) noexcept;
#endif

struct LoadedEngine {
  EngineRole role;
  std::string name;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
};

// Deserializes only from the immutable descriptors retained by
// VerifiedRuntimeBundle and checks exact names, I/O modes, dtypes, dimensions,
// locations, and optimization profiles against the authenticated manifest.
[[nodiscard]] std::vector<LoadedEngine> deserialize_verified_engines(
    nvinfer1::IRuntime& runtime,
    const VerifiedRuntimeBundle& bundle);

[[nodiscard]] const LoadedEngine& require_loaded_engine(
    const std::vector<LoadedEngine>& engines,
    EngineRole role);

}  // namespace magpie_tts_rt

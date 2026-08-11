#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <stdexcept>
#include <string>
#include <utility>

#include "manifest/manifest.hpp"

namespace magpie_tts_rt {

enum class RuntimeFingerprintStage {
  operating_system,
  architecture,
  cuda_runtime,
  cublas_runtime,
  tensorrt_runtime,
  nvidia_driver,
  cuda_device,
};

[[nodiscard]] std::string_view to_string(
    RuntimeFingerprintStage stage) noexcept;

class RuntimeFingerprintError final : public std::runtime_error {
 public:
  RuntimeFingerprintError(
      RuntimeFingerprintStage stage,
      std::string detail);

  [[nodiscard]] RuntimeFingerprintStage stage() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  RuntimeFingerprintStage stage_;
  std::string detail_;
};

// Parses the exact host-side fields used by the runtime fingerprint. This is
// public only inside the C++ implementation so deterministic tests can use an
// isolated os-release fixture instead of the process host.
[[nodiscard]] std::pair<std::string, std::string>
collect_linux_distribution_fingerprint(
    const std::filesystem::path& os_release_path);

// Resolves cuBLAS and cuBLASLt only through the authenticated plugin's dynamic
// dependency scope. Each provider symbol must belong to a file-backed mapping
// whose device/inode still matches a stable descriptor. The returned hashes
// cover those exact mapped artifacts; a loader-search candidate is never used
// as a substitute.
[[nodiscard]] RuntimeFingerprint::CublasIdentity
collect_cublas_runtime_identity(void* authenticated_plugin_handle);

// Collects the active process/runtime values for one CUDA device. The plugin
// ABI is supplied only after the authenticated plugin has reported its exact
// ABI version; it is never guessed by this collector.
[[nodiscard]] RuntimeFingerprint collect_runtime_fingerprint(
    std::int32_t cuda_device_index,
    std::uint32_t plugin_abi_version,
    const RuntimeFingerprint::CublasIdentity& cublas_identity);

}  // namespace magpie_tts_rt

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "manifest/manifest.hpp"
#include "runtime/model_loader.hpp"
#include "runtime/session_workspace.hpp"

namespace magpie_tts_rt {

enum class SessionResourceErrorCode {
  invalid_manifest_limit,
  device_memory_limit_exceeded,
  execution_context_creation_failed,
  cuda_stream_creation_failed,
  cuda_event_creation_failed,
  cuda_stream_synchronization_failed,
  missing_execution_context,
};

[[nodiscard]] std::string_view to_string(
    SessionResourceErrorCode code) noexcept;

class SessionResourceError final : public std::runtime_error {
 public:
  SessionResourceError(
      SessionResourceErrorCode code,
      std::string detail);

  [[nodiscard]] SessionResourceErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  SessionResourceErrorCode code_;
  std::string detail_;
};

struct EngineExecutionContext {
  EngineRole role;
  std::string name;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
};

class SessionResources final {
 public:
  SessionResources(
      const std::vector<LoadedEngine>& engines,
      const RuntimeBundleManifest& manifest);
  ~SessionResources();

  SessionResources(const SessionResources&) = delete;
  SessionResources& operator=(const SessionResources&) = delete;

  [[nodiscard]] nvinfer1::IExecutionContext& context(
      EngineRole role);
  [[nodiscard]] cudaStream_t generation_stream() const noexcept;
  [[nodiscard]] cudaStream_t codec_stream() const noexcept;
  [[nodiscard]] cudaEvent_t codes_ready_event(
      std::size_t batch_slot) const;
  [[nodiscard]] cudaEvent_t codes_consumed_event(
      std::size_t batch_slot) const;
  [[nodiscard]] cudaEvent_t audio_ready_event() const noexcept;
  [[nodiscard]] std::uint64_t context_device_memory_bytes() const noexcept;
  [[nodiscard]] SessionWorkspace& workspace() noexcept;
  [[nodiscard]] const SessionWorkspace& workspace() const noexcept;
  void synchronize_for_teardown();

 private:
  std::vector<EngineExecutionContext> contexts_;
  cudaStream_t generation_stream_{nullptr};
  cudaStream_t codec_stream_{nullptr};
  std::array<cudaEvent_t, kGenerationBatchSlotCount>
      codes_ready_events_{};
  std::array<cudaEvent_t, kGenerationBatchSlotCount>
      codes_consumed_events_{};
  cudaEvent_t audio_ready_event_{nullptr};
  std::uint64_t context_device_memory_bytes_{0};
  std::unique_ptr<SessionWorkspace> workspace_;
  bool teardown_synchronized_{false};
};

}  // namespace magpie_tts_rt

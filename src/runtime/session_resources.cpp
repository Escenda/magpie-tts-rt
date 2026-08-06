#include "runtime/session_resources.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const SessionResourceErrorCode code,
    const std::string_view detail) {
  return "session resource creation failed [code=" +
         std::string(to_string(code)) + "]: " + std::string(detail);
}

[[noreturn]] void fail(
    const SessionResourceErrorCode code,
    const std::string& detail) {
  throw SessionResourceError(code, detail);
}

void require_cuda(
    const cudaError_t status,
    const SessionResourceErrorCode code,
    const std::string_view operation) {
  if (status != cudaSuccess) {
    fail(
        code,
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace

std::string_view to_string(
    const SessionResourceErrorCode code) noexcept {
  switch (code) {
    case SessionResourceErrorCode::invalid_manifest_limit:
      return "invalid_manifest_limit";
    case SessionResourceErrorCode::device_memory_limit_exceeded:
      return "device_memory_limit_exceeded";
    case SessionResourceErrorCode::execution_context_creation_failed:
      return "execution_context_creation_failed";
    case SessionResourceErrorCode::cuda_stream_creation_failed:
      return "cuda_stream_creation_failed";
    case SessionResourceErrorCode::cuda_event_creation_failed:
      return "cuda_event_creation_failed";
    case SessionResourceErrorCode::cuda_stream_synchronization_failed:
      return "cuda_stream_synchronization_failed";
    case SessionResourceErrorCode::missing_execution_context:
      return "missing_execution_context";
  }
  return "unknown";
}

SessionResourceError::SessionResourceError(
    const SessionResourceErrorCode code,
    std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

SessionResourceErrorCode SessionResourceError::code() const noexcept {
  return code_;
}

const std::string& SessionResourceError::detail() const noexcept {
  return detail_;
}

SessionResources::SessionResources(
    const std::vector<LoadedEngine>& engines,
    const RuntimeBundleManifest& manifest) {
  const LimitsManifest& limits = manifest.limits;
  if (engines.empty() || limits.maximum_device_memory_bytes == 0) {
    fail(
        SessionResourceErrorCode::invalid_manifest_limit,
        "engines and a positive device-memory limit are required");
  }

  for (const LoadedEngine& loaded : engines) {
    const std::int64_t context_bytes =
        loaded.engine->getDeviceMemorySizeV2();
    if (context_bytes < 0) {
      fail(
          SessionResourceErrorCode::execution_context_creation_failed,
          loaded.name +
              " returned a negative context device-memory size");
    }
    const std::uint64_t unsigned_bytes =
        static_cast<std::uint64_t>(context_bytes);
    if (unsigned_bytes >
        std::numeric_limits<std::uint64_t>::max() -
            context_device_memory_bytes_) {
      fail(
          SessionResourceErrorCode::device_memory_limit_exceeded,
          "execution-context memory sum overflowed UINT64");
    }
    context_device_memory_bytes_ += unsigned_bytes;
  }
  if (context_device_memory_bytes_ >
      limits.maximum_device_memory_bytes) {
    fail(
        SessionResourceErrorCode::device_memory_limit_exceeded,
        "TensorRT contexts require " +
            std::to_string(context_device_memory_bytes_) +
            " bytes, exceeding manifest limit " +
            std::to_string(limits.maximum_device_memory_bytes));
  }

  require_cuda(
      cudaStreamCreateWithFlags(
          &generation_stream_, cudaStreamNonBlocking),
      SessionResourceErrorCode::cuda_stream_creation_failed,
      "create generation stream");
  try {
    require_cuda(
        cudaStreamCreateWithFlags(
            &codec_stream_, cudaStreamNonBlocking),
        SessionResourceErrorCode::cuda_stream_creation_failed,
        "create codec stream");
    for (std::size_t slot = 0;
         slot < kGenerationBatchSlotCount;
         ++slot) {
      require_cuda(
          cudaEventCreateWithFlags(
              &codes_ready_events_[slot],
              cudaEventDisableTiming),
          SessionResourceErrorCode::cuda_event_creation_failed,
          "create codes-ready event");
      require_cuda(
          cudaEventCreateWithFlags(
              &codes_consumed_events_[slot],
              cudaEventDisableTiming),
          SessionResourceErrorCode::cuda_event_creation_failed,
          "create codes-consumed event");
    }
    require_cuda(
        cudaEventCreateWithFlags(
            &audio_ready_event_, cudaEventDisableTiming),
        SessionResourceErrorCode::cuda_event_creation_failed,
        "create audio-ready event");

    contexts_.reserve(engines.size());
    for (const LoadedEngine& loaded : engines) {
      std::unique_ptr<nvinfer1::IExecutionContext> context(
          loaded.engine->createExecutionContext(
              nvinfer1::ExecutionContextAllocationStrategy::kSTATIC));
      if (context == nullptr) {
        fail(
            SessionResourceErrorCode::execution_context_creation_failed,
            "TensorRT returned a null context for " + loaded.name);
      }
      contexts_.push_back(EngineExecutionContext{
          .role = loaded.role,
          .name = loaded.name,
          .context = std::move(context),
      });
    }
    workspace_ = std::make_unique<SessionWorkspace>(
        manifest, context_device_memory_bytes_);
  } catch (...) {
    contexts_.clear();
    if (audio_ready_event_ != nullptr) {
      static_cast<void>(cudaEventDestroy(audio_ready_event_));
      audio_ready_event_ = nullptr;
    }
    for (cudaEvent_t& event : codes_consumed_events_) {
      if (event != nullptr) {
        static_cast<void>(cudaEventDestroy(event));
        event = nullptr;
      }
    }
    for (cudaEvent_t& event : codes_ready_events_) {
      if (event != nullptr) {
        static_cast<void>(cudaEventDestroy(event));
        event = nullptr;
      }
    }
    if (codec_stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(codec_stream_));
      codec_stream_ = nullptr;
    }
    if (generation_stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(generation_stream_));
      generation_stream_ = nullptr;
    }
    throw;
  }
}

SessionResources::~SessionResources() {
  // A startup or enqueue failure can unwind before the request pipeline has
  // armed a completion event. Keep TensorRT contexts and workspace storage
  // alive until both session streams have stopped using them. This is a
  // teardown-only wait; healthy per-request execution uses event dependencies.
  if (!teardown_synchronized_) {
    if (generation_stream_ != nullptr) {
      static_cast<void>(
          cudaStreamSynchronize(generation_stream_));
    }
    if (codec_stream_ != nullptr) {
      static_cast<void>(
          cudaStreamSynchronize(codec_stream_));
    }
  }
  workspace_.reset();
  contexts_.clear();
  if (audio_ready_event_ != nullptr) {
    static_cast<void>(cudaEventDestroy(audio_ready_event_));
  }
  for (cudaEvent_t event : codes_consumed_events_) {
    if (event != nullptr) {
      static_cast<void>(cudaEventDestroy(event));
    }
  }
  for (cudaEvent_t event : codes_ready_events_) {
    if (event != nullptr) {
      static_cast<void>(cudaEventDestroy(event));
    }
  }
  if (codec_stream_ != nullptr) {
    static_cast<void>(cudaStreamDestroy(codec_stream_));
  }
  if (generation_stream_ != nullptr) {
    static_cast<void>(cudaStreamDestroy(generation_stream_));
  }
}

void SessionResources::synchronize_for_teardown() {
  require_cuda(
      cudaStreamSynchronize(generation_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "synchronize generation stream for teardown");
  require_cuda(
      cudaStreamSynchronize(codec_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "synchronize codec stream for teardown");
  teardown_synchronized_ = true;
}

nvinfer1::IExecutionContext& SessionResources::context(
    const EngineRole role) {
  const auto found = std::find_if(
      contexts_.begin(),
      contexts_.end(),
      [role](const EngineExecutionContext& execution) {
        return execution.role == role;
      });
  if (found == contexts_.end()) {
    fail(
        SessionResourceErrorCode::missing_execution_context,
        "required engine role is absent: " +
            std::string(to_string(role)));
  }
  return *found->context;
}

cudaStream_t SessionResources::generation_stream() const noexcept {
  return generation_stream_;
}

cudaStream_t SessionResources::codec_stream() const noexcept {
  return codec_stream_;
}

cudaEvent_t SessionResources::codes_ready_event(
    const std::size_t batch_slot) const {
  if (batch_slot >= codes_ready_events_.size()) {
    throw std::out_of_range(
        "generation batch slot must be 0 or 1");
  }
  return codes_ready_events_.at(batch_slot);
}

cudaEvent_t SessionResources::codes_consumed_event(
    const std::size_t batch_slot) const {
  if (batch_slot >= codes_consumed_events_.size()) {
    throw std::out_of_range(
        "generation batch slot must be 0 or 1");
  }
  return codes_consumed_events_.at(batch_slot);
}

cudaEvent_t SessionResources::audio_ready_event() const noexcept {
  return audio_ready_event_;
}

std::uint64_t
SessionResources::context_device_memory_bytes() const noexcept {
  return context_device_memory_bytes_;
}

SessionWorkspace& SessionResources::workspace() noexcept {
  return *workspace_;
}

const SessionWorkspace& SessionResources::workspace() const noexcept {
  return *workspace_;
}

}  // namespace magpie_tts_rt

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "manifest/manifest.hpp"
#include "runtime/cuda_graph.hpp"
#include "runtime/cuda_graph_memory_accounting.hpp"
#include "runtime/main_decoder_cuda_graph.hpp"
#include "runtime/model_loader.hpp"
#include "runtime/nanocodec_cuda_graph.hpp"
#include "runtime/session_workspace.hpp"

namespace magpie_tts_rt {

enum class SessionResourceErrorCode {
  invalid_manifest_limit,
  device_memory_limit_exceeded,
  execution_context_creation_failed,
  cuda_stream_creation_failed,
  cuda_event_creation_failed,
  cuda_stream_synchronization_failed,
  cuda_graph_memory_query_failed,
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
  [[nodiscard]] nvinfer1::IExecutionContext& main_decoder_context(
      std::size_t cache_input);
  [[nodiscard]] nvinfer1::IExecutionContext& codec_steady_context(
      std::size_t state_input);
  [[nodiscard]] cudaStream_t generation_stream() const noexcept;
  [[nodiscard]] cudaStream_t codec_stream() const noexcept;
  [[nodiscard]] cudaEvent_t codes_ready_event(
      std::size_t batch_slot) const;
  [[nodiscard]] cudaEvent_t codes_consumed_event(
      std::size_t batch_slot) const;
  [[nodiscard]] cudaEvent_t audio_ready_event() const noexcept;
  [[nodiscard]] std::uint64_t context_device_memory_bytes() const noexcept;
  [[nodiscard]] std::uint64_t
  cuda_graph_device_memory_bytes() const noexcept;
  [[nodiscard]] SessionWorkspace& workspace() noexcept;
  [[nodiscard]] const SessionWorkspace& workspace() const noexcept;
  [[nodiscard]] bool local_ar_graph_ready() const noexcept;
  [[nodiscard]] bool main_decoder_warmed(
      std::size_t cache_input) const;
  [[nodiscard]] bool main_decoder_graph_ready(
      std::size_t cache_input) const;
  [[nodiscard]] bool main_decoder_graphs_ready() const noexcept;
  [[nodiscard]] bool nanocodec_initial_graph_ready() const noexcept;
  [[nodiscard]] bool nanocodec_steady_graph_ready(
      std::size_t state_input) const;
  [[nodiscard]] bool nanocodec_graphs_ready() const noexcept;
  [[nodiscard]] bool cuda_graph_memory_accounted() const noexcept;
  // Main Decoder graph launch parameters depend on request text length.
  // Calling this at every request boundary destroys the preceding request's
  // graphs and makes shape-based graph reuse impossible by construction.
  void begin_main_decoder_request();
  void record_main_decoder_eager_warmup(std::size_t cache_input);
  void capture_and_upload_main_decoder_graph(
      std::size_t cache_input,
      const CudaGraphExecutable::EnqueueOperation& enqueue);
  void launch_main_decoder_graph(std::size_t cache_input);
  // Releases request-scoped graph-accounting ownership after a failed
  // pipeline. This is a failure-only synchronization and never an eager
  // execution fallback.
  void abort_main_decoder_request() noexcept;
  // Opens one aggregate startup accounting scope before the golden pipeline
  // performs any deferred TensorRT, cuBLAS-bank, or CUDA-graph allocation.
  void begin_cuda_graph_memory_accounting();
  void capture_and_upload_local_ar_graph();
  void capture_and_upload_nanocodec_initial_graph();
  void capture_and_upload_nanocodec_steady_graph(
      std::size_t state_input);
  void finalize_cuda_graph_memory_accounting();
  void launch_local_ar_graph();
  void launch_nanocodec_initial_graph();
  void launch_nanocodec_steady_graph(std::size_t state_input);
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
  std::uint64_t maximum_device_memory_bytes_{0};
  std::uint64_t context_device_memory_bytes_{0};
  std::uint64_t cuda_graph_device_memory_bytes_{0};
  // Captured after explicit session allocation and before the first graph.
  // This is intentionally never replaced at a request boundary.
  CudaGraphMemorySnapshot aggregate_graph_memory_baseline_{};
  std::unique_ptr<nvinfer1::IExecutionContext>
      main_decoder_reverse_context_;
  std::unique_ptr<nvinfer1::IExecutionContext>
      codec_steady_reverse_context_;
  std::unique_ptr<SessionWorkspace> workspace_;
  MainDecoderCudaGraphSet main_decoder_graphs_;
  CudaGraphExecutable local_ar_graph_;
  NanoCodecCudaGraphSet nanocodec_graphs_;
  std::unique_lock<std::mutex> graph_memory_accounting_lock_;
  std::unique_lock<std::mutex> main_graph_memory_accounting_lock_;
  bool graph_memory_accounting_pending_{false};
  bool graph_memory_accounted_{false};
  bool main_graph_memory_accounting_pending_{false};
  bool main_graph_uses_startup_accounting_{false};
  bool teardown_synchronized_{false};
};

}  // namespace magpie_tts_rt

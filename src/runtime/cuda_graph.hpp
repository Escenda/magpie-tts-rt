#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

namespace magpie_tts_rt {

enum class CudaGraphErrorCode {
  invalid_state,
  capture_begin_failed,
  captured_enqueue_failed,
  capture_end_failed,
  instantiate_failed,
  upload_failed,
  launch_failed,
  destroy_failed,
};

[[nodiscard]] std::string_view to_string(
    CudaGraphErrorCode code) noexcept;

class CudaGraphError final : public std::runtime_error {
 public:
  CudaGraphError(
      CudaGraphErrorCode code,
      std::string detail);

  [[nodiscard]] CudaGraphErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  CudaGraphErrorCode code_;
  std::string detail_;
};

// One immutable captured execution. TensorRT captures context state and tensor
// addresses in the graph, so a ready instance must only be replayed with the
// context and buffers used by the capture left unchanged.
class CudaGraphExecutable final {
 public:
  using EnqueueOperation =
      std::function<bool(cudaStream_t)>;

  CudaGraphExecutable() = default;
  ~CudaGraphExecutable();

  CudaGraphExecutable(const CudaGraphExecutable&) = delete;
  CudaGraphExecutable& operator=(
      const CudaGraphExecutable&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  void capture_and_upload(
      cudaStream_t stream,
      const EnqueueOperation& enqueue);
  void launch(cudaStream_t stream) const;
  // Request boundaries must observe destruction failures. A failed destroy
  // leaves the executable owned by this object so poisoned-session teardown
  // can make one final best-effort release attempt.
  void reset_checked();
  void reset() noexcept;

 private:
  cudaGraphExec_t executable_{nullptr};
};

}  // namespace magpie_tts_rt

#if defined(MAGPIE_TTS_RT_CUDA_GRAPH_TESTING)
namespace magpie_tts_rt::testing {

using CudaGraphExecDestroyOperation =
    cudaError_t (*)(cudaGraphExec_t);

// GPU-test-only fault injection. Passing nullptr restores the CUDA runtime
// implementation. The production runtime does not export this hook.
void set_cuda_graph_exec_destroy_operation(
    CudaGraphExecDestroyOperation operation) noexcept;

}  // namespace magpie_tts_rt::testing
#endif

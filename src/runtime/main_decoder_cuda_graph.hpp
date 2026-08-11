#pragma once

#include <array>
#include <cstddef>

#include <cuda_runtime_api.h>

#include "runtime/cuda_graph.hpp"

namespace magpie_tts_rt {

// Main Decoder recurrent state alternates between cache A and cache B. Each
// direction owns a distinct TensorRT execution context because captured
// context state cannot be shared by two CUDA Graph executables. Graphs are
// request-scoped: dynamic text length changes TensorRT launch parameters, so
// the owner must reset this set before every request and capture both routes
// again after that request's eager context warmups.
class MainDecoderCudaGraphSet final {
 public:
  using EnqueueOperation = CudaGraphExecutable::EnqueueOperation;

  MainDecoderCudaGraphSet() = default;
  ~MainDecoderCudaGraphSet() = default;

  MainDecoderCudaGraphSet(const MainDecoderCudaGraphSet&) = delete;
  MainDecoderCudaGraphSet& operator=(
      const MainDecoderCudaGraphSet&) = delete;

  [[nodiscard]] bool warmed(std::size_t cache_input) const;
  [[nodiscard]] bool ready(std::size_t cache_input) const;
  [[nodiscard]] bool ready() const noexcept;

  // Records the one accepted eager execution for a direction. The caller
  // invokes this only after enqueueV3 accepted the production addresses and
  // shape for the current request. A second eager execution is rejected.
  void record_eager_warmup(
      std::size_t cache_input,
      const void* context_identity);

  // Captures the next invocation for the warmed direction. The captured
  // operation must contain both the device-position increment and enqueueV3.
  void capture(
      cudaStream_t stream,
      std::size_t cache_input,
      const void* context_identity,
      const EnqueueOperation& enqueue);

  void launch(cudaStream_t stream, std::size_t cache_input) const;
  // Production request boundary. Unlike reset(), this propagates CUDA
  // destruction failures so the worker poisons the owning session.
  void reset_checked();
  void reset() noexcept;

 private:
  std::array<CudaGraphExecutable, 2> graphs_;
  std::array<const void*, 2> context_identities_{};
  std::array<bool, 2> warmed_{};
};

}  // namespace magpie_tts_rt

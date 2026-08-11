#pragma once

#include <array>
#include <cstddef>

#include <cuda_runtime_api.h>

#include "runtime/cuda_graph.hpp"

namespace magpie_tts_rt {

// The fixed NanoCodec schedule has three immutable captured executions:
// initial 4 frames and both directions of the alternating steady 8-frame
// state. A TensorRT execution context is part of captured graph state, so the
// two steady directions must use distinct contexts as well as distinct tensor
// addresses. context_identity is never dereferenced; it records that ownership
// contract and rejects an unsafe second graph for the same context. The owner
// must keep all three contexts and every captured address alive and unchanged
// until both the codec stream is idle and reset() has destroyed the graph
// executables.
class NanoCodecCudaGraphSet final {
 public:
  using EnqueueOperation = CudaGraphExecutable::EnqueueOperation;

  NanoCodecCudaGraphSet() = default;
  ~NanoCodecCudaGraphSet() = default;

  NanoCodecCudaGraphSet(const NanoCodecCudaGraphSet&) = delete;
  NanoCodecCudaGraphSet& operator=(
      const NanoCodecCudaGraphSet&) = delete;

  [[nodiscard]] bool initial_ready() const noexcept;
  [[nodiscard]] bool steady_ready(
      std::size_t state_input) const;
  [[nodiscard]] bool ready() const noexcept;

  void capture_initial(
      cudaStream_t stream,
      const void* context_identity,
      const EnqueueOperation& enqueue);
  void capture_steady(
      cudaStream_t stream,
      std::size_t state_input,
      const void* context_identity,
      const EnqueueOperation& enqueue);

  void launch_initial(cudaStream_t stream) const;
  void launch_steady(
      cudaStream_t stream,
      std::size_t state_input) const;
  void reset() noexcept;

 private:
  void require_unique_context(
      const void* context_identity,
      const char* route) const;

  CudaGraphExecutable initial_;
  std::array<CudaGraphExecutable, 2> steady_;
  const void* initial_context_identity_{nullptr};
  std::array<const void*, 2> steady_context_identities_{};
};

}  // namespace magpie_tts_rt

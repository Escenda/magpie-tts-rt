#include "runtime/nanocodec_cuda_graph.hpp"

#include <string>

namespace magpie_tts_rt {
namespace {

[[noreturn]] void invalid_state(const std::string& detail) {
  throw CudaGraphError(CudaGraphErrorCode::invalid_state, detail);
}

void require_state_input(const std::size_t state_input) {
  if (state_input >= 2U) {
    invalid_state(
        "NanoCodec steady graph state input must be 0 or 1");
  }
}

}  // namespace

bool NanoCodecCudaGraphSet::initial_ready() const noexcept {
  return initial_.ready();
}

bool NanoCodecCudaGraphSet::steady_ready(
    const std::size_t state_input) const {
  require_state_input(state_input);
  return steady_.at(state_input).ready();
}

bool NanoCodecCudaGraphSet::ready() const noexcept {
  return initial_.ready() && steady_[0].ready() &&
         steady_[1].ready();
}

void NanoCodecCudaGraphSet::require_unique_context(
    const void* context_identity,
    const char* route) const {
  if (context_identity == nullptr) {
    invalid_state(
        std::string(route) +
        " requires a non-null TensorRT context identity");
  }
  if (context_identity == initial_context_identity_ ||
      context_identity == steady_context_identities_[0] ||
      context_identity == steady_context_identities_[1]) {
    invalid_state(
        std::string(route) +
        " attempted to capture a second graph from an already-bound "
        "TensorRT execution context");
  }
}

void NanoCodecCudaGraphSet::capture_initial(
    const cudaStream_t stream,
    const void* context_identity,
    const EnqueueOperation& enqueue) {
  require_unique_context(context_identity, "NanoCodec initial-4 graph");
  initial_.capture_and_upload(stream, enqueue);
  initial_context_identity_ = context_identity;
}

void NanoCodecCudaGraphSet::capture_steady(
    const cudaStream_t stream,
    const std::size_t state_input,
    const void* context_identity,
    const EnqueueOperation& enqueue) {
  require_state_input(state_input);
  require_unique_context(
      context_identity,
      state_input == 0U ? "NanoCodec steady-8 A-to-B graph"
                        : "NanoCodec steady-8 B-to-A graph");
  steady_.at(state_input).capture_and_upload(stream, enqueue);
  steady_context_identities_.at(state_input) = context_identity;
}

void NanoCodecCudaGraphSet::launch_initial(
    const cudaStream_t stream) const {
  initial_.launch(stream);
}

void NanoCodecCudaGraphSet::launch_steady(
    const cudaStream_t stream,
    const std::size_t state_input) const {
  require_state_input(state_input);
  steady_.at(state_input).launch(stream);
}

void NanoCodecCudaGraphSet::reset() noexcept {
  for (CudaGraphExecutable& graph : steady_) {
    graph.reset();
  }
  initial_.reset();
  initial_context_identity_ = nullptr;
  steady_context_identities_.fill(nullptr);
}

}  // namespace magpie_tts_rt

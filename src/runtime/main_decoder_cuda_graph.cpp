#include "runtime/main_decoder_cuda_graph.hpp"

#include <string>

namespace magpie_tts_rt {
namespace {

[[noreturn]] void invalid_state(const std::string& detail) {
  throw CudaGraphError(CudaGraphErrorCode::invalid_state, detail);
}

void require_cache_input(const std::size_t cache_input) {
  if (cache_input >= 2U) {
    invalid_state(
        "Main Decoder graph cache input must be 0 or 1");
  }
}

[[nodiscard]] const char* route_name(
    const std::size_t cache_input) noexcept {
  return cache_input == 0U ? "Main Decoder A-to-B"
                           : "Main Decoder B-to-A";
}

}  // namespace

bool MainDecoderCudaGraphSet::warmed(
    const std::size_t cache_input) const {
  require_cache_input(cache_input);
  return warmed_.at(cache_input);
}

bool MainDecoderCudaGraphSet::ready(
    const std::size_t cache_input) const {
  require_cache_input(cache_input);
  return graphs_.at(cache_input).ready();
}

bool MainDecoderCudaGraphSet::ready() const noexcept {
  return graphs_[0].ready() && graphs_[1].ready();
}

void MainDecoderCudaGraphSet::record_eager_warmup(
    const std::size_t cache_input,
    const void* const context_identity) {
  require_cache_input(cache_input);
  if (context_identity == nullptr) {
    invalid_state(
        std::string(route_name(cache_input)) +
        " eager warmup requires a non-null TensorRT context identity");
  }
  if (warmed_.at(cache_input) || graphs_.at(cache_input).ready()) {
    invalid_state(
        std::string(route_name(cache_input)) +
        " accepted more than one eager warmup in one request");
  }
  const std::size_t other = 1U - cache_input;
  if (context_identity == context_identities_.at(other)) {
    invalid_state(
        std::string(route_name(cache_input)) +
        " attempted to share a TensorRT context with the reverse graph");
  }
  context_identities_.at(cache_input) = context_identity;
  warmed_.at(cache_input) = true;
}

void MainDecoderCudaGraphSet::capture(
    const cudaStream_t stream,
    const std::size_t cache_input,
    const void* const context_identity,
    const EnqueueOperation& enqueue) {
  require_cache_input(cache_input);
  if (!warmed_.at(cache_input)) {
    invalid_state(
        std::string(route_name(cache_input)) +
        " graph capture was attempted before its eager warmup");
  }
  if (context_identity == nullptr ||
      context_identity != context_identities_.at(cache_input)) {
    invalid_state(
        std::string(route_name(cache_input)) +
        " graph capture did not use its warmed TensorRT context");
  }
  graphs_.at(cache_input).capture_and_upload(stream, enqueue);
}

void MainDecoderCudaGraphSet::launch(
    const cudaStream_t stream,
    const std::size_t cache_input) const {
  require_cache_input(cache_input);
  graphs_.at(cache_input).launch(stream);
}

void MainDecoderCudaGraphSet::reset() noexcept {
  for (CudaGraphExecutable& graph : graphs_) {
    graph.reset();
  }
  context_identities_.fill(nullptr);
  warmed_.fill(false);
}

void MainDecoderCudaGraphSet::reset_checked() {
  for (CudaGraphExecutable& graph : graphs_) {
    graph.reset_checked();
  }
  context_identities_.fill(nullptr);
  warmed_.fill(false);
}

}  // namespace magpie_tts_rt

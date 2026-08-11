#include "runtime/main_decoder_cuda_graph.hpp"
#include "runtime/synthesis_kernels.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

__global__ void decoder_step(
    const std::int64_t* input_cache,
    const std::int64_t* position,
    std::int64_t* output_cache) {
  output_cache[0] = input_cache[0] + position[0];
}

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_cuda(
    const cudaError_t status,
    const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        operation + ": " + cudaGetErrorString(status));
  }
}

void enqueue_eager_step(
    std::int64_t* position,
    const std::int64_t* input_cache,
    std::int64_t* output_cache,
    const cudaStream_t stream) {
  require_cuda(
      magpie_tts_rt::launch_advance_decoder_position(position, stream),
      "advance eager decoder position");
  decoder_step<<<1, 1, 0, stream>>>(
      input_cache, position, output_cache);
  require_cuda(cudaPeekAtLastError(), "enqueue eager decoder step");
}

void test_request_scoped_warm_capture_and_replay() {
  cudaStream_t stream = nullptr;
  std::int64_t* position = nullptr;
  std::int64_t* cache_a = nullptr;
  std::int64_t* cache_b = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create Main Decoder graph test stream");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&position), sizeof(*position)),
      "allocate device position");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&cache_a), sizeof(*cache_a)),
      "allocate cache A");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&cache_b), sizeof(*cache_b)),
      "allocate cache B");
  constexpr std::int64_t position_before_first_step = 217;
  constexpr std::int64_t initial_cache = 1;
  require_cuda(
      cudaMemcpyAsync(
          position,
          &position_before_first_step,
          sizeof(position_before_first_step),
          cudaMemcpyHostToDevice,
          stream),
      "initialize device position");
  require_cuda(
      cudaMemcpyAsync(
          cache_a,
          &initial_cache,
          sizeof(initial_cache),
          cudaMemcpyHostToDevice,
          stream),
      "initialize cache A");

  std::uint8_t context_a_to_b = 0;
  std::uint8_t context_b_to_a = 0;
  magpie_tts_rt::MainDecoderCudaGraphSet graphs;

  // The two eager results are real recurrent state, not discarded warmups.
  enqueue_eager_step(position, cache_a, cache_b, stream);  // position 218
  graphs.record_eager_warmup(0, &context_a_to_b);
  enqueue_eager_step(position, cache_b, cache_a, stream);  // position 219
  graphs.record_eager_warmup(1, &context_b_to_a);

  graphs.capture(
      stream,
      0,
      &context_a_to_b,
      [position, cache_a, cache_b](
          const cudaStream_t captured_stream) noexcept {
        if (magpie_tts_rt::launch_advance_decoder_position(
                position, captured_stream) != cudaSuccess) {
          return false;
        }
        decoder_step<<<1, 1, 0, captured_stream>>>(
            cache_a, position, cache_b);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  graphs.launch(stream, 0);  // position 220
  graphs.capture(
      stream,
      1,
      &context_b_to_a,
      [position, cache_a, cache_b](
          const cudaStream_t captured_stream) noexcept {
        if (magpie_tts_rt::launch_advance_decoder_position(
                position, captured_stream) != cudaSuccess) {
          return false;
        }
        decoder_step<<<1, 1, 0, captured_stream>>>(
            cache_b, position, cache_a);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  graphs.launch(stream, 1);  // position 221
  require(graphs.ready(), "both Main Decoder graphs are not ready");

  graphs.launch(stream, 0);  // position 222
  graphs.launch(stream, 1);  // position 223
  std::int64_t observed_position = 0;
  std::int64_t observed_cache_a = 0;
  require_cuda(
      cudaMemcpyAsync(
          &observed_position,
          position,
          sizeof(observed_position),
          cudaMemcpyDeviceToHost,
          stream),
      "copy final device position");
  require_cuda(
      cudaMemcpyAsync(
          &observed_cache_a,
          cache_a,
          sizeof(observed_cache_a),
          cudaMemcpyDeviceToHost,
          stream),
      "copy final cache A");
  require_cuda(
      cudaStreamSynchronize(stream),
      "complete Main Decoder graph sequence");
  require(
      observed_position == 223,
      "captured device position did not advance exactly once per step");
  require(
      observed_cache_a == 1324,
      "eager/captured/replayed cache sequence lost a production result");

  graphs.reset();
  require(!graphs.ready(), "request reset retained Main Decoder graphs");
  require(
      !graphs.warmed(0) && !graphs.warmed(1),
      "request reset retained eager-warmup state");
  require_cuda(cudaFree(cache_b), "free cache B");
  require_cuda(cudaFree(cache_a), "free cache A");
  require_cuda(cudaFree(position), "free device position");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy Main Decoder graph test stream");
}

void test_route_state_fails_closed() {
  cudaStream_t stream = nullptr;
  std::int64_t* value = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create Main Decoder state test stream");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&value), sizeof(*value)),
      "allocate Main Decoder state value");
  std::uint8_t shared_context = 0;
  magpie_tts_rt::MainDecoderCudaGraphSet graphs;

  bool capture_before_warmup_rejected = false;
  try {
    graphs.capture(
        stream,
        0,
        &shared_context,
        [](const cudaStream_t) noexcept { return true; });
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    capture_before_warmup_rejected =
        error.code() == magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(
      capture_before_warmup_rejected,
      "capture before eager warmup was accepted");

  graphs.record_eager_warmup(0, &shared_context);
  bool shared_context_rejected = false;
  try {
    graphs.record_eager_warmup(1, &shared_context);
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    shared_context_rejected =
        error.code() == magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(
      shared_context_rejected,
      "one TensorRT context was accepted for both cache directions");

  bool missing_graph_rejected = false;
  try {
    graphs.launch(stream, 0);
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    missing_graph_rejected =
        error.code() == magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(
      missing_graph_rejected,
      "a missing Main Decoder graph fell back to eager execution");
  graphs.reset();
  require_cuda(cudaFree(value), "free Main Decoder state value");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy Main Decoder state test stream");
}

}  // namespace

int main() {
  try {
    test_request_scoped_warm_capture_and_replay();
    test_route_state_fails_closed();
  } catch (const std::exception& error) {
    std::cerr << "Main Decoder CUDA graph GPU test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

#include "runtime/nanocodec_cuda_graph.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

__global__ void add_constant(
    const std::uint64_t* input,
    std::uint64_t* output,
    const std::uint64_t increment) {
  output[0] = input[0] + increment;
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

void test_fixed_routes_and_alternating_state() {
  cudaStream_t stream = nullptr;
  std::uint64_t* initial_input = nullptr;
  std::uint64_t* state_a = nullptr;
  std::uint64_t* state_b = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create NanoCodec graph test stream");
  require_cuda(
      cudaMalloc(
          reinterpret_cast<void**>(&initial_input),
          sizeof(*initial_input)),
      "allocate initial input");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&state_a), sizeof(*state_a)),
      "allocate state A");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&state_b), sizeof(*state_b)),
      "allocate state B");

  std::uint8_t initial_context = 0;
  std::uint8_t steady_a_to_b_context = 0;
  std::uint8_t steady_b_to_a_context = 0;
  magpie_tts_rt::NanoCodecCudaGraphSet graphs;
  graphs.capture_initial(
      stream,
      &initial_context,
      [initial_input, state_a](const cudaStream_t captured_stream) {
        add_constant<<<1, 1, 0, captured_stream>>>(
            initial_input, state_a, 4U);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  graphs.capture_steady(
      stream,
      0,
      &steady_a_to_b_context,
      [state_a, state_b](const cudaStream_t captured_stream) {
        add_constant<<<1, 1, 0, captured_stream>>>(
            state_a, state_b, 8U);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  graphs.capture_steady(
      stream,
      1,
      &steady_b_to_a_context,
      [state_a, state_b](const cudaStream_t captured_stream) {
        add_constant<<<1, 1, 0, captured_stream>>>(
            state_b, state_a, 8U);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  require(graphs.ready(), "fixed NanoCodec graphs are not all ready");

  constexpr std::uint64_t initial_value = 3;
  require_cuda(
      cudaMemcpyAsync(
          initial_input,
          &initial_value,
          sizeof(initial_value),
          cudaMemcpyHostToDevice,
          stream),
      "copy initial graph input");
  graphs.launch_initial(stream);
  for (std::size_t pair = 0; pair < 5; ++pair) {
    graphs.launch_steady(stream, 0);
    graphs.launch_steady(stream, 1);
  }
  std::uint64_t observed_a = 0;
  std::uint64_t observed_b = 0;
  require_cuda(
      cudaMemcpyAsync(
          &observed_a,
          state_a,
          sizeof(observed_a),
          cudaMemcpyDeviceToHost,
          stream),
      "copy state A");
  require_cuda(
      cudaMemcpyAsync(
          &observed_b,
          state_b,
          sizeof(observed_b),
          cudaMemcpyDeviceToHost,
          stream),
      "copy state B");
  require_cuda(
      cudaStreamSynchronize(stream),
      "complete alternating graph sequence");
  require(observed_a == 87U, "state A graph direction was corrupted");
  require(observed_b == 79U, "state B graph direction was corrupted");

  graphs.reset();
  require(!graphs.ready(), "NanoCodec graph reset retained readiness");
  require_cuda(cudaFree(state_b), "free state B");
  require_cuda(cudaFree(state_a), "free state A");
  require_cuda(cudaFree(initial_input), "free initial input");
  require_cuda(cudaStreamDestroy(stream), "destroy graph test stream");
}

void test_context_reuse_is_rejected() {
  cudaStream_t stream = nullptr;
  std::uint64_t* value = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create context-ownership test stream");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&value), sizeof(*value)),
      "allocate context-ownership value");
  std::uint8_t shared_context = 0;
  magpie_tts_rt::NanoCodecCudaGraphSet graphs;
  graphs.capture_steady(
      stream,
      0,
      &shared_context,
      [value](const cudaStream_t captured_stream) {
        add_constant<<<1, 1, 0, captured_stream>>>(value, value, 1U);
        return cudaPeekAtLastError() == cudaSuccess;
      });

  bool rejected = false;
  try {
    graphs.capture_steady(
        stream,
        1,
        &shared_context,
        [value](const cudaStream_t captured_stream) {
          add_constant<<<1, 1, 0, captured_stream>>>(value, value, 1U);
          return cudaPeekAtLastError() == cudaSuccess;
        });
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    rejected =
        error.code() == magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(
      rejected,
      "one TensorRT context was accepted for both steady graphs");
  require(
      !graphs.steady_ready(1),
      "rejected context reuse left a graph executable behind");
  graphs.reset();
  require_cuda(cudaFree(value), "free context-ownership value");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy context-ownership test stream");
}

void test_capture_failure_does_not_claim_context() {
  cudaStream_t stream = nullptr;
  std::uint64_t* value = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create capture-failure test stream");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&value), sizeof(*value)),
      "allocate capture-failure value");
  std::uint8_t context = 0;
  magpie_tts_rt::NanoCodecCudaGraphSet graphs;
  bool failure_reported = false;
  try {
    graphs.capture_initial(
        stream,
        &context,
        [](const cudaStream_t) noexcept { return false; });
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    failure_reported =
        error.code() ==
        magpie_tts_rt::CudaGraphErrorCode::captured_enqueue_failed;
  }
  require(failure_reported, "failed capture did not report its error");
  graphs.capture_initial(
      stream,
      &context,
      [value](const cudaStream_t captured_stream) {
        add_constant<<<1, 1, 0, captured_stream>>>(value, value, 1U);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  require(
      graphs.initial_ready(),
      "failed capture permanently claimed its context identity");
  graphs.reset();
  require_cuda(cudaFree(value), "free capture-failure value");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy capture-failure test stream");
}

void test_invalid_route_arguments_fail_closed() {
  cudaStream_t stream = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create invalid-route test stream");
  magpie_tts_rt::NanoCodecCudaGraphSet graphs;
  bool null_context_rejected = false;
  try {
    graphs.capture_initial(
        stream,
        nullptr,
        [](const cudaStream_t) noexcept { return true; });
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    null_context_rejected =
        error.code() == magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(null_context_rejected, "null context identity was accepted");

  bool state_index_rejected = false;
  try {
    graphs.launch_steady(stream, 2);
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    state_index_rejected =
        error.code() == magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(state_index_rejected, "invalid steady state index was accepted");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy invalid-route test stream");
}

void test_uncaptured_route_has_no_eager_fallback() {
  cudaStream_t stream = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create missing-graph test stream");
  magpie_tts_rt::NanoCodecCudaGraphSet graphs;
  bool rejected = false;
  try {
    graphs.launch_initial(stream);
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    rejected =
        error.code() == magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(rejected, "missing initial graph did not fail closed");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy missing-graph test stream");
}

}  // namespace

int main() {
  try {
    test_fixed_routes_and_alternating_state();
    test_context_reuse_is_rejected();
    test_capture_failure_does_not_claim_context();
    test_invalid_route_arguments_fail_closed();
    test_uncaptured_route_has_no_eager_fallback();
  } catch (const std::exception& error) {
    std::cerr << "NanoCodec CUDA graph GPU test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

#include "runtime/cuda_graph.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

__global__ void increment_kernel(std::uint64_t* value) {
  value[0] += 1U;
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

cudaError_t reject_graph_exec_destroy(cudaGraphExec_t) {
  return cudaErrorUnknown;
}

void test_capture_and_replay() {
  cudaStream_t stream = nullptr;
  std::uint64_t* value = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create graph test stream");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&value), sizeof(*value)),
      "allocate graph test value");
  require_cuda(
      cudaMemsetAsync(value, 0, sizeof(*value), stream),
      "clear graph test value");

  magpie_tts_rt::CudaGraphExecutable graph;
  bool launch_before_ready_rejected = false;
  try {
    graph.launch(stream);
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    launch_before_ready_rejected =
        error.code() ==
        magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(
      launch_before_ready_rejected,
      "graph launch before capture was not rejected");

  graph.capture_and_upload(
      stream,
      [value](const cudaStream_t captured_stream) noexcept {
        increment_kernel<<<1, 1, 0, captured_stream>>>(value);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  require(graph.ready(), "captured graph is not ready");

  bool second_capture_rejected = false;
  try {
    graph.capture_and_upload(
        stream,
        [](const cudaStream_t) noexcept { return true; });
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    second_capture_rejected =
        error.code() ==
        magpie_tts_rt::CudaGraphErrorCode::invalid_state;
  }
  require(
      second_capture_rejected,
      "a second graph capture was not rejected");

  constexpr std::uint64_t replays = 1000;
  for (std::uint64_t replay = 0; replay < replays; ++replay) {
    graph.launch(stream);
  }
  std::uint64_t observed = 0;
  require_cuda(
      cudaMemcpyAsync(
          &observed,
          value,
          sizeof(observed),
          cudaMemcpyDeviceToHost,
          stream),
      "copy graph replay result");
  require_cuda(
      cudaStreamSynchronize(stream),
      "complete graph replays");
  require(observed == replays, "graph replay count mismatch");

  graph.reset_checked();
  require(!graph.ready(), "graph reset retained its executable");
  require_cuda(cudaFree(value), "free graph test value");
  require_cuda(cudaStreamDestroy(stream), "destroy graph test stream");
}

void test_checked_reset_reports_destroy_failure() {
  cudaStream_t stream = nullptr;
  std::uint64_t* value = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create checked-reset test stream");
  require_cuda(
      cudaMalloc(reinterpret_cast<void**>(&value), sizeof(*value)),
      "allocate checked-reset test value");

  magpie_tts_rt::CudaGraphExecutable graph;
  graph.capture_and_upload(
      stream,
      [value](const cudaStream_t captured_stream) noexcept {
        increment_kernel<<<1, 1, 0, captured_stream>>>(value);
        return cudaPeekAtLastError() == cudaSuccess;
      });
  magpie_tts_rt::testing::set_cuda_graph_exec_destroy_operation(
      reject_graph_exec_destroy);
  bool destroy_failure_reported = false;
  try {
    graph.reset_checked();
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    destroy_failure_reported =
        error.code() ==
        magpie_tts_rt::CudaGraphErrorCode::destroy_failed;
  }
  require(
      destroy_failure_reported,
      "checked reset did not report executable destruction failure");
  require(
      graph.ready(),
      "failed checked reset discarded ownership of the executable");

  magpie_tts_rt::testing::set_cuda_graph_exec_destroy_operation(nullptr);
  graph.reset();
  require_cuda(cudaFree(value), "free checked-reset test value");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy checked-reset test stream");
}

void test_failed_capture_closes_stream_capture() {
  cudaStream_t stream = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create failed-capture stream");
  magpie_tts_rt::CudaGraphExecutable graph;
  bool enqueue_failure_reported = false;
  try {
    graph.capture_and_upload(
        stream,
        [](const cudaStream_t) noexcept { return false; });
  } catch (const magpie_tts_rt::CudaGraphError& error) {
    enqueue_failure_reported =
        error.code() ==
        magpie_tts_rt::CudaGraphErrorCode::captured_enqueue_failed;
  }
  require(
      enqueue_failure_reported,
      "failed captured enqueue did not report its typed error");
  cudaStreamCaptureStatus capture_status =
      cudaStreamCaptureStatusActive;
  require_cuda(
      cudaStreamIsCapturing(stream, &capture_status),
      "query failed-capture stream state");
  require(
      capture_status == cudaStreamCaptureStatusNone,
      "failed enqueue left the stream in capture mode");
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy failed-capture stream");
}

}  // namespace

int main() {
  try {
    test_capture_and_replay();
    test_checked_reset_reports_destroy_failure();
    test_failed_capture_closes_stream_capture();
  } catch (const std::exception& error) {
    std::cerr << "CUDA graph GPU test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kBatchHeads = 24;
constexpr std::int32_t kCacheCapacity = 467;
constexpr std::int32_t kHeadWidth = 64;
constexpr std::int32_t kFirstLength = 218;
constexpr std::int32_t kLastLength = 219;
constexpr std::size_t kCublasWorkspaceBytes = 4U * 1024U * 1024U;

#define CUDA_CHECK(expression)                                                 \
  do {                                                                         \
    const cudaError_t status = (expression);                                   \
    if (status != cudaSuccess) {                                               \
      std::fprintf(                                                            \
          stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,             \
          #expression, cudaGetErrorString(status));                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

#define CUBLAS_CHECK(expression)                                               \
  do {                                                                         \
    const cublasStatus_t status = (expression);                                \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::fprintf(                                                            \
          stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__,             \
          #expression, static_cast<int>(status));                              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

template <typename Element>
std::vector<Element> read_exact(
    const std::filesystem::path& path,
    const std::size_t element_count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  const std::size_t expected_bytes = element_count * sizeof(Element);
  const std::streamsize actual_bytes = input.tellg();
  if (actual_bytes < 0 ||
      static_cast<std::size_t>(actual_bytes) != expected_bytes) {
    std::fprintf(
        stderr,
        "%s has %lld bytes; expected %zu\n",
        path.c_str(),
        static_cast<long long>(actual_bytes),
        expected_bytes);
    std::exit(1);
  }
  input.seekg(0);
  std::vector<Element> values(element_count);
  input.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(expected_bytes));
  if (!input) {
    std::fprintf(stderr, "failed to read %s\n", path.c_str());
    std::exit(1);
  }
  return values;
}

struct alignas(16) Bf16Vector8 {
  __nv_bfloat16 values[8];
};

__global__ void initialize_output(__nv_bfloat16* output) {
  const std::int32_t index =
      static_cast<std::int32_t>(blockIdx.x) *
          static_cast<std::int32_t>(blockDim.x) +
      static_cast<std::int32_t>(threadIdx.x);
  if (index < kBatchHeads * kHeadWidth) {
    output[index] = __float2bfloat16_rn(0.0F);
  }
}

__global__ void select_length(
    const std::int64_t* position,
    const cudaGraphConditionalHandle handle) {
  const std::int64_t branch = position[0] - (kFirstLength - 1);
  cudaGraphSetConditional(
      handle,
      branch >= 0 && branch <= (kLastLength - kFirstLength)
          ? static_cast<unsigned int>(branch)
          : static_cast<unsigned int>(kLastLength - kFirstLength + 1));
}

__global__ void stage_values(
    const Bf16Vector8* source,
    Bf16Vector8* destination,
    const std::int32_t active_length) {
  constexpr std::int32_t vectors_per_row = kHeadWidth / 8;
  const std::int32_t vector_index =
      static_cast<std::int32_t>(blockIdx.x) *
          static_cast<std::int32_t>(blockDim.x) +
      static_cast<std::int32_t>(threadIdx.x);
  const std::int32_t vector_count =
      kBatchHeads * active_length * vectors_per_row;
  if (vector_index >= vector_count) {
    return;
  }
  const std::int32_t vector_in_row = vector_index % vectors_per_row;
  const std::int32_t active_row = vector_index / vectors_per_row;
  const std::int32_t batch_head = active_row / active_length;
  const std::int32_t cache_index = active_row % active_length;
  const std::int32_t source_vector =
      (batch_head * kCacheCapacity + cache_index) * vectors_per_row +
      vector_in_row;
  destination[vector_index] = source[source_vector];
}

cudaGraphNode_t add_selector_node(
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    const std::size_t dependency_count,
    const std::int64_t* position,
    const cudaGraphConditionalHandle handle) {
  void* arguments[] = {
      const_cast<std::int64_t**>(&position),
      const_cast<cudaGraphConditionalHandle*>(&handle),
  };
  cudaKernelNodeParams parameters{};
  parameters.func = reinterpret_cast<void*>(select_length);
  parameters.gridDim = dim3(1, 1, 1);
  parameters.blockDim = dim3(1, 1, 1);
  parameters.kernelParams = arguments;
  cudaGraphNode_t node = nullptr;
  CUDA_CHECK(cudaGraphAddKernelNode(
      &node, graph, dependencies, dependency_count, &parameters));
  return node;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(
        stderr,
        "usage: %s VALUE_BF16 PROBABILITIES_BF16 EXPECTED_CONTEXT_BF16\n",
        argv[0]);
    return 2;
  }
  const std::vector<__nv_bfloat16> host_values = read_exact<__nv_bfloat16>(
      argv[1],
      static_cast<std::size_t>(kBatchHeads) * kCacheCapacity * kHeadWidth);
  const std::vector<__nv_bfloat16> host_probabilities =
      read_exact<__nv_bfloat16>(
          argv[2],
          static_cast<std::size_t>(kBatchHeads) * kLastLength);
  const std::vector<__nv_bfloat16> expected = read_exact<__nv_bfloat16>(
      argv[3],
      static_cast<std::size_t>(kBatchHeads) * kHeadWidth);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::int64_t* position = nullptr;
  __nv_bfloat16* values = nullptr;
  __nv_bfloat16* probabilities = nullptr;
  __nv_bfloat16* compact_values = nullptr;
  __nv_bfloat16* output = nullptr;
  void* cublas_workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&position, sizeof(*position)));
  CUDA_CHECK(cudaMalloc(&values, host_values.size() * sizeof(*values)));
  CUDA_CHECK(cudaMalloc(
      &probabilities, host_probabilities.size() * sizeof(*probabilities)));
  CUDA_CHECK(cudaMalloc(
      &compact_values,
      static_cast<std::size_t>(kBatchHeads) * kLastLength * kHeadWidth *
          sizeof(*compact_values)));
  CUDA_CHECK(cudaMalloc(
      &output,
      static_cast<std::size_t>(kBatchHeads) * kHeadWidth * sizeof(*output)));
  CUDA_CHECK(cudaMalloc(&cublas_workspace, kCublasWorkspaceBytes));
  const std::int64_t host_position = kLastLength - 1;
  CUDA_CHECK(cudaMemcpy(
      position, &host_position, sizeof(host_position), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      values,
      host_values.data(),
      host_values.size() * sizeof(*values),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      probabilities,
      host_probabilities.data(),
      host_probabilities.size() * sizeof(*probabilities),
      cudaMemcpyHostToDevice));

  cublasHandle_t cublas = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUBLAS_CHECK(cublasSetWorkspace(
      cublas, cublas_workspace, kCublasWorkspaceBytes));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  initialize_output<<<6, 256, 0, stream>>>(output);
  CUDA_CHECK(cudaPeekAtLastError());

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  const cudaGraphEdgeData* dependency_data = nullptr;
  std::size_t dependency_count = 0;
  CUDA_CHECK(cudaStreamGetCaptureInfo(
      stream,
      &capture_status,
      nullptr,
      &graph,
      &dependencies,
      &dependency_data,
      &dependency_count));
  if (capture_status != cudaStreamCaptureStatusActive || graph == nullptr ||
      dependency_count != 1) {
    std::fprintf(stderr, "unexpected outer capture state\n");
    return 3;
  }

  cudaGraphConditionalHandle handle{};
  CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph));
  const cudaGraphNode_t selector_node = add_selector_node(
      graph, dependencies, dependency_count, position, handle);
  cudaGraphNodeParams conditional_parameters{};
  conditional_parameters.type = cudaGraphNodeTypeConditional;
  conditional_parameters.conditional.handle = handle;
  conditional_parameters.conditional.type = cudaGraphCondTypeSwitch;
  conditional_parameters.conditional.size =
      kLastLength - kFirstLength + 1;
  conditional_parameters.conditional.ctx = nullptr;
  cudaGraphNode_t conditional_node = nullptr;
  CUDA_CHECK(cudaGraphAddNode(
      &conditional_node,
      graph,
      &selector_node,
      nullptr,
      1,
      &conditional_parameters));

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  for (std::int32_t branch = 0;
       branch <= kLastLength - kFirstLength;
       ++branch) {
    const std::int32_t active_length = kFirstLength + branch;
    cudaStream_t body_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&body_stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublas, body_stream));
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(
        body_stream,
        conditional_parameters.conditional.phGraph_out[branch],
        nullptr,
        nullptr,
        0,
        cudaStreamCaptureModeThreadLocal));
    constexpr std::int32_t threads = 256;
    const std::int32_t vectors =
        kBatchHeads * active_length * (kHeadWidth / 8);
    stage_values<<<(vectors + threads - 1) / threads, threads, 0, body_stream>>>(
        reinterpret_cast<const Bf16Vector8*>(values),
        reinterpret_cast<Bf16Vector8*>(compact_values),
        active_length);
    CUDA_CHECK(cudaPeekAtLastError());
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        kHeadWidth,
        1,
        active_length,
        &alpha,
        compact_values,
        CUDA_R_16BF,
        kHeadWidth,
        static_cast<long long>(active_length * kHeadWidth),
        probabilities,
        CUDA_R_16BF,
        active_length,
        static_cast<long long>(active_length),
        &beta,
        output,
        CUDA_R_16BF,
        kHeadWidth,
        static_cast<long long>(kHeadWidth),
        kBatchHeads,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaGraph_t captured_body = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(body_stream, &captured_body));
    if (captured_body !=
        conditional_parameters.conditional.phGraph_out[branch]) {
      std::fprintf(stderr, "conditional body graph changed\n");
      return 4;
    }
    CUDA_CHECK(cudaStreamDestroy(body_stream));
  }

  cudaGraphNode_t mutable_conditional_node = conditional_node;
  CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      stream,
      &mutable_conditional_node,
      nullptr,
      1,
      cudaStreamSetCaptureDependencies));
  cudaGraph_t captured = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<__nv_bfloat16> actual(expected.size());
  CUDA_CHECK(cudaMemcpyAsync(
      actual.data(),
      output,
      actual.size() * sizeof(*output),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::size_t mismatch_count = 0;
  float maximum_absolute_error = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (reinterpret_cast<const std::uint16_t*>(actual.data())[index] !=
        reinterpret_cast<const std::uint16_t*>(expected.data())[index]) {
      ++mismatch_count;
    }
    const float error = std::abs(
        __bfloat162float(actual[index]) - __bfloat162float(expected[index]));
    maximum_absolute_error =
        maximum_absolute_error > error ? maximum_absolute_error : error;
  }
  std::printf(
      "schema=main-device-position-conditional-pv-v1 "
      "position=%lld active_length=%d mismatch=%zu/%zu max_abs=%.9g\n",
      static_cast<long long>(host_position),
      kLastLength,
      mismatch_count,
      actual.size(),
      maximum_absolute_error);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(captured));
  CUBLAS_CHECK(cublasDestroy(cublas));
  CUDA_CHECK(cudaFree(cublas_workspace));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(compact_values));
  CUDA_CHECK(cudaFree(probabilities));
  CUDA_CHECK(cudaFree(values));
  CUDA_CHECK(cudaFree(position));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return mismatch_count == 0 ? 0 : 5;
}

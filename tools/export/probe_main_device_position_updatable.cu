#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

constexpr std::int32_t kBatchHeads = 24;
constexpr std::int32_t kCacheCapacity = 467;
constexpr std::int32_t kHeadWidth = 64;
constexpr std::int32_t kCapturedLength = 219;
constexpr std::int32_t kUpdatedLength = 220;
constexpr std::int32_t kMaximumParameters = 64;
constexpr std::size_t kMaximumParameterBytes = 4096;
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

#define DRIVER_CHECK(expression)                                               \
  do {                                                                         \
    const CUresult status = (expression);                                      \
    if (status != CUDA_SUCCESS) {                                              \
      const char* message = nullptr;                                           \
      static_cast<void>(cuGetErrorString(status, &message));                   \
      std::fprintf(                                                            \
          stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,             \
          #expression, message == nullptr ? "unknown" : message);            \
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

struct DeviceUpdate {
  cudaGraphDeviceNode_t node;
  dim3 grid;
  std::int32_t parameter_count;
  std::size_t offsets[kMaximumParameters];
  std::size_t sizes[kMaximumParameters];
  alignas(16) std::byte values[kMaximumParameterBytes];
  std::int32_t* status;
};

__global__ void update_target(DeviceUpdate* update) {
  cudaError_t first_error = cudaSuccess;
  for (std::int32_t index = 0; index < update->parameter_count; ++index) {
    const cudaError_t status = cudaGraphKernelNodeSetParam(
        update->node,
        update->offsets[index],
        update->values + update->offsets[index],
        update->sizes[index]);
    if (first_error == cudaSuccess && status != cudaSuccess) {
      first_error = status;
    }
  }
  const cudaError_t grid_status =
      cudaGraphKernelNodeSetGridDim(update->node, update->grid);
  if (first_error == cudaSuccess && grid_status != cudaSuccess) {
    first_error = grid_status;
  }
  update->status[0] = static_cast<std::int32_t>(first_error);
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

cudaGraphNode_t single_library_kernel(cudaGraph_t graph) {
  std::size_t node_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  cudaGraphNode_t result = nullptr;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeCount;
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) {
      continue;
    }
    std::size_t dependency_count = 0;
    CUDA_CHECK(cudaGraphNodeGetDependencies(
        node, nullptr, nullptr, &dependency_count));
    if (node_count > 1 && dependency_count == 0) {
      continue;
    }
    if (result != nullptr) {
      std::fprintf(stderr, "more than one library kernel found\n");
      std::exit(1);
    }
    result = node;
  }
  if (result == nullptr) {
    std::fprintf(stderr, "library kernel node not found\n");
    std::exit(1);
  }
  return result;
}

cudaKernelNodeParamsV2 library_parameters(const cudaGraphNode_t node) {
  cudaGraphNodeParams parameters{};
  CUDA_CHECK(cudaGraphNodeGetParams(node, &parameters));
  if (parameters.type != cudaGraphNodeTypeKernel) {
    std::fprintf(stderr, "library node is not a kernel\n");
    std::exit(1);
  }
  return parameters.kernel;
}

bool same_function(
    const cudaKernelNodeParamsV2& left,
    const cudaKernelNodeParamsV2& right) {
  if (left.functionType != right.functionType) {
    return false;
  }
  if (left.functionType == cudaKernelFunctionTypeFunction) {
    return left.cuFunc == right.cuFunc;
  }
  if (left.functionType == cudaKernelFunctionTypeKernel) {
    return left.kern == right.kern;
  }
  return left.func == right.func;
}

std::size_t parameter_count(const cudaKernelNodeParamsV2& parameters) {
  std::size_t count = 0;
  if (parameters.functionType == cudaKernelFunctionTypeFunction) {
    DRIVER_CHECK(cuFuncGetParamCount(
        reinterpret_cast<CUfunction>(parameters.cuFunc), &count));
    return count;
  }
  if (parameters.functionType == cudaKernelFunctionTypeKernel) {
    DRIVER_CHECK(cuKernelGetParamCount(
        reinterpret_cast<CUkernel>(parameters.kern), &count));
    return count;
  }
  std::fprintf(stderr, "unsupported captured function type\n");
  std::exit(1);
}

void parameter_info(
    const cudaKernelNodeParamsV2& parameters,
    const std::size_t index,
    std::size_t* offset,
    std::size_t* size) {
  if (parameters.functionType == cudaKernelFunctionTypeFunction) {
    DRIVER_CHECK(cuFuncGetParamInfo(
        reinterpret_cast<CUfunction>(parameters.cuFunc),
        index,
        offset,
        size));
    return;
  }
  if (parameters.functionType == cudaKernelFunctionTypeKernel) {
    DRIVER_CHECK(cuKernelGetParamInfo(
        reinterpret_cast<CUkernel>(parameters.kern),
        index,
        offset,
        size));
    return;
  }
  std::fprintf(stderr, "unsupported captured function type\n");
  std::exit(1);
}

cudaGraph_t capture_reference(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const __nv_bfloat16* compact_values,
    const __nv_bfloat16* probabilities,
    __nv_bfloat16* output,
    const std::int32_t active_length,
    const bool include_updater,
    DeviceUpdate* update) {
  CUBLAS_CHECK(cublasSetStream(cublas, stream));
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  if (include_updater) {
    update_target<<<1, 1, 0, stream>>>(update);
    CUDA_CHECK(cudaPeekAtLastError());
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
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
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  return graph;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(
        stderr,
        "usage: %s VALUE_BF16 PROBABILITIES_K219_BF16\n",
        argv[0]);
    return 2;
  }
  const std::vector<__nv_bfloat16> host_values = read_exact<__nv_bfloat16>(
      argv[1],
      static_cast<std::size_t>(kBatchHeads) * kCacheCapacity * kHeadWidth);
  const std::vector<__nv_bfloat16> probabilities_219 =
      read_exact<__nv_bfloat16>(
          argv[2],
          static_cast<std::size_t>(kBatchHeads) * kCapturedLength);
  std::vector<__nv_bfloat16> host_probabilities(
      static_cast<std::size_t>(kBatchHeads) * kUpdatedLength);
  for (std::int32_t row = 0; row < kBatchHeads; ++row) {
    const __nv_bfloat16* source =
        probabilities_219.data() +
        static_cast<std::size_t>(row) * kCapturedLength;
    __nv_bfloat16* destination =
        host_probabilities.data() +
        static_cast<std::size_t>(row) * kUpdatedLength;
    std::copy(source, source + kCapturedLength, destination);
    destination[kCapturedLength] = source[row % kCapturedLength];
  }

  cudaStream_t stream = nullptr;
  cudaStream_t reference_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(
      &reference_stream, cudaStreamNonBlocking));
  __nv_bfloat16* values = nullptr;
  __nv_bfloat16* compact_values = nullptr;
  __nv_bfloat16* probabilities = nullptr;
  __nv_bfloat16* graph_output = nullptr;
  __nv_bfloat16* direct_output = nullptr;
  void* cublas_workspace = nullptr;
  std::int32_t* update_status = nullptr;
  DeviceUpdate* update = nullptr;
  CUDA_CHECK(cudaMalloc(&values, host_values.size() * sizeof(*values)));
  CUDA_CHECK(cudaMalloc(
      &compact_values,
      static_cast<std::size_t>(kBatchHeads) * kUpdatedLength * kHeadWidth *
          sizeof(*compact_values)));
  CUDA_CHECK(cudaMalloc(
      &probabilities, host_probabilities.size() * sizeof(*probabilities)));
  CUDA_CHECK(cudaMalloc(
      &graph_output,
      static_cast<std::size_t>(kBatchHeads) * kHeadWidth *
          sizeof(*graph_output)));
  CUDA_CHECK(cudaMalloc(
      &direct_output,
      static_cast<std::size_t>(kBatchHeads) * kHeadWidth *
          sizeof(*direct_output)));
  CUDA_CHECK(cudaMalloc(&cublas_workspace, kCublasWorkspaceBytes));
  CUDA_CHECK(cudaMalloc(&update_status, sizeof(*update_status)));
  CUDA_CHECK(cudaMallocManaged(&update, sizeof(*update)));
  std::memset(update, 0, sizeof(*update));
  update->status = update_status;
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
  constexpr std::int32_t threads = 256;
  constexpr std::int32_t vectors =
      kBatchHeads * kUpdatedLength * (kHeadWidth / 8);
  stage_values<<<(vectors + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const Bf16Vector8*>(values),
      reinterpret_cast<Bf16Vector8*>(compact_values),
      kUpdatedLength);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cublasHandle_t cublas = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUBLAS_CHECK(cublasSetWorkspace(
      cublas, cublas_workspace, kCublasWorkspaceBytes));
  cudaGraph_t target_graph = capture_reference(
      cublas,
      stream,
      compact_values,
      probabilities,
      graph_output,
      kCapturedLength,
      true,
      update);
  cudaGraph_t reference_graph = capture_reference(
      cublas,
      reference_stream,
      compact_values,
      probabilities,
      graph_output,
      kUpdatedLength,
      false,
      update);
  const cudaGraphNode_t target_node = single_library_kernel(target_graph);
  const cudaGraphNode_t reference_node =
      single_library_kernel(reference_graph);
  const cudaKernelNodeParamsV2 target_parameters =
      library_parameters(target_node);
  const cudaKernelNodeParamsV2 reference_parameters =
      library_parameters(reference_node);
  if (!same_function(target_parameters, reference_parameters) ||
      target_parameters.blockDim.x != reference_parameters.blockDim.x ||
      target_parameters.blockDim.y != reference_parameters.blockDim.y ||
      target_parameters.blockDim.z != reference_parameters.blockDim.z ||
      target_parameters.sharedMemBytes !=
          reference_parameters.sharedMemBytes) {
    std::fprintf(stderr, "K219 and K220 selected different kernels\n");
    return 3;
  }
  if (target_parameters.kernelParams == nullptr ||
      reference_parameters.kernelParams == nullptr) {
    std::fprintf(stderr, "captured cuBLAS node has no kernelParams\n");
    return 4;
  }
  const std::size_t captured_parameter_count =
      parameter_count(reference_parameters);
  if (captured_parameter_count > kMaximumParameters) {
    std::fprintf(
        stderr, "too many parameters: %zu\n", captured_parameter_count);
    return 5;
  }
  update->parameter_count =
      static_cast<std::int32_t>(captured_parameter_count);
  std::size_t packed_bytes = 0;
  for (std::size_t index = 0;
       index < captured_parameter_count;
       ++index) {
    std::size_t offset = 0;
    std::size_t size = 0;
    parameter_info(
        reference_parameters, index, &offset, &size);
    if (offset + size > kMaximumParameterBytes) {
      std::fprintf(stderr, "parameter ABI exceeds fixed buffer\n");
      return 6;
    }
    update->offsets[index] = offset;
    update->sizes[index] = size;
    std::memcpy(
        update->values + offset,
        reference_parameters.kernelParams[index],
        size);
    packed_bytes = std::max(packed_bytes, offset + size);
  }
  update->grid = reference_parameters.gridDim;
  cudaKernelNodeAttrValue attribute{};
  attribute.deviceUpdatableKernelNode.deviceUpdatable = 1;
  CUDA_CHECK(cudaGraphKernelNodeSetAttribute(
      target_node,
      cudaLaunchAttributeDeviceUpdatableKernelNode,
      &attribute));
  update->node = attribute.deviceUpdatableKernelNode.devNode;
  if (update->node == nullptr) {
    std::fprintf(stderr, "device node handle was not returned\n");
    return 7;
  }

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  CUBLAS_CHECK(cublasSetStream(cublas, reference_stream));
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      cublas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      kHeadWidth,
      1,
      kUpdatedLength,
      &alpha,
      compact_values,
      CUDA_R_16BF,
      kHeadWidth,
      static_cast<long long>(kUpdatedLength * kHeadWidth),
      probabilities,
      CUDA_R_16BF,
      kUpdatedLength,
      static_cast<long long>(kUpdatedLength),
      &beta,
      direct_output,
      CUDA_R_16BF,
      kHeadWidth,
      static_cast<long long>(kHeadWidth),
      kBatchHeads,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CUDA_CHECK(cudaStreamSynchronize(reference_stream));

  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(
      &executable, target_graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphUpload(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<__nv_bfloat16> actual(
      static_cast<std::size_t>(kBatchHeads) * kHeadWidth);
  std::vector<__nv_bfloat16> expected(actual.size());
  std::int32_t host_update_status = -1;
  CUDA_CHECK(cudaMemcpyAsync(
      actual.data(),
      graph_output,
      actual.size() * sizeof(*graph_output),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      &host_update_status,
      update_status,
      sizeof(host_update_status),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      expected.data(),
      direct_output,
      expected.size() * sizeof(*direct_output),
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
        std::max(maximum_absolute_error, error);
  }
  std::printf(
      "schema=main-device-position-updatable-pv-v1 "
      "captured_k=%d updated_k=%d param_count=%zu param_bytes=%zu "
      "grid=%ux%ux%u update_status=%d mismatch=%zu/%zu max_abs=%.9g\n",
      kCapturedLength,
      kUpdatedLength,
      captured_parameter_count,
      packed_bytes,
      reference_parameters.gridDim.x,
      reference_parameters.gridDim.y,
      reference_parameters.gridDim.z,
      host_update_status,
      mismatch_count,
      actual.size(),
      maximum_absolute_error);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(reference_graph));
  CUDA_CHECK(cudaGraphDestroy(target_graph));
  CUBLAS_CHECK(cublasDestroy(cublas));
  CUDA_CHECK(cudaFree(update));
  CUDA_CHECK(cudaFree(update_status));
  CUDA_CHECK(cudaFree(cublas_workspace));
  CUDA_CHECK(cudaFree(direct_output));
  CUDA_CHECK(cudaFree(graph_output));
  CUDA_CHECK(cudaFree(probabilities));
  CUDA_CHECK(cudaFree(compact_values));
  CUDA_CHECK(cudaFree(values));
  CUDA_CHECK(cudaStreamDestroy(reference_stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return host_update_status == static_cast<std::int32_t>(cudaSuccess) &&
                 mismatch_count == 0
             ? 0
             : 8;
}

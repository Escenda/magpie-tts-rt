#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace {

constexpr std::int32_t kBatchHeads = 24;
constexpr std::int32_t kHeadWidth = 64;
constexpr std::int32_t kMinimumLength = 219;
constexpr std::int32_t kMaximumLength = 467;
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

enum class Operation : std::uint32_t { kQk = 1, kPv = 2 };

struct TableRecordHeader {
  std::uint32_t operation;
  std::uint32_t active_length;
  std::uint64_t function;
  std::uint32_t function_type;
  std::uint32_t grid[3];
  std::uint32_t block[3];
  std::uint32_t shared_memory;
  std::uint32_t parameter_count;
  std::uint32_t parameter_bytes;
};

struct CapturedParameters {
  cudaGraph_t graph;
  cudaKernelNodeParamsV2 kernel;
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> sizes;
  std::vector<std::byte> packed;
};

cudaGraphNode_t only_kernel(cudaGraph_t graph) {
  std::size_t count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
  cudaGraphNode_t result = nullptr;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeCount;
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) {
      continue;
    }
    if (result != nullptr) {
      std::fprintf(stderr, "captured cuBLAS graph has multiple kernels\n");
      std::exit(1);
    }
    result = node;
  }
  if (result == nullptr) {
    std::fprintf(stderr, "captured cuBLAS graph has no kernel\n");
    std::exit(1);
  }
  return result;
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

std::uint64_t function_identity(const cudaKernelNodeParamsV2& parameters) {
  if (parameters.functionType == cudaKernelFunctionTypeFunction) {
    return reinterpret_cast<std::uint64_t>(parameters.cuFunc);
  }
  if (parameters.functionType == cudaKernelFunctionTypeKernel) {
    return reinterpret_cast<std::uint64_t>(parameters.kern);
  }
  return reinterpret_cast<std::uint64_t>(parameters.func);
}

CapturedParameters capture(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const Operation operation,
    const std::int32_t active_length,
    const __nv_bfloat16* first,
    const __nv_bfloat16* second,
    __nv_bfloat16* output) {
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  CUBLAS_CHECK(cublasSetStream(cublas, stream));
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  if (operation == Operation::kQk) {
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        active_length,
        1,
        kHeadWidth,
        &alpha,
        first,
        CUDA_R_16BF,
        active_length,
        static_cast<long long>(kHeadWidth * active_length),
        second,
        CUDA_R_16BF,
        kHeadWidth,
        static_cast<long long>(kHeadWidth),
        &beta,
        output,
        CUDA_R_16BF,
        active_length,
        static_cast<long long>(active_length),
        kBatchHeads,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        kHeadWidth,
        1,
        active_length,
        &alpha,
        first,
        CUDA_R_16BF,
        kHeadWidth,
        static_cast<long long>(active_length * kHeadWidth),
        second,
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
  }
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  const cudaGraphNode_t node = only_kernel(graph);
  cudaGraphNodeParams generic{};
  CUDA_CHECK(cudaGraphNodeGetParams(node, &generic));
  if (generic.type != cudaGraphNodeTypeKernel ||
      (generic.kernel.kernelParams == nullptr &&
       generic.kernel.extra == nullptr)) {
    std::fprintf(stderr, "captured node parameter contract mismatch\n");
    std::exit(1);
  }
  CapturedParameters result{graph, generic.kernel, {}, {}, {}};
  const std::size_t count = parameter_count(result.kernel);
  result.offsets.resize(count);
  result.sizes.resize(count);
  std::size_t packed_bytes = 0;
  for (std::size_t index = 0; index < count; ++index) {
    parameter_info(
        result.kernel,
        index,
        &result.offsets[index],
        &result.sizes[index]);
    packed_bytes = std::max(
        packed_bytes,
        result.offsets[index] + result.sizes[index]);
  }
  result.packed.resize(packed_bytes);
  if (result.kernel.kernelParams != nullptr) {
    for (std::size_t index = 0; index < count; ++index) {
      std::memcpy(
          result.packed.data() + result.offsets[index],
          result.kernel.kernelParams[index],
          result.sizes[index]);
    }
  } else {
    const void* parameter_buffer = nullptr;
    std::size_t parameter_buffer_bytes = 0;
    for (std::size_t index = 0;; index += 2) {
      const void* token = result.kernel.extra[index];
      if (token == CU_LAUNCH_PARAM_END) {
        break;
      }
      const void* value = result.kernel.extra[index + 1];
      if (token == CU_LAUNCH_PARAM_BUFFER_POINTER) {
        parameter_buffer = value;
      } else if (token == CU_LAUNCH_PARAM_BUFFER_SIZE) {
        parameter_buffer_bytes =
            *static_cast<const std::size_t*>(value);
      } else {
        std::fprintf(stderr, "unknown captured kernel extra token\n");
        std::exit(1);
      }
    }
    if (parameter_buffer == nullptr ||
        parameter_buffer_bytes < packed_bytes) {
      std::fprintf(stderr, "captured parameter buffer is incomplete\n");
      std::exit(1);
    }
    std::memcpy(result.packed.data(), parameter_buffer, packed_bytes);
  }
  return result;
}

void append_record(
    std::ofstream& table,
    const Operation operation,
    const std::int32_t active_length,
    const CapturedParameters& captured) {
  const TableRecordHeader header{
      static_cast<std::uint32_t>(operation),
      static_cast<std::uint32_t>(active_length),
      function_identity(captured.kernel),
      static_cast<std::uint32_t>(captured.kernel.functionType),
      {captured.kernel.gridDim.x,
       captured.kernel.gridDim.y,
       captured.kernel.gridDim.z},
      {captured.kernel.blockDim.x,
       captured.kernel.blockDim.y,
       captured.kernel.blockDim.z},
      captured.kernel.sharedMemBytes,
      static_cast<std::uint32_t>(captured.offsets.size()),
      static_cast<std::uint32_t>(captured.packed.size()),
  };
  table.write(
      reinterpret_cast<const char*>(&header), sizeof(header));
  for (std::size_t index = 0; index < captured.offsets.size(); ++index) {
    const std::array<std::uint64_t, 2> layout{
        captured.offsets[index], captured.sizes[index]};
    table.write(
        reinterpret_cast<const char*>(layout.data()), sizeof(layout));
  }
  table.write(
      reinterpret_cast<const char*>(captured.packed.data()),
      static_cast<std::streamsize>(captured.packed.size()));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s OUTPUT_TABLE\n", argv[0]);
    return 2;
  }
  std::ofstream table(argv[1], std::ios::binary | std::ios::trunc);
  if (!table) {
    std::fprintf(stderr, "cannot create %s\n", argv[1]);
    return 3;
  }
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  __nv_bfloat16* key_transposed = nullptr;
  __nv_bfloat16* query = nullptr;
  __nv_bfloat16* qk_output = nullptr;
  __nv_bfloat16* compact_values = nullptr;
  __nv_bfloat16* probabilities = nullptr;
  __nv_bfloat16* pv_output = nullptr;
  void* workspace = nullptr;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kBatchHeads) * kMaximumLength * kHeadWidth;
  CUDA_CHECK(cudaMalloc(
      &key_transposed, cache_elements * sizeof(*key_transposed)));
  CUDA_CHECK(cudaMalloc(
      &query,
      static_cast<std::size_t>(kBatchHeads) * kHeadWidth * sizeof(*query)));
  CUDA_CHECK(cudaMalloc(
      &qk_output,
      static_cast<std::size_t>(kBatchHeads) * kMaximumLength *
          sizeof(*qk_output)));
  CUDA_CHECK(cudaMalloc(
      &compact_values, cache_elements * sizeof(*compact_values)));
  CUDA_CHECK(cudaMalloc(
      &probabilities,
      static_cast<std::size_t>(kBatchHeads) * kMaximumLength *
          sizeof(*probabilities)));
  CUDA_CHECK(cudaMalloc(
      &pv_output,
      static_cast<std::size_t>(kBatchHeads) * kHeadWidth *
          sizeof(*pv_output)));
  CUDA_CHECK(cudaMalloc(&workspace, kCublasWorkspaceBytes));
  cublasHandle_t cublas = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUBLAS_CHECK(cublasSetWorkspace(
      cublas, workspace, kCublasWorkspaceBytes));

  for (const Operation operation : {Operation::kQk, Operation::kPv}) {
    for (std::int32_t active_length = kMinimumLength;
         active_length <= kMaximumLength;
         ++active_length) {
      const CapturedParameters captured =
          operation == Operation::kQk
              ? capture(
                    cublas,
                    stream,
                    operation,
                    active_length,
                    key_transposed,
                    query,
                    qk_output)
              : capture(
                    cublas,
                    stream,
                    operation,
                    active_length,
                    compact_values,
                    probabilities,
                    pv_output);
      append_record(table, operation, active_length, captured);
      std::printf(
          "%s,%d,%p,%d,%u,%u,%u,%u,%u,%u,%u,%zu,%zu",
          operation == Operation::kQk ? "qk" : "pv",
          active_length,
          reinterpret_cast<void*>(function_identity(captured.kernel)),
          static_cast<int>(captured.kernel.functionType),
          captured.kernel.gridDim.x,
          captured.kernel.gridDim.y,
          captured.kernel.gridDim.z,
          captured.kernel.blockDim.x,
          captured.kernel.blockDim.y,
          captured.kernel.blockDim.z,
          captured.kernel.sharedMemBytes,
          captured.offsets.size(),
          captured.packed.size());
      for (std::size_t index = 0; index < captured.offsets.size(); ++index) {
        std::printf(
            ",%zu:%zu", captured.offsets[index], captured.sizes[index]);
      }
      std::printf("\n");
      CUDA_CHECK(cudaGraphDestroy(captured.graph));
    }
  }
  table.close();
  if (!table) {
    std::fprintf(stderr, "failed to write %s\n", argv[1]);
    return 4;
  }

  CUBLAS_CHECK(cublasDestroy(cublas));
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(pv_output));
  CUDA_CHECK(cudaFree(probabilities));
  CUDA_CHECK(cudaFree(compact_values));
  CUDA_CHECK(cudaFree(qk_output));
  CUDA_CHECK(cudaFree(query));
  CUDA_CHECK(cudaFree(key_transposed));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

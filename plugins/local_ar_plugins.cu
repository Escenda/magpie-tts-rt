#include "local_ar_plugins.hpp"
#include "magpie_tts_rt/magpie_tts_rt_plugin.h"

#include <NvInfer.h>

#include <cub/block/block_radix_sort.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

namespace magpie_tts_rt::plugins {
namespace {

constexpr std::int32_t kSamplingThreads = 256;
constexpr std::int32_t kItemsPerThread = 8;
constexpr std::int32_t kTopK = 80;
constexpr float kTemperature = 0.6F;
constexpr float kCfgScale = 2.5F;
constexpr std::int64_t kPhiloxStride = 2048;
constexpr std::int64_t kMaximumCounter =
    (std::numeric_limits<std::int64_t>::max() - (kPhiloxStride - 1)) /
    kPhiloxStride;
constexpr std::int32_t kCompactEosVocabulary = kLocalArCodebookSize + 1;
constexpr std::size_t kGuidedEosBytes =
    static_cast<std::size_t>(kLocalArPositions) *
    static_cast<std::size_t>(kCompactEosVocabulary) * sizeof(float);
constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::size_t kAlignedGuidedEosBytes =
    (kGuidedEosBytes + kWorkspaceAlignment - 1) &
    ~(kWorkspaceAlignment - 1);
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "magpie_tts_rt";
constexpr char kSamplingPluginName[] = "MagpieLocalARSampling";
constexpr char kEosPluginName[] = "MagpieLocalAREos";
constexpr char kLayerNormPluginName[] = "MagpieLayerNorm";
constexpr char kGeluTanhPluginName[] = "MagpieGeluTanh";
constexpr char kSoftmaxPluginName[] = "MagpieSoftmax";
constexpr std::int32_t kSoftmaxMode = 0;
constexpr std::int32_t kMainCrossAttentionSoftmaxMode = 1;
constexpr std::int32_t kMainSelfAttentionContextMode = 2;
constexpr std::int32_t kMainCrossAttentionContextMode = 3;
constexpr std::int32_t kMainAttentionPriorNormalizationMode = 4;
constexpr std::int32_t kMainSelfAttentionStepScoresMode = 5;
constexpr std::int32_t kMainSelfAttentionStepContextMode = 6;
constexpr std::int32_t kMainAlignmentMeanMode = 7;
constexpr std::int32_t kNormalizationWidth = kLocalArEmbeddingWidth;
constexpr float kLayerNormEpsilon = 1.0e-5F;
constexpr std::int32_t kLayerNormWarpSize = 32;
constexpr std::int32_t kLayerNormWarps = 4;
constexpr std::int32_t kGeluThreads = 256;
constexpr std::int32_t kMainCrossAttentionThreads = 256;
constexpr float kMainCrossAttentionScale = 0.08838834764831845F;
constexpr std::size_t kMainCrossAttentionWorkspaceBytes =
    static_cast<std::size_t>(kMainDecoderBatch) *
    static_cast<std::size_t>(kMainDecoderPrefillLength) *
    static_cast<std::size_t>(kMaximumTextSequenceLength) *
    sizeof(__nv_bfloat16);
constexpr std::int32_t kMainSelfAttentionHeadWidth =
    kMainDecoderCrossAttentionWidth / 2;
constexpr std::int32_t kMainSelfAttentionBatchHeads =
    kMainDecoderBatch * kAttentionHeads;
constexpr std::size_t kMainStepValueWorkspaceBytes =
    static_cast<std::size_t>(kMainSelfAttentionBatchHeads) *
    static_cast<std::size_t>(kMainDecoderCacheCapacity) *
    static_cast<std::size_t>(kMainSelfAttentionHeadWidth) *
    sizeof(__nv_bfloat16);
constexpr std::size_t kMainStepContextWorkspaceBytes =
    kMainStepValueWorkspaceBytes;

using MainSelfAttentionContextKernel =
    typename cutlass::gemm::kernel::DefaultGemmUniversal<
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        2,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        2,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            cutlass::bfloat16_t,
            2,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        6,
        cutlass::arch::OpMultiplyAdd>::GemmKernel;

using MainSelfAttentionContextGemm =
    cutlass::gemm::device::GemmUniversalAdapter<
        MainSelfAttentionContextKernel>;

using MainCrossAttentionContextKernel =
    typename cutlass::gemm::kernel::DefaultGemmUniversal<
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        1,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        1,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            cutlass::bfloat16_t,
            1,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        2,
        cutlass::arch::OpMultiplyAdd>::GemmKernel;

using MainCrossAttentionContextGemm =
    cutlass::gemm::device::GemmUniversalAdapter<
        MainCrossAttentionContextKernel>;

struct WelfordData final {
  float mean;
  float sigma2;
  float count;
};

template <typename Element, std::int32_t Size>
struct alignas(sizeof(Element) * Size) AlignedVector final {
  Element values[Size];
};

__global__ void stage_main_step_values_kernel(
    const AlignedVector<__nv_bfloat16, 8>* capacity_stride_values,
    AlignedVector<__nv_bfloat16, 8>* compact_values,
    const std::int32_t active_length) {
  constexpr std::int32_t vectors_per_token =
      kMainSelfAttentionHeadWidth / 8;
  const std::int32_t linear_index =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t vectors_per_matrix =
      active_length * vectors_per_token;
  const std::int32_t vector_count =
      kMainSelfAttentionBatchHeads * vectors_per_matrix;
  if (linear_index >= vector_count) {
    return;
  }
  const std::int32_t matrix = linear_index / vectors_per_matrix;
  const std::int32_t matrix_vector = linear_index % vectors_per_matrix;
  const std::int32_t token = matrix_vector / vectors_per_token;
  const std::int32_t token_vector = matrix_vector % vectors_per_token;
  const std::int32_t source_index =
      (matrix * kMainDecoderCacheCapacity + token) *
          vectors_per_token +
      token_vector;
  compact_values[linear_index] =
      capacity_stride_values[source_index];
}

[[nodiscard]] __device__ WelfordData welford_online_sum(
    const float value,
    const WelfordData current) {
  const float delta = value - current.mean;
  const float new_count = current.count + 1.0F;
  const float new_mean = current.mean + delta * (1.0F / new_count);
  return {
      new_mean,
      current.sigma2 + delta * (value - new_mean),
      new_count};
}

[[nodiscard]] __device__ WelfordData welford_combine(
    const WelfordData data_b,
    const WelfordData data_a) {
  const float delta = data_b.mean - data_a.mean;
  const float count = data_a.count + data_b.count;
  if (count <= 0.0F) {
    return {0.0F, 0.0F, 0.0F};
  }
  const float coefficient = 1.0F / count;
  const float n_a = data_a.count * coefficient;
  const float n_b = data_b.count * coefficient;
  return {
      n_a * data_a.mean + n_b * data_b.mean,
      data_a.sigma2 + data_b.sigma2 +
          delta * delta * data_a.count * n_b,
      count};
}

[[nodiscard]] __device__ WelfordData layer_norm_stats(
    const __nv_bfloat16* input,
    float* shared) {
  using Vector = AlignedVector<__nv_bfloat16, 4>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(input);
  constexpr std::int32_t vector_count = kNormalizationWidth / 4;
  constexpr std::int32_t thread_count =
      kLayerNormWarpSize * kLayerNormWarps;
  const std::int32_t linear_thread =
      static_cast<std::int32_t>(threadIdx.x) +
      static_cast<std::int32_t>(threadIdx.y) * kLayerNormWarpSize;
  WelfordData value{0.0F, 0.0F, 0.0F};
  for (std::int32_t vector_index = linear_thread;
       vector_index < vector_count;
       vector_index += thread_count) {
    const Vector vector = input_vectors[vector_index];
#pragma unroll
    for (std::int32_t element = 0; element < 4; ++element) {
      value = welford_online_sum(
          __bfloat162float(vector.values[element]), value);
    }
  }

  constexpr unsigned int full_warp_mask = 0xFFFFFFFFU;
  for (std::int32_t offset = kLayerNormWarpSize / 2; offset > 0;
       offset >>= 1) {
    const WelfordData other{
        __shfl_down_sync(full_warp_mask, value.mean, offset),
        __shfl_down_sync(full_warp_mask, value.sigma2, offset),
        __shfl_down_sync(full_warp_mask, value.count, offset)};
    value = welford_combine(value, other);
  }

  if (blockDim.y > 1) {
    float* mean_sigma = shared;
    float* count = shared + blockDim.y;
    for (std::int32_t offset = blockDim.y / 2; offset > 0; offset /= 2) {
      if (threadIdx.x == 0 && threadIdx.y >= offset &&
          threadIdx.y < 2 * offset) {
        const std::int32_t write_y =
            static_cast<std::int32_t>(threadIdx.y) - offset;
        mean_sigma[2 * write_y] = value.mean;
        mean_sigma[2 * write_y + 1] = value.sigma2;
        count[write_y] = value.count;
      }
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y < offset) {
        const std::int32_t read_y =
            static_cast<std::int32_t>(threadIdx.y);
        const WelfordData other{
            mean_sigma[2 * read_y],
            mean_sigma[2 * read_y + 1],
            count[read_y]};
        value = welford_combine(value, other);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean_sigma[0] = value.mean;
      mean_sigma[1] =
          value.sigma2 / static_cast<float>(kNormalizationWidth);
    }
    __syncthreads();
    return {mean_sigma[0], mean_sigma[1], 0.0F};
  }

  return {
      __shfl_sync(full_warp_mask, value.mean, 0),
      __shfl_sync(full_warp_mask, value.sigma2, 0) /
          static_cast<float>(kNormalizationWidth),
      0.0F};
}

__global__ void layer_norm_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output) {
  extern __shared__ float shared[];
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  const __nv_bfloat16* row_input =
      input + row * kNormalizationWidth;
  WelfordData stats = layer_norm_stats(row_input, shared);
  const float reciprocal_standard_deviation =
      rsqrtf(stats.sigma2 + kLayerNormEpsilon);

  using Vector = AlignedVector<__nv_bfloat16, 4>;
  const auto* input_vectors =
      reinterpret_cast<const Vector*>(row_input);
  const auto* weight_vectors =
      reinterpret_cast<const Vector*>(weight);
  auto* output_vectors = reinterpret_cast<Vector*>(
      output + row * kNormalizationWidth);
  constexpr std::int32_t vector_count = kNormalizationWidth / 4;
  const std::int32_t linear_thread =
      static_cast<std::int32_t>(threadIdx.x) +
      static_cast<std::int32_t>(threadIdx.y) * kLayerNormWarpSize;
  const std::int32_t thread_count =
      static_cast<std::int32_t>(blockDim.x * blockDim.y);
  for (std::int32_t vector_index = linear_thread;
       vector_index < vector_count;
       vector_index += thread_count) {
    const Vector input_vector = input_vectors[vector_index];
    const Vector weight_vector = weight_vectors[vector_index];
    Vector output_vector{};
#pragma unroll
    for (std::int32_t element = 0; element < 4; ++element) {
      const float normalized =
          reciprocal_standard_deviation *
          (__bfloat162float(input_vector.values[element]) - stats.mean);
      output_vector.values[element] = __float2bfloat16_rn(
          __bfloat162float(weight_vector.values[element]) * normalized);
    }
    output_vectors[vector_index] = output_vector;
  }
}

__global__ void gelu_tanh_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const std::size_t element_count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  constexpr float beta = 0.7978845608028654F;
  constexpr float kappa = 0.044715F;
  const float value = __bfloat162float(input[index]);
  const float value_cube = value * value * value;
  const float inner = beta * (value + kappa * value_cube);
  output[index] = __float2bfloat16_rn(
      0.5F * value * (1.0F + tanhf(inner)));
}

template <std::int32_t WarpBatch, std::int32_t WarpSize>
__device__ void softmax_warp_reduce_max(float (&values)[WarpBatch]) {
#pragma unroll
  for (std::int32_t offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (std::int32_t batch = 0; batch < WarpBatch; ++batch) {
      const float other =
          __shfl_xor_sync(0xFFFFFFFFU, values[batch], offset, WarpSize);
      values[batch] =
          values[batch] < other ? other : values[batch];
    }
  }
}

template <std::int32_t WarpBatch, std::int32_t WarpSize>
__device__ void softmax_warp_reduce_sum(float (&values)[WarpBatch]) {
#pragma unroll
  for (std::int32_t offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (std::int32_t batch = 0; batch < WarpBatch; ++batch) {
      values[batch] +=
          __shfl_xor_sync(0xFFFFFFFFU, values[batch], offset, WarpSize);
    }
  }
}

// This is the inference-only BF16/FP32 specialization of PyTorch 2.11
// PersistentSoftmax.cuh. Keeping its warp layout, padding, reduction order,
// exponentiation, and BF16 store is required for the accepted oracle bytes.
template <std::int32_t Log2Elements>
__global__ void persistent_softmax_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const std::int32_t batch_count,
    const std::int32_t element_count) {
  constexpr std::int32_t next_power_of_two = 1 << Log2Elements;
  constexpr std::int32_t warp_size =
      next_power_of_two < 32 ? next_power_of_two : 32;
  constexpr std::int32_t warp_iterations =
      next_power_of_two / warp_size;
  constexpr std::int32_t warp_batch =
      next_power_of_two <= 128 ? 2 : 1;

  const std::int32_t first_batch =
      (static_cast<std::int32_t>(blockDim.y) *
           static_cast<std::int32_t>(blockIdx.x) +
       static_cast<std::int32_t>(threadIdx.y)) *
      warp_batch;
  std::int32_t local_batches = batch_count - first_batch;
  if (local_batches > warp_batch) {
    local_batches = warp_batch;
  }
  const std::int32_t local_index =
      static_cast<std::int32_t>(threadIdx.x);
  const std::int32_t offset =
      first_batch * element_count + local_index;
  input += offset;
  output += offset;

  float elements[warp_batch][warp_iterations];
#pragma unroll
  for (std::int32_t batch = 0; batch < warp_batch; ++batch) {
    const std::int32_t batch_element_count =
        batch >= local_batches ? 0 : element_count;
#pragma unroll
    for (std::int32_t iteration = 0; iteration < warp_iterations;
         ++iteration) {
      const std::int32_t element_index =
          local_index + iteration * warp_size;
      elements[batch][iteration] =
          element_index < batch_element_count
              ? __bfloat162float(
                    input[batch * element_count +
                          iteration * warp_size])
              : -CUDART_INF_F;
    }
  }

  float maximum[warp_batch];
#pragma unroll
  for (std::int32_t batch = 0; batch < warp_batch; ++batch) {
    maximum[batch] = elements[batch][0];
#pragma unroll
    for (std::int32_t iteration = 0; iteration < warp_iterations;
         ++iteration) {
      maximum[batch] =
          maximum[batch] > elements[batch][iteration]
              ? maximum[batch]
              : elements[batch][iteration];
    }
  }
  softmax_warp_reduce_max<warp_batch, warp_size>(maximum);

  float sum[warp_batch]{0.0F};
#pragma unroll
  for (std::int32_t batch = 0; batch < warp_batch; ++batch) {
#pragma unroll
    for (std::int32_t iteration = 0; iteration < warp_iterations;
         ++iteration) {
      elements[batch][iteration] =
          expf(elements[batch][iteration] - maximum[batch]);
      sum[batch] += elements[batch][iteration];
    }
  }
  softmax_warp_reduce_sum<warp_batch, warp_size>(sum);

#pragma unroll
  for (std::int32_t batch = 0; batch < warp_batch; ++batch) {
    if (batch >= local_batches) {
      break;
    }
#pragma unroll
    for (std::int32_t iteration = 0; iteration < warp_iterations;
         ++iteration) {
      const std::int32_t element_index =
          local_index + iteration * warp_size;
      if (element_index < element_count) {
        const float value =
            sum[batch] == 0.0F
                ? CUDART_NAN_F
                : elements[batch][iteration] / sum[batch];
        output[batch * element_count + iteration * warp_size] =
            __float2bfloat16_rn(value);
      }
    }
  }
}

__global__ void main_cross_attention_scale_mask_kernel(
    const bool* memory_mask,
    __nv_bfloat16* scores,
    const std::int32_t query_length,
    const std::int32_t memory_length) {
  const std::int32_t index =
      static_cast<std::int32_t>(blockIdx.x) *
          static_cast<std::int32_t>(blockDim.x) +
      static_cast<std::int32_t>(threadIdx.x);
  const std::int32_t element_count =
      kMainDecoderBatch * query_length * memory_length;
  if (index >= element_count) {
    return;
  }
  const std::int32_t memory_index = index % memory_length;
  const std::int32_t query_and_batch = index / memory_length;
  const std::int32_t batch_index = query_and_batch / query_length;
  if (!memory_mask[batch_index * memory_length + memory_index]) {
    scores[index] = __float2bfloat16_rn(-CUDART_INF_F);
    return;
  }
  const __nv_bfloat16 rounded_dot = scores[index];
  scores[index] = __float2bfloat16_rn(
      __bfloat162float(rounded_dot) * kMainCrossAttentionScale);
}

__global__ void main_attention_prior_normalization_kernel(
    const __nv_bfloat16* probabilities,
    const __nv_bfloat16* attention_prior,
    __nv_bfloat16* output,
    const std::int32_t memory_length) {
  __shared__ float warp_sums[8];
  __shared__ __nv_bfloat16 rounded_denominator;
  const std::int32_t batch_index =
      static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t thread_index =
      static_cast<std::int32_t>(threadIdx.x);
  const std::int32_t batch_offset =
      batch_index * memory_length;
  float local_sum = 0.0F;
  for (std::int32_t memory_index = thread_index;
       memory_index < memory_length;
       memory_index += static_cast<std::int32_t>(blockDim.x)) {
    const std::int32_t index = batch_offset + memory_index;
    const __nv_bfloat16 rounded_prior = __float2bfloat16_rn(
        __bfloat162float(attention_prior[index]) +
        0x1p-126F);
    const __nv_bfloat16 numerator = __float2bfloat16_rn(
        __bfloat162float(probabilities[index]) *
        __bfloat162float(rounded_prior));
    output[index] = numerator;
    local_sum += __bfloat162float(numerator);
  }
  constexpr unsigned int full_warp_mask = 0xFFFFFFFFU;
  for (std::int32_t offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(
        full_warp_mask, local_sum, offset);
  }
  const std::int32_t lane = thread_index & 31;
  const std::int32_t warp = thread_index >> 5;
  if (lane == 0) {
    warp_sums[warp] = local_sum;
  }
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < 8 ? warp_sums[lane] : 0.0F;
    for (std::int32_t offset = 16; offset > 0; offset >>= 1) {
      block_sum += __shfl_down_sync(
          full_warp_mask, block_sum, offset);
    }
    if (lane == 0) {
      rounded_denominator = __float2bfloat16_rn(block_sum);
    }
  }
  __syncthreads();
  const float denominator =
      __bfloat162float(rounded_denominator);
  for (std::int32_t memory_index = thread_index;
       memory_index < memory_length;
       memory_index += static_cast<std::int32_t>(blockDim.x)) {
    const std::int32_t index = batch_offset + memory_index;
    output[index] = __float2bfloat16_rn(
        __bfloat162float(output[index]) / denominator);
  }
}

__global__ void main_alignment_mean_kernel(
    const __nv_bfloat16* alignment_0,
    const __nv_bfloat16* alignment_1,
    const __nv_bfloat16* alignment_2,
    const __nv_bfloat16* alignment_3,
    __nv_bfloat16* output,
    const std::int32_t element_count) {
  const std::int32_t index =
      static_cast<std::int32_t>(blockIdx.x) *
          static_cast<std::int32_t>(blockDim.x) +
      static_cast<std::int32_t>(threadIdx.x);
  if (index >= element_count) {
    return;
  }
  float sum = __bfloat162float(alignment_0[index]);
  sum += __bfloat162float(alignment_1[index]);
  sum += __bfloat162float(alignment_2[index]);
  sum += __bfloat162float(alignment_3[index]);
  output[index] = __float2bfloat16_rn(sum * 0.25F);
}

[[nodiscard]] __device__ std::uint32_t philox_word(
    const std::uint32_t seed,
    const std::uint64_t offset) {
  std::uint32_t c0 = static_cast<std::uint32_t>(offset);
  std::uint32_t c1 = static_cast<std::uint32_t>(offset >> 32U);
  std::uint32_t c2 = 0;
  std::uint32_t c3 = 0;
  std::uint32_t k0 = seed;
  std::uint32_t k1 = 0;
  for (std::int32_t round = 0; round < 10; ++round) {
    const std::uint32_t previous_c0 = c0;
    const std::uint32_t previous_c2 = c2;
    c0 = __umulhi(0xCD9E8D57U, previous_c2) ^ c1 ^ k0;
    c2 = __umulhi(0xD2511F53U, previous_c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57U * previous_c2;
    c3 = 0xD2511F53U * previous_c0;
    k0 += 0x9E3779B9U;
    k1 += 0xBB67AE85U;
  }
  return c0;
}

[[nodiscard]] __device__ float philox_uniform(
    const std::uint32_t seed,
    const std::uint64_t offset) {
  const std::int32_t signed_word =
      static_cast<std::int32_t>(philox_word(seed, offset));
  const std::int64_t magnitude =
      signed_word < 0 ? -static_cast<std::int64_t>(signed_word) - 1
                      : static_cast<std::int64_t>(signed_word);
  return static_cast<float>(magnitude) * 4.6566127342e-10F;
}

[[nodiscard]] __device__ float guided_logit(
    const __nv_bfloat16 conditional,
    const __nv_bfloat16 unconditional) {
  const __nv_bfloat16 conditional_scaled =
      __float2bfloat16_rn(__bfloat162float(conditional) * kCfgScale);
  const __nv_bfloat16 guided = __float2bfloat16_rn(
      __bfloat162float(conditional_scaled) +
      __bfloat162float(unconditional) * (1.0F - kCfgScale));
  return __bfloat162float(guided);
}

[[nodiscard]] __device__ float clamped_gumbel(float uniform) {
  uniform = fmaxf(0.00000006F, fminf(0.99999994F, uniform));
  return -__logf(-__logf(uniform));
}

#if defined(MAGPIE_TTS_RT_PLUGIN_TESTING)
__global__ void test_clamped_gumbel_kernel(
    const float* uniform,
    float* gumbel,
    const std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    gumbel[index] = clamped_gumbel(uniform[index]);
  }
}
#endif

__global__ void sampling_kernel(
    const __nv_bfloat16* logits,
    const bool* unfinished,
    const bool* finished,
    const bool* forbid_eos,
    const std::int64_t* rng_seed,
    const std::int64_t* rng_counter,
    const __nv_bfloat16* embedding_weight,
    std::int64_t* sampled_token,
    __nv_bfloat16* next_embedding,
    std::int64_t* updated_rng_counter,
    std::int32_t* invalid_rows) {
  using BlockSort =
      cub::BlockRadixSort<float, kSamplingThreads, kItemsPerThread>;
  __shared__ typename BlockSort::TempStorage sort_storage;
  __shared__ float reduction_scores[kSamplingThreads];
  __shared__ std::int32_t reduction_ids[kSamplingThreads];
  __shared__ float threshold;
  __shared__ std::int32_t chosen_token;
  __shared__ std::int32_t invalid;
  __shared__ std::int32_t finite_candidates;

  if (threadIdx.x == 0) {
    invalid = 0;
    finite_candidates = 0;
  }
  __syncthreads();

  const bool row_unfinished = unfinished[0];
  const bool row_finished = finished[0];
  const bool row_forbid_eos = forbid_eos[0];
  const std::int64_t seed_value = rng_seed[0];
  const std::int64_t counter_value = rng_counter[0];
  if (row_unfinished && row_finished) {
    atomicOr(&invalid, 1);
  }
  if (row_finished && row_forbid_eos) {
    atomicOr(&invalid, 1);
  }
  const bool seed_valid =
      seed_value >= 0 &&
      static_cast<std::uint64_t>(seed_value) <
          (static_cast<std::uint64_t>(1) << 32U);
  const bool counter_valid =
      counter_value >= 0 && counter_value <= kMaximumCounter;
  if (!seed_valid || !counter_valid) {
    atomicOr(&invalid, 1);
  }

  float values[kItemsPerThread];
#pragma unroll
  for (std::int32_t item = 0; item < kItemsPerThread; ++item) {
    const std::int32_t token =
        static_cast<std::int32_t>(threadIdx.x) * kItemsPerThread + item;
    float value = -CUDART_INF_F;
    if (token < kLocalArVocabularySize) {
      value = guided_logit(
          logits[token], logits[kLocalArVocabularySize + token]);
      if (!isfinite(value)) {
        atomicOr(&invalid, 1);
        value = -CUDART_INF_F;
      }
      const bool forbidden_special =
          token >= kLocalArCodebookSize && token != kLocalArAudioEosId;
      if (forbidden_special ||
          ((row_unfinished || row_forbid_eos) &&
           token == kLocalArAudioEosId)) {
        value = -CUDART_INF_F;
      }
      if (row_finished) {
        value = token == kLocalArAudioEosId ? 0.0F : -CUDART_INF_F;
      }
      if (isfinite(value)) {
        atomicAdd(&finite_candidates, 1);
      }
    }
    values[item] = value;
  }

  float sorted_values[kItemsPerThread];
#pragma unroll
  for (std::int32_t item = 0; item < kItemsPerThread; ++item) {
    sorted_values[item] = values[item];
  }
  BlockSort(sort_storage).SortDescending(sorted_values);
  if (threadIdx.x == (kTopK - 1) / kItemsPerThread) {
    threshold = sorted_values[(kTopK - 1) % kItemsPerThread];
  }
  __syncthreads();
  if (finite_candidates == 0) {
    atomicOr(&invalid, 1);
  }

  float thread_best = -CUDART_INF_F;
  std::int32_t thread_best_id = kLocalArVocabularySize;
  const std::uint32_t seed =
      seed_valid ? static_cast<std::uint32_t>(seed_value) : 0U;
  const std::uint64_t offset_base =
      counter_valid
          ? static_cast<std::uint64_t>(counter_value) *
                static_cast<std::uint64_t>(kPhiloxStride)
          : 0U;
#pragma unroll
  for (std::int32_t item = 0; item < kItemsPerThread; ++item) {
    const std::int32_t token =
        static_cast<std::int32_t>(threadIdx.x) * kItemsPerThread + item;
    if (token >= kLocalArVocabularySize || values[item] < threshold) {
      continue;
    }
    const float gumbel =
        clamped_gumbel(philox_uniform(seed, offset_base + token));
    const float score = values[item] / kTemperature + gumbel;
    if (score > thread_best ||
        (score == thread_best && token < thread_best_id)) {
      thread_best = score;
      thread_best_id = token;
    }
  }
  reduction_scores[threadIdx.x] = thread_best;
  reduction_ids[threadIdx.x] = thread_best_id;
  __syncthreads();

  for (std::int32_t stride = kSamplingThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < static_cast<unsigned>(stride)) {
      const float other_score = reduction_scores[threadIdx.x + stride];
      const std::int32_t other_id = reduction_ids[threadIdx.x + stride];
      const float own_score = reduction_scores[threadIdx.x];
      const std::int32_t own_id = reduction_ids[threadIdx.x];
      if (other_score > own_score ||
          (other_score == own_score && other_id < own_id)) {
        reduction_scores[threadIdx.x] = other_score;
        reduction_ids[threadIdx.x] = other_id;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    chosen_token = reduction_ids[0];
    if (chosen_token < 0 || chosen_token >= kLocalArVocabularySize ||
        reduction_scores[0] == -CUDART_INF_F) {
      invalid |= 1;
      chosen_token = 0;
    }
    sampled_token[0] = chosen_token;
    updated_rng_counter[0] =
        counter_valid ? counter_value + 1 : counter_value;
    invalid_rows[0] = invalid;
  }
  __syncthreads();

  for (std::int32_t width = static_cast<std::int32_t>(threadIdx.x);
       width < kLocalArEmbeddingWidth;
       width += kSamplingThreads) {
    const __nv_bfloat16 value =
        embedding_weight[chosen_token * kLocalArEmbeddingWidth + width];
    next_embedding[width] = value;
    next_embedding[kLocalArEmbeddingWidth + width] = value;
  }
}

__global__ void eos_projection_kernel(
    const __nv_bfloat16* decoder_hidden,
    const bool* unfinished,
    const bool* finished,
    const bool* forbid_eos,
    const __nv_bfloat16* final_weight,
    const __nv_bfloat16* final_bias,
    float* guided_logits) {
  if (unfinished[0] || finished[0] || forbid_eos[0]) {
    return;
  }
  const std::int32_t compact_index = static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t position = compact_index / kCompactEosVocabulary;
  const std::int32_t compact_token = compact_index % kCompactEosVocabulary;
  const std::int32_t token =
      compact_token == kLocalArCodebookSize ? kLocalArAudioEosId
                                            : compact_token;
  const std::int32_t output_index =
      position * kLocalArVocabularySize + token;

  float conditional_sum = 0.0F;
  float unconditional_sum = 0.0F;
  for (std::int32_t width = static_cast<std::int32_t>(threadIdx.x);
       width < kLocalArEmbeddingWidth;
       width += static_cast<std::int32_t>(blockDim.x)) {
    const float weight = __bfloat162float(
        final_weight[output_index * kLocalArEmbeddingWidth + width]);
    conditional_sum += __bfloat162float(decoder_hidden[width]) * weight;
    unconditional_sum +=
        __bfloat162float(decoder_hidden[kLocalArEmbeddingWidth + width]) *
        weight;
  }
  __shared__ float conditional_reduction[kSamplingThreads];
  __shared__ float unconditional_reduction[kSamplingThreads];
  conditional_reduction[threadIdx.x] = conditional_sum;
  unconditional_reduction[threadIdx.x] = unconditional_sum;
  __syncthreads();
  for (std::int32_t stride = kSamplingThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < static_cast<unsigned>(stride)) {
      conditional_reduction[threadIdx.x] +=
          conditional_reduction[threadIdx.x + stride];
      unconditional_reduction[threadIdx.x] +=
          unconditional_reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float bias = __bfloat162float(final_bias[output_index]);
    const __nv_bfloat16 conditional = __float2bfloat16_rn(
        conditional_reduction[0] + bias);
    const __nv_bfloat16 unconditional = __float2bfloat16_rn(
        unconditional_reduction[0] + bias);
    const __nv_bfloat16 unconditional_scaled =
        __float2bfloat16_rn(__bfloat162float(unconditional) *
                            (1.0F - kCfgScale));
    const __nv_bfloat16 guided = __float2bfloat16_rn(
        __bfloat162float(unconditional_scaled) +
        __bfloat162float(conditional) * kCfgScale);
    guided_logits[compact_index] = __bfloat162float(guided);
  }
}

__global__ void eos_reduce_kernel(
    const float* guided_logits,
    const std::int64_t* codec_tokens,
    const bool* unfinished,
    const bool* finished,
    const bool* forbid_eos,
    std::int32_t* frame_eos) {
  const std::int32_t position = static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t frame = position / kLocalArCodebooks;
  if (finished[0]) {
    if (threadIdx.x == 0) {
      atomicOr(frame_eos + frame, 1);
    }
    return;
  }
  if (unfinished[0] || forbid_eos[0]) {
    return;
  }
  float value = -CUDART_INF_F;
  for (std::int32_t token = static_cast<std::int32_t>(threadIdx.x);
       token < kLocalArCodebookSize;
       token += static_cast<std::int32_t>(blockDim.x)) {
    value = fmaxf(
        value,
        guided_logits[position * kCompactEosVocabulary + token]);
  }
  __shared__ float reduction[kSamplingThreads];
  reduction[threadIdx.x] = value;
  __syncthreads();
  for (std::int32_t stride = kSamplingThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < static_cast<unsigned>(stride)) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const std::int32_t codebook = position % kLocalArCodebooks;
    const float eos_logit = guided_logits[
        position * kCompactEosVocabulary + kLocalArCodebookSize];
    const bool argmax_eos = eos_logit > reduction[0];
    const bool sampled_eos =
        codec_tokens[codebook * kLocalArFrames + frame] ==
        kLocalArAudioEosId;
    const bool eos = argmax_eos || sampled_eos;
    if (eos) {
      atomicOr(frame_eos + frame, 1);
    }
  }
}

__global__ void eos_finalize_kernel(
    const std::int32_t* frame_eos,
    std::int32_t* end_frame_index) {
  if (frame_eos[0] != 0) {
    end_frame_index[0] = 0;
  } else if (frame_eos[1] != 0) {
    end_frame_index[0] = 1;
  } else {
    end_frame_index[0] = -1;
  }
}

[[nodiscard]] bool has_dims(
    const nvinfer1::Dims& dims,
    const std::initializer_list<std::int32_t> expected) noexcept {
  if (dims.nbDims != static_cast<std::int32_t>(expected.size())) {
    return false;
  }
  std::int32_t index = 0;
  for (const std::int32_t value : expected) {
    if (dims.d[index] != value) {
      return false;
    }
    ++index;
  }
  return true;
}

[[nodiscard]] bool is_supported_layer_norm_shape(
    const nvinfer1::Dims& dims) noexcept {
  if (dims.nbDims != 3 || dims.d[2] != kNormalizationWidth) {
    return false;
  }
  const bool local_shape =
      dims.d[0] == 2 && dims.d[1] == 1;
  const bool text_shape =
      dims.d[0] == 1 && dims.d[1] >= 1 &&
      dims.d[1] <= kMaximumTextSequenceLength;
  const bool main_decoder_shape =
      dims.d[0] == kMainDecoderBatch && dims.d[1] >= 1 &&
      dims.d[1] <= kMaximumTextSequenceLength;
  return local_shape || text_shape || main_decoder_shape;
}

[[nodiscard]] bool is_supported_gelu_shape(
    const nvinfer1::Dims& dims) noexcept {
  if (dims.nbDims != 3 || dims.d[1] != kLocalArFeedForwardWidth) {
    return false;
  }
  const bool local_shape =
      dims.d[0] == 2 && dims.d[2] == 1;
  const bool text_shape =
      dims.d[0] == 1 && dims.d[2] >= 1 &&
      dims.d[2] <= kMaximumTextSequenceLength;
  const bool main_decoder_shape =
      dims.d[0] == kMainDecoderBatch && dims.d[2] >= 1 &&
      dims.d[2] <= kMainDecoderPrefillLength;
  return local_shape || text_shape || main_decoder_shape;
}

[[nodiscard]] bool is_supported_softmax_shape(
    const nvinfer1::Dims& dims) noexcept {
  if (dims.nbDims != 4 || dims.d[3] < 1 ||
      dims.d[3] > kMaximumTextSequenceLength) {
    return false;
  }
  const bool local_shape =
      dims.d[0] == kMainDecoderBatch &&
      dims.d[1] == kAttentionHeads && dims.d[2] == 1 &&
      dims.d[3] <= kLocalArPositions;
  const bool text_shape =
      dims.d[0] == 1 && dims.d[1] == kAttentionHeads &&
      dims.d[2] == dims.d[3];
  const bool main_prefill_self_shape =
      dims.d[0] == kMainDecoderBatch &&
      dims.d[1] == kAttentionHeads &&
      dims.d[2] == kMainDecoderPrefillLength &&
      dims.d[3] == kMainDecoderPrefillLength;
  const bool main_prefill_cross_shape =
      dims.d[0] == kMainDecoderBatch &&
      dims.d[1] == kMainDecoderCrossAttentionHeads &&
      dims.d[2] == kMainDecoderPrefillLength;
  const bool main_step_self_shape =
      dims.d[0] == kMainDecoderBatch &&
      dims.d[1] == kAttentionHeads && dims.d[2] == 1 &&
      dims.d[3] >= kMainDecoderPrefillLength + 1 &&
      dims.d[3] <= kMainDecoderCacheCapacity;
  const bool main_step_cross_shape =
      dims.d[0] == kMainDecoderBatch &&
      dims.d[1] == kMainDecoderCrossAttentionHeads &&
      dims.d[2] == 1;
  return local_shape || text_shape || main_prefill_self_shape ||
         main_prefill_cross_shape || main_step_self_shape ||
         main_step_cross_shape;
}

[[nodiscard]] bool is_supported_main_cross_attention_shape(
    const nvinfer1::Dims& query,
    const nvinfer1::Dims& key,
    const nvinfer1::Dims& memory_mask,
    const nvinfer1::Dims& shape_reference) noexcept {
  if (query.nbDims != 4 || key.nbDims != 4 ||
      memory_mask.nbDims != 2 || shape_reference.nbDims != 4) {
    return false;
  }
  const std::int32_t query_length = query.d[2];
  const std::int32_t memory_length = key.d[1];
  const bool accepted_query_length =
      query_length == 1 ||
      query_length == kMainDecoderPrefillLength;
  return query.d[0] == kMainDecoderBatch &&
         query.d[1] == kMainDecoderCrossAttentionHeads &&
         accepted_query_length &&
         query.d[3] == kMainDecoderCrossAttentionWidth &&
         key.d[0] == kMainDecoderBatch &&
         memory_length >= 1 &&
         memory_length <= kMaximumTextSequenceLength &&
         key.d[2] == kMainDecoderCrossAttentionHeads &&
         key.d[3] == kMainDecoderCrossAttentionWidth &&
         memory_mask.d[0] == kMainDecoderBatch &&
         memory_mask.d[1] == memory_length &&
         shape_reference.d[0] == kMainDecoderBatch &&
         shape_reference.d[1] == kMainDecoderCrossAttentionHeads &&
         shape_reference.d[2] == query_length &&
         shape_reference.d[3] == memory_length;
}

[[nodiscard]] bool is_supported_main_self_attention_context_shape(
    const nvinfer1::Dims& probabilities,
    const nvinfer1::Dims& value,
    const nvinfer1::Dims& shape_reference) noexcept {
  if (probabilities.nbDims != 4 || value.nbDims != 4 ||
      shape_reference.nbDims != 4) {
    return false;
  }
  const std::int32_t query_length = probabilities.d[2];
  const std::int32_t key_length = probabilities.d[3];
  constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  return query_length == kMainDecoderPrefillLength &&
         key_length == kMainDecoderPrefillLength &&
         probabilities.d[0] == kMainDecoderBatch &&
         probabilities.d[1] == kAttentionHeads &&
         value.d[0] == kMainDecoderBatch &&
         value.d[1] == kAttentionHeads &&
         value.d[2] == key_length &&
         value.d[3] == head_width &&
         shape_reference.d[0] == kMainDecoderBatch &&
         shape_reference.d[1] == kAttentionHeads &&
         shape_reference.d[2] == query_length &&
         shape_reference.d[3] == head_width;
}

[[nodiscard]] bool is_supported_main_self_attention_step_context_shape(
    const nvinfer1::Dims& probabilities,
    const nvinfer1::Dims& value,
    const nvinfer1::Dims& shape_reference) noexcept {
  if (probabilities.nbDims != 4 || value.nbDims != 4 ||
      shape_reference.nbDims != 4) {
    return false;
  }
  const std::int32_t active_length = probabilities.d[3];
  constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  return active_length >= kMainDecoderPrefillLength + 1 &&
         active_length <= kMainDecoderCacheCapacity &&
         probabilities.d[0] == kMainDecoderBatch &&
         probabilities.d[1] == kAttentionHeads &&
         probabilities.d[2] == 1 &&
         value.d[0] == kMainDecoderBatch &&
         value.d[1] == kAttentionHeads &&
         value.d[2] == kMainDecoderCacheCapacity &&
         value.d[3] == head_width &&
         shape_reference.d[0] == kMainDecoderBatch &&
         shape_reference.d[1] == kAttentionHeads &&
         shape_reference.d[2] == 1 &&
         shape_reference.d[3] == head_width;
}

[[nodiscard]] bool is_supported_main_self_attention_step_scores_shape(
    const nvinfer1::Dims& query,
    const nvinfer1::Dims& key_transposed,
    const nvinfer1::Dims& shape_reference) noexcept {
  constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  if (shape_reference.nbDims != 4) {
    return false;
  }
  const std::int32_t active_length = shape_reference.d[3];
  return has_dims(
             query,
             {
                 kMainDecoderBatch,
                 kAttentionHeads,
                 1,
                 head_width,
             }) &&
         has_dims(
             key_transposed,
             {
                 kMainDecoderBatch,
                 kAttentionHeads,
                 head_width,
                 active_length,
             }) &&
         active_length >= kMainDecoderPrefillLength + 1 &&
         active_length <= kMainDecoderCacheCapacity &&
         shape_reference.d[0] == kMainDecoderBatch &&
         shape_reference.d[1] == kAttentionHeads &&
         shape_reference.d[2] == 1;
}

[[nodiscard]] bool is_supported_main_cross_attention_context_shape(
    const nvinfer1::Dims& probabilities,
    const nvinfer1::Dims& value,
    const nvinfer1::Dims& shape_reference) noexcept {
  if (probabilities.nbDims != 4 || value.nbDims != 4 ||
      shape_reference.nbDims != 4) {
    return false;
  }
  const std::int32_t memory_length = probabilities.d[3];
  const std::int32_t query_length = probabilities.d[2];
  const bool accepted_query_length =
      query_length == 1 ||
      query_length == kMainDecoderPrefillLength;
  return probabilities.d[0] == kMainDecoderBatch &&
         probabilities.d[1] ==
             kMainDecoderCrossAttentionHeads &&
         accepted_query_length &&
         memory_length >= 1 &&
         memory_length <= kMaximumTextSequenceLength &&
         value.d[0] == kMainDecoderBatch &&
         value.d[1] == kMainDecoderCrossAttentionHeads &&
         value.d[2] == memory_length &&
         value.d[3] == kMainDecoderCrossAttentionWidth &&
         shape_reference.d[0] == kMainDecoderBatch &&
         shape_reference.d[1] ==
             kMainDecoderCrossAttentionHeads &&
         shape_reference.d[2] == query_length &&
         shape_reference.d[3] ==
             kMainDecoderCrossAttentionWidth;
}

[[nodiscard]] bool is_supported_main_attention_prior_normalization_shape(
    const nvinfer1::Dims& probabilities,
    const nvinfer1::Dims& attention_prior) noexcept {
  if (probabilities.nbDims != 4 || attention_prior.nbDims != 3) {
    return false;
  }
  const std::int32_t memory_length = probabilities.d[3];
  return probabilities.d[0] == kMainDecoderBatch &&
         probabilities.d[1] == 1 && probabilities.d[2] == 1 &&
         memory_length >= 1 &&
         memory_length <= kMaximumTextSequenceLength &&
         has_dims(
             attention_prior,
             {kMainDecoderBatch, 1, memory_length});
}

[[nodiscard]] bool is_supported_main_alignment_mean_shape(
    const nvinfer1::Dims& alignment_0,
    const nvinfer1::Dims& alignment_1,
    const nvinfer1::Dims& alignment_2,
    const nvinfer1::Dims& alignment_3) noexcept {
  if (alignment_0.nbDims != 2) {
    return false;
  }
  const std::int32_t text_length = alignment_0.d[1];
  return alignment_0.d[0] == kMainDecoderBatch &&
         text_length >= 1 &&
         text_length <= kMaximumTextSequenceLength &&
         has_dims(
             alignment_1,
             {kMainDecoderBatch, text_length}) &&
         has_dims(
             alignment_2,
             {kMainDecoderBatch, text_length}) &&
         has_dims(
             alignment_3,
             {kMainDecoderBatch, text_length});
}

[[nodiscard]] std::int32_t softmax_input_count_for_mode(
    const std::int32_t mode) noexcept {
  if (mode == kMainCrossAttentionSoftmaxMode ||
      mode == kMainAlignmentMeanMode) {
    return 4;
  }
  if (mode == kMainSelfAttentionContextMode ||
      mode == kMainSelfAttentionStepContextMode ||
      mode == kMainCrossAttentionContextMode ||
      mode == kMainSelfAttentionStepScoresMode) {
    return 3;
  }
  if (mode == kMainAttentionPriorNormalizationMode) {
    return 2;
  }
  return mode == kSoftmaxMode ? 1 : 0;
}

[[nodiscard]] std::int32_t softmax_shape_reference_index(
    const std::int32_t mode) noexcept {
  if (mode == kMainCrossAttentionSoftmaxMode) {
    return 3;
  }
  if (mode == kMainAttentionPriorNormalizationMode) {
    return 0;
  }
  return mode == kMainSelfAttentionContextMode ||
                 mode == kMainSelfAttentionStepContextMode ||
                 mode == kMainCrossAttentionContextMode ||
                 mode == kMainSelfAttentionStepScoresMode
             ? 2
             : 0;
}

class SamplingPlugin final : public nvinfer1::IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime {
 public:
  SamplingPlugin() { initialize_fields(); }
  SamplingPlugin(const SamplingPlugin&) { initialize_fields(); }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      const nvinfer1::PluginCapabilityType type) noexcept override {
    if (type == nvinfer1::PluginCapabilityType::kCORE) {
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) SamplingPlugin(*this);
  }

  const char* getPluginName() const noexcept override {
    return kSamplingPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  std::int32_t getNbOutputs() const noexcept override { return 4; }

  std::int32_t configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(outputs);
    return input_count == 7 && output_count == 4 ? 0 : -1;
  }

  bool supportsFormatCombination(
      const std::int32_t position,
      const nvinfer1::DynamicPluginTensorDesc* tensors,
      const std::int32_t input_count,
      const std::int32_t output_count) noexcept override {
    if (input_count != 7 || output_count != 4 || position < 0 ||
        position >= input_count + output_count) {
      return false;
    }
    const std::array<nvinfer1::DataType, 11> types{
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kBOOL,
        nvinfer1::DataType::kBOOL,
        nvinfer1::DataType::kBOOL,
        nvinfer1::DataType::kINT64,
        nvinfer1::DataType::kINT64,
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kINT64,
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kINT64,
        nvinfer1::DataType::kINT32};
    return tensors[position].desc.format == nvinfer1::PluginFormat::kLINEAR &&
           tensors[position].desc.type ==
               types[static_cast<std::size_t>(position)];
  }

  std::int32_t getOutputDataTypes(
      nvinfer1::DataType* output_types,
      const std::int32_t output_count,
      const nvinfer1::DataType* input_types,
      const std::int32_t input_count) const noexcept override {
    static_cast<void>(input_types);
    if (input_count != 7 || output_count != 4) {
      return -1;
    }
    output_types[0] = nvinfer1::DataType::kINT64;
    output_types[1] = nvinfer1::DataType::kBF16;
    output_types[2] = nvinfer1::DataType::kINT64;
    output_types[3] = nvinfer1::DataType::kINT32;
    return 0;
  }

  std::int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs,
      const std::int32_t input_count,
      const nvinfer1::DimsExprs* shape_inputs,
      const std::int32_t shape_input_count,
      nvinfer1::DimsExprs* outputs,
      const std::int32_t output_count,
      nvinfer1::IExprBuilder& expression_builder) noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(shape_inputs);
    if (input_count != 7 || shape_input_count != 0 || output_count != 4) {
      return -1;
    }
    outputs[0].nbDims = 1;
    outputs[0].d[0] = expression_builder.constant(1);
    outputs[1].nbDims = 2;
    outputs[1].d[0] = expression_builder.constant(2);
    outputs[1].d[1] = expression_builder.constant(kLocalArEmbeddingWidth);
    outputs[2].nbDims = 1;
    outputs[2].d[0] = expression_builder.constant(1);
    outputs[3].nbDims = 1;
    outputs[3].d[0] = expression_builder.constant(1);
    return 0;
  }

  std::size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) const noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(input_count);
    static_cast<void>(outputs);
    static_cast<void>(output_count);
    return 0;
  }

  std::int32_t onShapeChange(
      const nvinfer1::PluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::PluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(outputs);
    if (input_count != 7 || output_count != 4) {
      return -1;
    }
    return has_dims(inputs[0].dims, {2, kLocalArVocabularySize}) &&
                   has_dims(inputs[1].dims, {1}) &&
                   has_dims(inputs[2].dims, {1}) &&
                   has_dims(inputs[3].dims, {1}) &&
                   has_dims(inputs[4].dims, {1}) &&
                   has_dims(inputs[5].dims, {1}) &&
                   has_dims(
                       inputs[6].dims,
                       {kLocalArVocabularySize, kLocalArEmbeddingWidth})
               ? 0
               : -1;
  }

  std::int32_t enqueue(
      const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override {
    static_cast<void>(input_desc);
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    return launch_local_ar_sampling(
        inputs[0],
        static_cast<const bool*>(inputs[1]),
        static_cast<const bool*>(inputs[2]),
        static_cast<const bool*>(inputs[3]),
        static_cast<const std::int64_t*>(inputs[4]),
        static_cast<const std::int64_t*>(inputs[5]),
        inputs[6],
        static_cast<std::int64_t*>(outputs[0]),
        outputs[1],
        static_cast<std::int64_t*>(outputs[2]),
        static_cast<std::int32_t*>(outputs[3]),
        stream);
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    return clone();
  }

  nvinfer1::PluginFieldCollection const* getFieldsToSerialize()
      noexcept override {
    return &fields_;
  }

 private:
  void initialize_fields() noexcept {
    fields_.nbFields = 0;
    fields_.fields = nullptr;
  }

  nvinfer1::PluginFieldCollection fields_{};
};

class EosPlugin final : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuild,
                        public nvinfer1::IPluginV3OneRuntime {
 public:
  EosPlugin() { initialize_fields(); }
  EosPlugin(const EosPlugin&) { initialize_fields(); }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      const nvinfer1::PluginCapabilityType type) noexcept override {
    if (type == nvinfer1::PluginCapabilityType::kCORE) {
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) EosPlugin(*this);
  }

  const char* getPluginName() const noexcept override {
    return kEosPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  std::int32_t getNbOutputs() const noexcept override { return 1; }

  std::int32_t configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(outputs);
    return input_count == 7 && output_count == 1 ? 0 : -1;
  }

  bool supportsFormatCombination(
      const std::int32_t position,
      const nvinfer1::DynamicPluginTensorDesc* tensors,
      const std::int32_t input_count,
      const std::int32_t output_count) noexcept override {
    if (input_count != 7 || output_count != 1 || position < 0 ||
        position >= input_count + output_count) {
      return false;
    }
    const std::array<nvinfer1::DataType, 8> types{
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kINT64,
        nvinfer1::DataType::kBOOL,
        nvinfer1::DataType::kBOOL,
        nvinfer1::DataType::kBOOL,
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kINT32};
    return tensors[position].desc.format == nvinfer1::PluginFormat::kLINEAR &&
           tensors[position].desc.type ==
               types[static_cast<std::size_t>(position)];
  }

  std::int32_t getOutputDataTypes(
      nvinfer1::DataType* output_types,
      const std::int32_t output_count,
      const nvinfer1::DataType* input_types,
      const std::int32_t input_count) const noexcept override {
    static_cast<void>(input_types);
    if (input_count != 7 || output_count != 1) {
      return -1;
    }
    output_types[0] = nvinfer1::DataType::kINT32;
    return 0;
  }

  std::int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs,
      const std::int32_t input_count,
      const nvinfer1::DimsExprs* shape_inputs,
      const std::int32_t shape_input_count,
      nvinfer1::DimsExprs* outputs,
      const std::int32_t output_count,
      nvinfer1::IExprBuilder& expression_builder) noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(shape_inputs);
    if (input_count != 7 || shape_input_count != 0 || output_count != 1) {
      return -1;
    }
    outputs[0].nbDims = 1;
    outputs[0].d[0] = expression_builder.constant(1);
    return 0;
  }

  std::size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) const noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(input_count);
    static_cast<void>(outputs);
    static_cast<void>(output_count);
    return local_ar_eos_workspace_size();
  }

  std::int32_t onShapeChange(
      const nvinfer1::PluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::PluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(outputs);
    if (input_count != 7 || output_count != 1) {
      return -1;
    }
    return has_dims(inputs[0].dims, {2, kLocalArEmbeddingWidth}) &&
                   has_dims(
                       inputs[1].dims,
                       {1, kLocalArCodebooks, kLocalArFrames}) &&
                   has_dims(inputs[2].dims, {1}) &&
                   has_dims(inputs[3].dims, {1}) &&
                   has_dims(inputs[4].dims, {1}) &&
                   has_dims(
                       inputs[5].dims,
                       {kLocalArPositions * kLocalArVocabularySize,
                        kLocalArEmbeddingWidth}) &&
                   has_dims(
                       inputs[6].dims,
                       {kLocalArPositions * kLocalArVocabularySize})
               ? 0
               : -1;
  }

  std::int32_t enqueue(
      const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override {
    static_cast<void>(input_desc);
    static_cast<void>(output_desc);
    return launch_local_ar_eos(
        inputs[0],
        static_cast<const std::int64_t*>(inputs[1]),
        static_cast<const bool*>(inputs[2]),
        static_cast<const bool*>(inputs[3]),
        static_cast<const bool*>(inputs[4]),
        inputs[5],
        inputs[6],
        workspace,
        static_cast<std::int32_t*>(outputs[0]),
        stream);
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    return clone();
  }

  nvinfer1::PluginFieldCollection const* getFieldsToSerialize()
      noexcept override {
    return &fields_;
  }

 private:
  void initialize_fields() noexcept {
    fields_.nbFields = 0;
    fields_.fields = nullptr;
  }

  nvinfer1::PluginFieldCollection fields_{};
};

class LayerNormPlugin final : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime {
 public:
  LayerNormPlugin() { initialize_fields(); }
  LayerNormPlugin(const LayerNormPlugin&) { initialize_fields(); }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      const nvinfer1::PluginCapabilityType type) noexcept override {
    if (type == nvinfer1::PluginCapabilityType::kCORE) {
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) LayerNormPlugin(*this);
  }

  const char* getPluginName() const noexcept override {
    return kLayerNormPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  std::int32_t getNbOutputs() const noexcept override { return 1; }

  std::int32_t configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(outputs);
    return input_count == 2 && output_count == 1 ? 0 : -1;
  }

  bool supportsFormatCombination(
      const std::int32_t position,
      const nvinfer1::DynamicPluginTensorDesc* tensors,
      const std::int32_t input_count,
      const std::int32_t output_count) noexcept override {
    if (input_count != 2 || output_count != 1 || position < 0 ||
        position >= input_count + output_count) {
      return false;
    }
    return tensors[position].desc.format ==
               nvinfer1::PluginFormat::kLINEAR &&
           tensors[position].desc.type == nvinfer1::DataType::kBF16;
  }

  std::int32_t getOutputDataTypes(
      nvinfer1::DataType* output_types,
      const std::int32_t output_count,
      const nvinfer1::DataType* input_types,
      const std::int32_t input_count) const noexcept override {
    if (input_count != 2 || output_count != 1 ||
        input_types[0] != nvinfer1::DataType::kBF16 ||
        input_types[1] != nvinfer1::DataType::kBF16) {
      return -1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    return 0;
  }

  std::int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs,
      const std::int32_t input_count,
      const nvinfer1::DimsExprs* shape_inputs,
      const std::int32_t shape_input_count,
      nvinfer1::DimsExprs* outputs,
      const std::int32_t output_count,
      nvinfer1::IExprBuilder& expression_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expression_builder);
    if (input_count != 2 || shape_input_count != 0 || output_count != 1) {
      return -1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  std::size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) const noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(input_count);
    static_cast<void>(outputs);
    static_cast<void>(output_count);
    return 0;
  }

  std::int32_t onShapeChange(
      const nvinfer1::PluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::PluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(outputs);
    if (input_count != 2 || output_count != 1) {
      return -1;
    }
    return is_supported_layer_norm_shape(inputs[0].dims) &&
                   has_dims(inputs[1].dims, {kNormalizationWidth})
               ? 0
               : -1;
  }

  std::int32_t enqueue(
      const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    const nvinfer1::Dims& dims = input_desc[0].dims;
    if (!is_supported_layer_norm_shape(dims)) {
      return -1;
    }
    return launch_layer_norm(
        inputs[0],
        inputs[1],
        outputs[0],
        dims.d[0] * dims.d[1],
        stream);
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    return clone();
  }

  nvinfer1::PluginFieldCollection const* getFieldsToSerialize()
      noexcept override {
    return &fields_;
  }

 private:
  void initialize_fields() noexcept {
    fields_.nbFields = 0;
    fields_.fields = nullptr;
  }

  nvinfer1::PluginFieldCollection fields_{};
};

class GeluTanhPlugin final : public nvinfer1::IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime {
 public:
  GeluTanhPlugin() { initialize_fields(); }
  GeluTanhPlugin(const GeluTanhPlugin&) { initialize_fields(); }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      const nvinfer1::PluginCapabilityType type) noexcept override {
    if (type == nvinfer1::PluginCapabilityType::kCORE) {
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) GeluTanhPlugin(*this);
  }

  const char* getPluginName() const noexcept override {
    return kGeluTanhPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  std::int32_t getNbOutputs() const noexcept override { return 1; }

  std::int32_t configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(outputs);
    return input_count == 1 && output_count == 1 ? 0 : -1;
  }

  bool supportsFormatCombination(
      const std::int32_t position,
      const nvinfer1::DynamicPluginTensorDesc* tensors,
      const std::int32_t input_count,
      const std::int32_t output_count) noexcept override {
    if (input_count != 1 || output_count != 1 || position < 0 ||
        position >= input_count + output_count) {
      return false;
    }
    return tensors[position].desc.format ==
               nvinfer1::PluginFormat::kLINEAR &&
           tensors[position].desc.type == nvinfer1::DataType::kBF16;
  }

  std::int32_t getOutputDataTypes(
      nvinfer1::DataType* output_types,
      const std::int32_t output_count,
      const nvinfer1::DataType* input_types,
      const std::int32_t input_count) const noexcept override {
    if (input_count != 1 || output_count != 1 ||
        input_types[0] != nvinfer1::DataType::kBF16) {
      return -1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    return 0;
  }

  std::int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs,
      const std::int32_t input_count,
      const nvinfer1::DimsExprs* shape_inputs,
      const std::int32_t shape_input_count,
      nvinfer1::DimsExprs* outputs,
      const std::int32_t output_count,
      nvinfer1::IExprBuilder& expression_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expression_builder);
    if (input_count != 1 || shape_input_count != 0 || output_count != 1) {
      return -1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  std::size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) const noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(input_count);
    static_cast<void>(outputs);
    static_cast<void>(output_count);
    return 0;
  }

  std::int32_t onShapeChange(
      const nvinfer1::PluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::PluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(outputs);
    if (input_count != 1 || output_count != 1) {
      return -1;
    }
    return is_supported_gelu_shape(inputs[0].dims)
               ? 0
               : -1;
  }

  std::int32_t enqueue(
      const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    const nvinfer1::Dims& dims = input_desc[0].dims;
    if (!is_supported_gelu_shape(dims)) {
      return -1;
    }
    const std::size_t element_count =
        static_cast<std::size_t>(dims.d[0]) *
        static_cast<std::size_t>(dims.d[1]) *
        static_cast<std::size_t>(dims.d[2]);
    return launch_gelu_tanh(
        inputs[0], outputs[0], element_count, stream);
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    return clone();
  }

  nvinfer1::PluginFieldCollection const* getFieldsToSerialize()
      noexcept override {
    return &fields_;
  }

 private:
  void initialize_fields() noexcept {
    fields_.nbFields = 0;
    fields_.fields = nullptr;
  }

  nvinfer1::PluginFieldCollection fields_{};
};

class SoftmaxPlugin final : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit SoftmaxPlugin(const std::int32_t mode = kSoftmaxMode)
      : mode_(mode) {
    initialize_fields();
    initialize_cublas();
  }
  SoftmaxPlugin(const SoftmaxPlugin& other) : mode_(other.mode_) {
    initialize_fields();
    initialize_cublas();
  }
  ~SoftmaxPlugin() override {
    if (cublas_handle_ != nullptr) {
      static_cast<void>(cublasDestroy(cublas_handle_));
    }
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      const nvinfer1::PluginCapabilityType type) noexcept override {
    if (type == nvinfer1::PluginCapabilityType::kCORE) {
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) SoftmaxPlugin(*this);
  }

  const char* getPluginName() const noexcept override {
    return kSoftmaxPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  std::int32_t getNbOutputs() const noexcept override { return 1; }

  std::int32_t configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(outputs);
    const std::int32_t expected_inputs =
        softmax_input_count_for_mode(mode_);
    return input_count == expected_inputs && output_count == 1 ? 0 : -1;
  }

  bool supportsFormatCombination(
      const std::int32_t position,
      const nvinfer1::DynamicPluginTensorDesc* tensors,
      const std::int32_t input_count,
      const std::int32_t output_count) noexcept override {
    const std::int32_t expected_inputs =
        softmax_input_count_for_mode(mode_);
    if (input_count != expected_inputs || output_count != 1 ||
        position < 0 ||
        position >= input_count + output_count) {
      return false;
    }
    if (tensors[position].desc.format !=
        nvinfer1::PluginFormat::kLINEAR) {
      return false;
    }
    if (mode_ == kSoftmaxMode) {
      return tensors[position].desc.type ==
             nvinfer1::DataType::kBF16;
    }
    if (mode_ == kMainSelfAttentionContextMode ||
        mode_ == kMainSelfAttentionStepContextMode ||
        mode_ == kMainCrossAttentionContextMode ||
        mode_ == kMainSelfAttentionStepScoresMode ||
        mode_ == kMainAttentionPriorNormalizationMode ||
        mode_ == kMainAlignmentMeanMode) {
      return tensors[position].desc.type ==
             nvinfer1::DataType::kBF16;
    }
    const std::array<nvinfer1::DataType, 5> types{
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kBOOL,
        nvinfer1::DataType::kBF16,
        nvinfer1::DataType::kBF16};
    return tensors[position].desc.type ==
           types[static_cast<std::size_t>(position)];
  }

  std::int32_t getOutputDataTypes(
      nvinfer1::DataType* output_types,
      const std::int32_t output_count,
      const nvinfer1::DataType* input_types,
      const std::int32_t input_count) const noexcept override {
    const std::int32_t expected_inputs =
        softmax_input_count_for_mode(mode_);
    if (input_count != expected_inputs || output_count != 1 ||
        input_types[0] != nvinfer1::DataType::kBF16) {
      return -1;
    }
    if (mode_ == kMainCrossAttentionSoftmaxMode &&
        (input_types[1] != nvinfer1::DataType::kBF16 ||
         input_types[2] != nvinfer1::DataType::kBOOL ||
         input_types[3] != nvinfer1::DataType::kBF16)) {
      return -1;
    }
    if ((mode_ == kMainSelfAttentionContextMode ||
         mode_ == kMainSelfAttentionStepContextMode ||
         mode_ == kMainCrossAttentionContextMode ||
         mode_ == kMainSelfAttentionStepScoresMode) &&
        (input_types[1] != nvinfer1::DataType::kBF16 ||
         input_types[2] != nvinfer1::DataType::kBF16)) {
      return -1;
    }
    if (mode_ == kMainAttentionPriorNormalizationMode &&
        input_types[1] != nvinfer1::DataType::kBF16) {
      return -1;
    }
    if (mode_ == kMainAlignmentMeanMode &&
        (input_types[1] != nvinfer1::DataType::kBF16 ||
         input_types[2] != nvinfer1::DataType::kBF16 ||
         input_types[3] != nvinfer1::DataType::kBF16)) {
      return -1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    return 0;
  }

  std::int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs,
      const std::int32_t input_count,
      const nvinfer1::DimsExprs* shape_inputs,
      const std::int32_t shape_input_count,
      nvinfer1::DimsExprs* outputs,
      const std::int32_t output_count,
      nvinfer1::IExprBuilder& expression_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expression_builder);
    const std::int32_t expected_inputs =
        softmax_input_count_for_mode(mode_);
    if (input_count != expected_inputs || shape_input_count != 0 ||
        output_count != 1) {
      return -1;
    }
    outputs[0] = inputs[softmax_shape_reference_index(mode_)];
    return 0;
  }

  std::size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      const std::int32_t output_count) const noexcept override {
    static_cast<void>(outputs);
    static_cast<void>(output_count);
    static_cast<void>(inputs);
    if (mode_ == kMainCrossAttentionSoftmaxMode && input_count == 4) {
      return main_cross_attention_softmax_workspace_size();
    }
    if (mode_ == kMainSelfAttentionStepContextMode &&
        input_count == 3) {
      return main_self_attention_step_context_workspace_size();
    }
    return 0;
  }

  std::int32_t onShapeChange(
      const nvinfer1::PluginTensorDesc* inputs,
      const std::int32_t input_count,
      const nvinfer1::PluginTensorDesc* outputs,
      const std::int32_t output_count) noexcept override {
    static_cast<void>(outputs);
    const std::int32_t expected_inputs =
        softmax_input_count_for_mode(mode_);
    if (input_count != expected_inputs || output_count != 1) {
      return -1;
    }
    if (mode_ == kSoftmaxMode) {
      return is_supported_softmax_shape(inputs[0].dims) ? 0 : -1;
    }
    if (mode_ == kMainSelfAttentionContextMode) {
      return is_supported_main_self_attention_context_shape(
                 inputs[0].dims,
                 inputs[1].dims,
                 inputs[2].dims)
                 ? 0
                 : -1;
    }
    if (mode_ == kMainSelfAttentionStepContextMode) {
      return is_supported_main_self_attention_step_context_shape(
                 inputs[0].dims,
                 inputs[1].dims,
                 inputs[2].dims)
                 ? 0
                 : -1;
    }
    if (mode_ == kMainSelfAttentionStepScoresMode) {
      return is_supported_main_self_attention_step_scores_shape(
                 inputs[0].dims,
                 inputs[1].dims,
                 inputs[2].dims)
                 ? 0
                 : -1;
    }
    if (mode_ == kMainCrossAttentionContextMode) {
      return is_supported_main_cross_attention_context_shape(
                 inputs[0].dims,
                 inputs[1].dims,
                 inputs[2].dims)
                 ? 0
                 : -1;
    }
    if (mode_ == kMainAttentionPriorNormalizationMode) {
      return is_supported_main_attention_prior_normalization_shape(
                 inputs[0].dims,
                 inputs[1].dims)
                 ? 0
                 : -1;
    }
    if (mode_ == kMainAlignmentMeanMode) {
      return is_supported_main_alignment_mean_shape(
                 inputs[0].dims,
                 inputs[1].dims,
                 inputs[2].dims,
                 inputs[3].dims)
                 ? 0
                 : -1;
    }
    return is_supported_main_cross_attention_shape(
               inputs[0].dims,
               inputs[1].dims,
               inputs[2].dims,
               inputs[3].dims)
               ? 0
               : -1;
  }

  std::int32_t enqueue(
      const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    if (input_desc == nullptr || inputs == nullptr ||
        outputs == nullptr) {
      return -1;
    }
    if (mode_ == kMainCrossAttentionSoftmaxMode) {
      if (!is_supported_main_cross_attention_shape(
            input_desc[0].dims,
            input_desc[1].dims,
            input_desc[2].dims,
            input_desc[3].dims)) {
        return -1;
      }
      return launch_main_cross_attention_softmax(
          cublas_handle_,
          inputs[0],
          inputs[1],
          static_cast<const bool*>(inputs[2]),
          workspace,
          outputs[0],
          input_desc[0].dims.d[2],
          input_desc[1].dims.d[1],
          stream);
    }
    if (mode_ == kMainSelfAttentionContextMode) {
      if (!is_supported_main_self_attention_context_shape(
              input_desc[0].dims,
              input_desc[1].dims,
              input_desc[2].dims)) {
        return -1;
      }
      return launch_main_self_attention_context(
          inputs[0],
          inputs[1],
          outputs[0],
          input_desc[0].dims.d[2],
          input_desc[0].dims.d[3],
          stream);
    }
    if (mode_ == kMainSelfAttentionStepContextMode) {
      if (!is_supported_main_self_attention_step_context_shape(
              input_desc[0].dims,
              input_desc[1].dims,
              input_desc[2].dims)) {
        return -1;
      }
      return launch_main_self_attention_step_context(
          cublas_handle_,
          inputs[0],
          inputs[1],
          workspace,
          outputs[0],
          input_desc[0].dims.d[3],
          stream);
    }
    if (mode_ == kMainSelfAttentionStepScoresMode) {
      if (!is_supported_main_self_attention_step_scores_shape(
              input_desc[0].dims,
              input_desc[1].dims,
              input_desc[2].dims)) {
        return -1;
      }
      return launch_main_self_attention_step_scores(
          cublas_handle_,
          inputs[0],
          inputs[1],
          outputs[0],
          input_desc[2].dims.d[3],
          stream);
    }
    if (mode_ == kMainCrossAttentionContextMode) {
      if (!is_supported_main_cross_attention_context_shape(
              input_desc[0].dims,
              input_desc[1].dims,
              input_desc[2].dims)) {
        return -1;
      }
      return launch_main_cross_attention_context(
          cublas_handle_,
          inputs[0],
          inputs[1],
          outputs[0],
          input_desc[0].dims.d[2],
          input_desc[0].dims.d[3],
          stream);
    }
    if (mode_ == kMainAttentionPriorNormalizationMode) {
      if (!is_supported_main_attention_prior_normalization_shape(
              input_desc[0].dims,
              input_desc[1].dims)) {
        return -1;
      }
      return launch_main_attention_prior_normalization(
          inputs[0],
          inputs[1],
          outputs[0],
          input_desc[0].dims.d[3],
          stream);
    }
    if (mode_ == kMainAlignmentMeanMode) {
      if (!is_supported_main_alignment_mean_shape(
              input_desc[0].dims,
              input_desc[1].dims,
              input_desc[2].dims,
              input_desc[3].dims)) {
        return -1;
      }
      return launch_main_alignment_mean(
          inputs[0],
          inputs[1],
          inputs[2],
          inputs[3],
          outputs[0],
          input_desc[0].dims.d[1],
          stream);
    }
    const nvinfer1::Dims& dims = input_desc[0].dims;
    if (!is_supported_softmax_shape(dims)) {
      return -1;
    }
    return launch_softmax(
        inputs[0],
        outputs[0],
        dims.d[0] * dims.d[1] * dims.d[2],
        dims.d[3],
        stream);
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    return clone();
  }

  nvinfer1::PluginFieldCollection const* getFieldsToSerialize()
      noexcept override {
    return &fields_;
  }

 private:
  void initialize_cublas() noexcept {
    if (mode_ != kMainCrossAttentionSoftmaxMode &&
        mode_ != kMainCrossAttentionContextMode &&
        mode_ != kMainSelfAttentionStepContextMode &&
        mode_ != kMainSelfAttentionStepScoresMode) {
      return;
    }
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
      cublas_handle_ = nullptr;
      return;
    }
    if (cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH) !=
        CUBLAS_STATUS_SUCCESS) {
      static_cast<void>(cublasDestroy(cublas_handle_));
      cublas_handle_ = nullptr;
    }
  }

  void initialize_fields() noexcept {
    serialized_fields_[0] = nvinfer1::PluginField{
        "mode",
        &mode_,
        nvinfer1::PluginFieldType::kINT32,
        1};
    fields_.nbFields = 1;
    fields_.fields = serialized_fields_.data();
  }

  std::int32_t mode_{kSoftmaxMode};
  cublasHandle_t cublas_handle_{nullptr};
  std::array<nvinfer1::PluginField, 1> serialized_fields_{};
  nvinfer1::PluginFieldCollection fields_{};
};

template <typename Plugin, const char* Name>
class Creator final : public nvinfer1::IPluginCreatorV3One {
 public:
  Creator() {
    fields_.nbFields = 0;
    fields_.fields = nullptr;
  }

  const char* getPluginName() const noexcept override { return Name; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &fields_;
  }

  nvinfer1::IPluginV3* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fields,
      const nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    if (fields == nullptr || fields->nbFields != 0) {
      return nullptr;
    }
    return new (std::nothrow) Plugin();
  }

 private:
  nvinfer1::PluginFieldCollection fields_{};
};

class SoftmaxCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  SoftmaxCreator() {
    advertised_fields_[0] = nvinfer1::PluginField{
        "mode",
        nullptr,
        nvinfer1::PluginFieldType::kINT32,
        1};
    fields_.nbFields = 1;
    fields_.fields = advertised_fields_.data();
  }

  const char* getPluginName() const noexcept override {
    return kSoftmaxPluginName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &fields_;
  }

  nvinfer1::IPluginV3* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fields,
      const nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    if (fields == nullptr || fields->nbFields < 0 ||
        fields->nbFields > 1) {
      return nullptr;
    }
    std::int32_t mode = kSoftmaxMode;
    if (fields->nbFields == 1) {
      const nvinfer1::PluginField& field = fields->fields[0];
      if (field.name == nullptr ||
          std::string_view(field.name) != "mode" ||
          field.type != nvinfer1::PluginFieldType::kINT32 ||
          field.length != 1 || field.data == nullptr) {
        return nullptr;
      }
      mode = *static_cast<const std::int32_t*>(field.data);
    }
    if (mode != kSoftmaxMode &&
        mode != kMainCrossAttentionSoftmaxMode &&
        mode != kMainSelfAttentionContextMode &&
        mode != kMainCrossAttentionContextMode &&
        mode != kMainAttentionPriorNormalizationMode &&
        mode != kMainSelfAttentionStepScoresMode &&
        mode != kMainSelfAttentionStepContextMode &&
        mode != kMainAlignmentMeanMode) {
      return nullptr;
    }
    return new (std::nothrow) SoftmaxPlugin(mode);
  }

 private:
  std::array<nvinfer1::PluginField, 1> advertised_fields_{};
  nvinfer1::PluginFieldCollection fields_{};
};

using SamplingCreator = Creator<SamplingPlugin, kSamplingPluginName>;
using EosCreator = Creator<EosPlugin, kEosPluginName>;
using LayerNormCreator = Creator<LayerNormPlugin, kLayerNormPluginName>;
using GeluTanhCreator = Creator<GeluTanhPlugin, kGeluTanhPluginName>;

SamplingCreator sampling_creator;
EosCreator eos_creator;
LayerNormCreator layer_norm_creator;
GeluTanhCreator gelu_tanh_creator;
SoftmaxCreator softmax_creator;
std::mutex registration_mutex;

}  // namespace

mtt_plugin_status_t register_plugins_explicitly() noexcept {
  const std::lock_guard<std::mutex> lock(registration_mutex);
  nvinfer1::IPluginRegistry* registry = ::getPluginRegistry();
  if (registry == nullptr) {
    return MTT_PLUGIN_STATUS_REGISTRY_UNAVAILABLE;
  }

  nvinfer1::IPluginCreatorInterface* registered_sampling =
      registry->getCreator(
          kSamplingPluginName, kPluginVersion, kPluginNamespace);
  nvinfer1::IPluginCreatorInterface* registered_eos =
      registry->getCreator(kEosPluginName, kPluginVersion, kPluginNamespace);
  nvinfer1::IPluginCreatorInterface* registered_layer_norm =
      registry->getCreator(
          kLayerNormPluginName, kPluginVersion, kPluginNamespace);
  nvinfer1::IPluginCreatorInterface* registered_gelu_tanh =
      registry->getCreator(
          kGeluTanhPluginName, kPluginVersion, kPluginNamespace);
  nvinfer1::IPluginCreatorInterface* registered_softmax =
      registry->getCreator(
          kSoftmaxPluginName, kPluginVersion, kPluginNamespace);
  auto* const expected_sampling =
      static_cast<nvinfer1::IPluginCreatorInterface*>(&sampling_creator);
  auto* const expected_eos =
      static_cast<nvinfer1::IPluginCreatorInterface*>(&eos_creator);
  auto* const expected_layer_norm =
      static_cast<nvinfer1::IPluginCreatorInterface*>(&layer_norm_creator);
  auto* const expected_gelu_tanh =
      static_cast<nvinfer1::IPluginCreatorInterface*>(&gelu_tanh_creator);
  auto* const expected_softmax =
      static_cast<nvinfer1::IPluginCreatorInterface*>(&softmax_creator);
  if (registered_sampling == expected_sampling &&
      registered_eos == expected_eos &&
      registered_layer_norm == expected_layer_norm &&
      registered_gelu_tanh == expected_gelu_tanh &&
      registered_softmax == expected_softmax) {
    return MTT_PLUGIN_STATUS_ALREADY_REGISTERED;
  }
  if (registered_sampling != nullptr || registered_eos != nullptr ||
      registered_layer_norm != nullptr || registered_gelu_tanh != nullptr ||
      registered_softmax != nullptr) {
    return MTT_PLUGIN_STATUS_REGISTRATION_CONFLICT;
  }

  if (!registry->registerCreator(sampling_creator, kPluginNamespace)) {
    registered_sampling = registry->getCreator(
        kSamplingPluginName, kPluginVersion, kPluginNamespace);
    return registered_sampling != nullptr
               ? MTT_PLUGIN_STATUS_REGISTRATION_CONFLICT
               : MTT_PLUGIN_STATUS_REGISTRATION_FAILED;
  }
  if (!registry->registerCreator(eos_creator, kPluginNamespace)) {
    static_cast<void>(registry->deregisterCreator(sampling_creator));
    registered_eos =
        registry->getCreator(kEosPluginName, kPluginVersion, kPluginNamespace);
    return registered_eos != nullptr
               ? MTT_PLUGIN_STATUS_REGISTRATION_CONFLICT
               : MTT_PLUGIN_STATUS_REGISTRATION_FAILED;
  }
  if (!registry->registerCreator(layer_norm_creator, kPluginNamespace)) {
    static_cast<void>(registry->deregisterCreator(eos_creator));
    static_cast<void>(registry->deregisterCreator(sampling_creator));
    registered_layer_norm = registry->getCreator(
        kLayerNormPluginName, kPluginVersion, kPluginNamespace);
    return registered_layer_norm != nullptr
               ? MTT_PLUGIN_STATUS_REGISTRATION_CONFLICT
               : MTT_PLUGIN_STATUS_REGISTRATION_FAILED;
  }
  if (!registry->registerCreator(gelu_tanh_creator, kPluginNamespace)) {
    static_cast<void>(registry->deregisterCreator(layer_norm_creator));
    static_cast<void>(registry->deregisterCreator(eos_creator));
    static_cast<void>(registry->deregisterCreator(sampling_creator));
    registered_gelu_tanh = registry->getCreator(
        kGeluTanhPluginName, kPluginVersion, kPluginNamespace);
    return registered_gelu_tanh != nullptr
               ? MTT_PLUGIN_STATUS_REGISTRATION_CONFLICT
               : MTT_PLUGIN_STATUS_REGISTRATION_FAILED;
  }
  if (!registry->registerCreator(softmax_creator, kPluginNamespace)) {
    static_cast<void>(registry->deregisterCreator(gelu_tanh_creator));
    static_cast<void>(registry->deregisterCreator(layer_norm_creator));
    static_cast<void>(registry->deregisterCreator(eos_creator));
    static_cast<void>(registry->deregisterCreator(sampling_creator));
    registered_softmax = registry->getCreator(
        kSoftmaxPluginName, kPluginVersion, kPluginNamespace);
    return registered_softmax != nullptr
               ? MTT_PLUGIN_STATUS_REGISTRATION_CONFLICT
               : MTT_PLUGIN_STATUS_REGISTRATION_FAILED;
  }
  return MTT_PLUGIN_STATUS_OK;
}

int launch_local_ar_sampling(
    const void* logits_bf16,
    const bool* unfinished,
    const bool* finished,
    const bool* forbid_eos,
    const std::int64_t* rng_seed,
    const std::int64_t* rng_counter,
    const void* embedding_weight_bf16,
    std::int64_t* sampled_token,
    void* next_embedding_bf16,
    std::int64_t* updated_rng_counter,
    std::int32_t* invalid_rows,
    cudaStream_t stream) noexcept {
  if (logits_bf16 == nullptr || unfinished == nullptr ||
      finished == nullptr || forbid_eos == nullptr || rng_seed == nullptr ||
      rng_counter == nullptr || embedding_weight_bf16 == nullptr ||
      sampled_token == nullptr || next_embedding_bf16 == nullptr ||
      updated_rng_counter == nullptr || invalid_rows == nullptr) {
    return -1;
  }
  sampling_kernel<<<1, kSamplingThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(logits_bf16),
      unfinished,
      finished,
      forbid_eos,
      rng_seed,
      rng_counter,
      static_cast<const __nv_bfloat16*>(embedding_weight_bf16),
      sampled_token,
      static_cast<__nv_bfloat16*>(next_embedding_bf16),
      updated_rng_counter,
      invalid_rows);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

std::size_t local_ar_eos_workspace_size() noexcept {
  return kAlignedGuidedEosBytes +
         static_cast<std::size_t>(kLocalArFrames) * sizeof(std::int32_t);
}

int launch_local_ar_eos(
    const void* decoder_hidden_bf16,
    const std::int64_t* codec_tokens,
    const bool* unfinished,
    const bool* finished,
    const bool* forbid_eos,
    const void* final_weight_bf16,
    const void* final_bias_bf16,
    void* workspace,
    std::int32_t* end_frame_index,
    cudaStream_t stream) noexcept {
  if (decoder_hidden_bf16 == nullptr || codec_tokens == nullptr ||
      unfinished == nullptr || finished == nullptr || forbid_eos == nullptr ||
      final_weight_bf16 == nullptr || final_bias_bf16 == nullptr ||
      workspace == nullptr || end_frame_index == nullptr) {
    return -1;
  }
  auto* workspace_bytes = static_cast<std::byte*>(workspace);
  auto* guided_logits = reinterpret_cast<float*>(workspace_bytes);
  auto* frame_eos = reinterpret_cast<std::int32_t*>(
      workspace_bytes + kAlignedGuidedEosBytes);
  if (cudaMemsetAsync(
          frame_eos,
          0,
          static_cast<std::size_t>(kLocalArFrames) * sizeof(std::int32_t),
          stream) != cudaSuccess) {
    return -1;
  }
  eos_projection_kernel<<<
      kLocalArPositions * kCompactEosVocabulary,
      kSamplingThreads,
      0,
      stream>>>(
      static_cast<const __nv_bfloat16*>(decoder_hidden_bf16),
      unfinished,
      finished,
      forbid_eos,
      static_cast<const __nv_bfloat16*>(final_weight_bf16),
      static_cast<const __nv_bfloat16*>(final_bias_bf16),
      guided_logits);
  eos_reduce_kernel<<<kLocalArPositions, kSamplingThreads, 0, stream>>>(
      guided_logits,
      codec_tokens,
      unfinished,
      finished,
      forbid_eos,
      frame_eos);
  eos_finalize_kernel<<<1, 1, 0, stream>>>(frame_eos, end_frame_index);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int launch_layer_norm(
    const void* input_bf16,
    const void* weight_bf16,
    void* output_bf16,
    const std::int32_t row_count,
    cudaStream_t stream) noexcept {
  if (input_bf16 == nullptr || weight_bf16 == nullptr ||
      output_bf16 == nullptr || row_count < 1 ||
      row_count > kMaximumNormalizationRows) {
    return -1;
  }
  constexpr dim3 threads(kLayerNormWarpSize, kLayerNormWarps, 1);
  constexpr std::size_t shared_bytes =
      static_cast<std::size_t>(kLayerNormWarps) * 3U / 2U *
      sizeof(float);
  layer_norm_kernel<<<row_count, threads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(input_bf16),
      static_cast<const __nv_bfloat16*>(weight_bf16),
      static_cast<__nv_bfloat16*>(output_bf16));
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int launch_gelu_tanh(
    const void* input_bf16,
    void* output_bf16,
    const std::size_t element_count,
    cudaStream_t stream) noexcept {
  constexpr std::size_t maximum_elements =
      static_cast<std::size_t>(kLocalArFeedForwardWidth) *
      static_cast<std::size_t>(kMaximumTextSequenceLength);
  if (input_bf16 == nullptr || output_bf16 == nullptr ||
      element_count == 0 || element_count > maximum_elements) {
    return -1;
  }
  const std::size_t blocks =
      (element_count + static_cast<std::size_t>(kGeluThreads) - 1U) /
      static_cast<std::size_t>(kGeluThreads);
  gelu_tanh_kernel<<<blocks, kGeluThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input_bf16),
      static_cast<__nv_bfloat16*>(output_bf16),
      element_count);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int launch_softmax(
    const void* input_bf16,
    void* output_bf16,
    const std::int32_t batch_count,
    const std::int32_t element_count,
    cudaStream_t stream) noexcept {
  if (input_bf16 == nullptr || output_bf16 == nullptr ||
      batch_count < 1 || element_count < 1 ||
      element_count > kMaximumTextSequenceLength) {
    return -1;
  }
  std::int32_t log2_elements = 0;
  while ((1 << log2_elements) < element_count) {
    ++log2_elements;
  }
  const std::int32_t next_power_of_two = 1 << log2_elements;
  const std::int32_t warp_size =
      std::min(next_power_of_two, 32);
  const std::int32_t batches_per_warp =
      next_power_of_two <= 128 ? 2 : 1;
  constexpr std::int32_t threads_per_block = 128;
  const std::int32_t warps_per_block =
      threads_per_block / warp_size;
  const std::int32_t batches_per_block =
      warps_per_block * batches_per_warp;
  const std::int32_t blocks =
      (batch_count + batches_per_block - 1) /
      batches_per_block;
  const dim3 threads(
      static_cast<unsigned>(warp_size),
      static_cast<unsigned>(warps_per_block),
      1U);
  const auto* input =
      static_cast<const __nv_bfloat16*>(input_bf16);
  auto* output = static_cast<__nv_bfloat16*>(output_bf16);
  switch (log2_elements) {
    case 0:
      persistent_softmax_kernel<0>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 1:
      persistent_softmax_kernel<1>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 2:
      persistent_softmax_kernel<2>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 3:
      persistent_softmax_kernel<3>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 4:
      persistent_softmax_kernel<4>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 5:
      persistent_softmax_kernel<5>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 6:
      persistent_softmax_kernel<6>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 7:
      persistent_softmax_kernel<7>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 8:
      persistent_softmax_kernel<8>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    case 9:
      persistent_softmax_kernel<9>
          <<<blocks, threads, 0, stream>>>(
              output, input, batch_count, element_count);
      break;
    default:
      return -1;
  }
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

std::size_t main_cross_attention_softmax_workspace_size() noexcept {
  return kMainCrossAttentionWorkspaceBytes;
}

int launch_main_cross_attention_softmax(
    cublasHandle_t cublas_handle,
    const void* query_bf16,
    const void* key_bf16,
    const bool* memory_mask,
    void* workspace,
    void* output_bf16,
    const std::int32_t query_length,
    const std::int32_t memory_length,
    cudaStream_t stream) noexcept {
  if (cublas_handle == nullptr ||
      query_bf16 == nullptr || key_bf16 == nullptr ||
      memory_mask == nullptr || workspace == nullptr ||
      output_bf16 == nullptr ||
      (query_length != 1 &&
       query_length != kMainDecoderPrefillLength) ||
      memory_length < 1 ||
      memory_length > kMaximumTextSequenceLength) {
    return -1;
  }
  const std::int32_t element_count =
      kMainDecoderBatch * query_length * memory_length;
  const std::int32_t blocks =
      (element_count + kMainCrossAttentionThreads - 1) /
      kMainCrossAttentionThreads;

  if (cublasSetStream(cublas_handle, stream) !=
      CUBLAS_STATUS_SUCCESS) {
    return -1;
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  if (cublasGemmStridedBatchedEx(
          cublas_handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          memory_length,
          query_length,
          kMainDecoderCrossAttentionWidth,
          &alpha,
          key_bf16,
          CUDA_R_16BF,
          kMainDecoderCrossAttentionWidth,
          static_cast<long long>(memory_length) *
              kMainDecoderCrossAttentionWidth,
          query_bf16,
          CUDA_R_16BF,
          kMainDecoderCrossAttentionWidth,
          static_cast<long long>(query_length) *
              kMainDecoderCrossAttentionWidth,
          &beta,
          workspace,
          CUDA_R_16BF,
          memory_length,
          static_cast<long long>(query_length) * memory_length,
          kMainDecoderBatch,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP) !=
      CUBLAS_STATUS_SUCCESS) {
    return -1;
  }

  main_cross_attention_scale_mask_kernel
      <<<blocks, kMainCrossAttentionThreads, 0, stream>>>(
          memory_mask,
          static_cast<__nv_bfloat16*>(workspace),
          query_length,
          memory_length);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return -1;
  }
  return launch_softmax(
      workspace,
      output_bf16,
      kMainDecoderBatch * query_length,
      memory_length,
      stream);
}

int launch_main_self_attention_context(
    const void* probabilities_bf16,
    const void* value_bf16,
    void* output_bf16,
    const std::int32_t query_length,
    const std::int32_t key_length,
    cudaStream_t stream) noexcept {
  if (probabilities_bf16 == nullptr || value_bf16 == nullptr ||
      output_bf16 == nullptr ||
      query_length != kMainDecoderPrefillLength ||
      key_length != kMainDecoderPrefillLength) {
    return -1;
  }

  constexpr std::int32_t output_width =
      kMainDecoderCrossAttentionWidth / 2;
  constexpr std::int32_t batch_count =
      kMainDecoderBatch * kAttentionHeads;
  using OutputOp =
      typename MainSelfAttentionContextKernel::Epilogue::OutputOp;
  const typename OutputOp::Params output_op{1.0F, 0.0F};

  // GemmUniversalAdapter transposes row-major problems internally.  Swapping
  // A/B and M/N here preserves the oracle's row-major
  // [Q,K] x [K,64] contract while selecting PyTorch's accepted CUTLASS
  // kernel configuration exactly.
  typename MainSelfAttentionContextGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kBatched,
      cutlass::gemm::GemmCoord{
          output_width,
          query_length,
          key_length,
      },
      batch_count,
      output_op,
      const_cast<void*>(value_bf16),
      const_cast<void*>(probabilities_bf16),
      output_bf16,
      output_bf16,
      static_cast<std::int64_t>(key_length) * output_width,
      static_cast<std::int64_t>(query_length) * key_length,
      static_cast<std::int64_t>(query_length) * output_width,
      static_cast<std::int64_t>(query_length) * output_width,
      output_width,
      key_length,
      output_width,
      output_width};

  MainSelfAttentionContextGemm operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess ||
      MainSelfAttentionContextGemm::get_workspace_size(arguments) != 0U) {
    return -1;
  }
  if (operation.initialize(arguments, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return -1;
  }
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -1;
}

std::size_t main_self_attention_step_context_workspace_size() noexcept {
  return kMainStepContextWorkspaceBytes;
}

int launch_main_self_attention_step_context(
    cublasHandle_t cublas_handle,
    const void* probabilities_bf16,
    const void* value_bf16,
    void* workspace,
    void* output_bf16,
    const std::int32_t active_length,
    cudaStream_t stream) noexcept {
  if (cublas_handle == nullptr || probabilities_bf16 == nullptr ||
      value_bf16 == nullptr || workspace == nullptr ||
      output_bf16 == nullptr ||
      active_length < kMainDecoderPrefillLength + 1 ||
      active_length > kMainDecoderCacheCapacity) {
    return -1;
  }
  auto* compact_values = static_cast<__nv_bfloat16*>(workspace);
  // The active QK/softmax path already produces a compact [24,1,K]
  // probability tensor.  PyTorch clones only the non-contiguous value slice
  // to [24,K,64] before the context GEMM.  Reconstruct that value layout in
  // bounded TensorRT workspace and keep the probability leading dimension
  // and batch stride equal to K; restoring the old capacity-467 probability
  // stride changes cuBLAS results for some accepted inputs.
  constexpr std::int32_t staging_threads = 256;
  const std::int32_t value_vectors =
      kMainSelfAttentionBatchHeads * active_length *
      (kMainSelfAttentionHeadWidth / 8);
  stage_main_step_values_kernel<<<
      (value_vectors + staging_threads - 1) / staging_threads,
      staging_threads,
      0,
      stream>>>(
      reinterpret_cast<
          const AlignedVector<__nv_bfloat16, 8>*>(value_bf16),
      reinterpret_cast<AlignedVector<__nv_bfloat16, 8>*>(
          compact_values),
      active_length);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return -1;
  }
  if (cublasSetStream(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return -1;
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  return cublasGemmStridedBatchedEx(
             cublas_handle,
             CUBLAS_OP_N,
             CUBLAS_OP_N,
             kMainSelfAttentionHeadWidth,
             1,
             active_length,
             &alpha,
             compact_values,
             CUDA_R_16BF,
             kMainSelfAttentionHeadWidth,
             static_cast<long long>(
                 active_length * kMainSelfAttentionHeadWidth),
             probabilities_bf16,
             CUDA_R_16BF,
             active_length,
             static_cast<long long>(active_length),
             &beta,
             output_bf16,
             CUDA_R_16BF,
             kMainSelfAttentionHeadWidth,
             static_cast<long long>(kMainSelfAttentionHeadWidth),
             kMainSelfAttentionBatchHeads,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS
             ? 0
             : -1;
}

int launch_main_self_attention_step_scores(
    cublasHandle_t cublas_handle,
    const void* query_bf16,
    const void* key_transposed_bf16,
    void* output_bf16,
    const std::int32_t active_length,
    cudaStream_t stream) noexcept {
  if (cublas_handle == nullptr || query_bf16 == nullptr ||
      key_transposed_bf16 == nullptr || output_bf16 == nullptr ||
      active_length < kMainDecoderPrefillLength + 1 ||
      active_length > kMainDecoderCacheCapacity) {
    return -1;
  }
  if (cublasSetStream(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return -1;
  }
  constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  constexpr std::int32_t batch_count =
      kMainDecoderBatch * kAttentionHeads;
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  return cublasGemmStridedBatchedEx(
             cublas_handle,
             CUBLAS_OP_N,
             CUBLAS_OP_N,
             active_length,
             1,
             head_width,
             &alpha,
             key_transposed_bf16,
             CUDA_R_16BF,
             active_length,
             static_cast<long long>(
                 head_width * active_length),
             query_bf16,
             CUDA_R_16BF,
             head_width,
             static_cast<long long>(head_width),
             &beta,
             output_bf16,
             CUDA_R_16BF,
             active_length,
             static_cast<long long>(active_length),
             batch_count,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP) ==
             CUBLAS_STATUS_SUCCESS
         ? 0
         : -1;
}

int launch_main_cross_attention_context(
    cublasHandle_t cublas_handle,
    const void* probabilities_bf16,
    const void* value_bf16,
    void* output_bf16,
    const std::int32_t query_length,
    const std::int32_t memory_length,
    cudaStream_t stream) noexcept {
  if (cublas_handle == nullptr ||
      probabilities_bf16 == nullptr || value_bf16 == nullptr ||
      output_bf16 == nullptr ||
      (query_length != 1 &&
       query_length != kMainDecoderPrefillLength) ||
      memory_length < 1 ||
      memory_length > kMaximumTextSequenceLength) {
    return -1;
  }

  if (query_length == 1) {
    using OutputOp =
        typename MainCrossAttentionContextKernel::Epilogue::OutputOp;
    const typename OutputOp::Params output_op{1.0F, 0.0F};
    typename MainCrossAttentionContextGemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kBatched,
        cutlass::gemm::GemmCoord{
            kMainDecoderCrossAttentionWidth,
            query_length,
            memory_length,
        },
        kMainDecoderBatch * kMainDecoderCrossAttentionHeads,
        output_op,
        const_cast<void*>(value_bf16),
        const_cast<void*>(probabilities_bf16),
        output_bf16,
        output_bf16,
        static_cast<std::int64_t>(memory_length) *
            kMainDecoderCrossAttentionWidth,
        static_cast<std::int64_t>(query_length) * memory_length,
        static_cast<std::int64_t>(query_length) *
            kMainDecoderCrossAttentionWidth,
        static_cast<std::int64_t>(query_length) *
            kMainDecoderCrossAttentionWidth,
        kMainDecoderCrossAttentionWidth,
        memory_length,
        kMainDecoderCrossAttentionWidth,
        kMainDecoderCrossAttentionWidth};

    MainCrossAttentionContextGemm operation;
    if (operation.can_implement(arguments) !=
            cutlass::Status::kSuccess ||
        MainCrossAttentionContextGemm::get_workspace_size(arguments) !=
            0U ||
        operation.initialize(arguments, nullptr, stream) !=
            cutlass::Status::kSuccess) {
      return -1;
    }
    return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -1;
  }

  if (cublasSetStream(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return -1;
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  return cublasGemmStridedBatchedEx(
             cublas_handle,
             CUBLAS_OP_N,
             CUBLAS_OP_N,
             kMainDecoderCrossAttentionWidth,
             query_length,
             memory_length,
             &alpha,
             value_bf16,
             CUDA_R_16BF,
             kMainDecoderCrossAttentionWidth,
             static_cast<long long>(
                 memory_length *
                 kMainDecoderCrossAttentionWidth),
             probabilities_bf16,
             CUDA_R_16BF,
             memory_length,
             static_cast<long long>(
                 query_length * memory_length),
             &beta,
             output_bf16,
             CUDA_R_16BF,
             kMainDecoderCrossAttentionWidth,
             static_cast<long long>(
                 query_length *
                 kMainDecoderCrossAttentionWidth),
             kMainDecoderBatch *
                 kMainDecoderCrossAttentionHeads,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP) ==
             CUBLAS_STATUS_SUCCESS
         ? 0
         : -1;
}

int launch_main_attention_prior_normalization(
    const void* probabilities_bf16,
    const void* attention_prior_bf16,
    void* output_bf16,
    const std::int32_t memory_length,
    cudaStream_t stream) noexcept {
  if (probabilities_bf16 == nullptr ||
      attention_prior_bf16 == nullptr ||
      output_bf16 == nullptr || memory_length < 1 ||
      memory_length > kMaximumTextSequenceLength) {
    return -1;
  }
  main_attention_prior_normalization_kernel
      <<<kMainDecoderBatch, kMainCrossAttentionThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(probabilities_bf16),
          static_cast<const __nv_bfloat16*>(attention_prior_bf16),
          static_cast<__nv_bfloat16*>(output_bf16),
          memory_length);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int launch_main_alignment_mean(
    const void* alignment_0_bf16,
    const void* alignment_1_bf16,
    const void* alignment_2_bf16,
    const void* alignment_3_bf16,
    void* output_bf16,
    const std::int32_t text_length,
    cudaStream_t stream) noexcept {
  if (alignment_0_bf16 == nullptr ||
      alignment_1_bf16 == nullptr ||
      alignment_2_bf16 == nullptr ||
      alignment_3_bf16 == nullptr ||
      output_bf16 == nullptr ||
      text_length < 1 ||
      text_length > kMaximumTextSequenceLength) {
    return -1;
  }
  const std::int32_t element_count =
      kMainDecoderBatch * text_length;
  const std::int32_t blocks =
      (element_count + kMainCrossAttentionThreads - 1) /
      kMainCrossAttentionThreads;
  main_alignment_mean_kernel
      <<<blocks, kMainCrossAttentionThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(alignment_0_bf16),
          static_cast<const __nv_bfloat16*>(alignment_1_bf16),
          static_cast<const __nv_bfloat16*>(alignment_2_bf16),
          static_cast<const __nv_bfloat16*>(alignment_3_bf16),
          static_cast<__nv_bfloat16*>(output_bf16),
          element_count);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

#if defined(MAGPIE_TTS_RT_PLUGIN_TESTING)
int launch_test_clamped_gumbel(
    const float* uniform,
    float* gumbel,
    const std::size_t count,
    cudaStream_t stream) noexcept {
  if (uniform == nullptr || gumbel == nullptr || count == 0) {
    return -1;
  }
  constexpr std::size_t threads = 256;
  const std::size_t blocks = (count + threads - 1) / threads;
  test_clamped_gumbel_kernel<<<blocks, threads, 0, stream>>>(
      uniform, gumbel, count);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}
#endif

}  // namespace magpie_tts_rt::plugins

extern "C" MTT_PLUGIN_API mtt_plugin_status_t
mtt_plugin_get_api_v1(mtt_plugin_api_v1_t* api) {
  if (api == nullptr) {
    return MTT_PLUGIN_STATUS_INVALID_ARGUMENT;
  }
  if (api->struct_size != sizeof(mtt_plugin_api_v1_t) ||
      api->abi_version != MTT_PLUGIN_ABI_VERSION_1) {
    return MTT_PLUGIN_STATUS_ABI_MISMATCH;
  }

  mtt_plugin_api_v1_t populated{};
  populated.struct_size = sizeof(populated);
  populated.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  populated.creator_count = MTT_PLUGIN_CREATOR_COUNT_V1;
  populated.register_plugins =
      &magpie_tts_rt::plugins::register_plugins_explicitly;

  const auto populate_creator = [](
                                    mtt_plugin_creator_v1_t& creator,
                                    const char* name) {
    creator.struct_size = sizeof(creator);
    creator.abi_version = MTT_PLUGIN_ABI_VERSION_1;
    static_cast<void>(std::strncpy(
        creator.name, name, MTT_PLUGIN_CREATOR_NAME_CAPACITY - 1U));
    static_cast<void>(std::strncpy(
        creator.version,
        magpie_tts_rt::plugins::kPluginVersion,
        MTT_PLUGIN_CREATOR_VERSION_CAPACITY - 1U));
    static_cast<void>(std::strncpy(
        creator.plugin_namespace,
        magpie_tts_rt::plugins::kPluginNamespace,
        MTT_PLUGIN_NAMESPACE_CAPACITY - 1U));
  };
  populate_creator(
      populated.creators[0],
      magpie_tts_rt::plugins::kSamplingPluginName);
  populate_creator(
      populated.creators[1], magpie_tts_rt::plugins::kEosPluginName);
  populate_creator(
      populated.creators[2],
      magpie_tts_rt::plugins::kLayerNormPluginName);
  populate_creator(
      populated.creators[3],
      magpie_tts_rt::plugins::kGeluTanhPluginName);
  populate_creator(
      populated.creators[4],
      magpie_tts_rt::plugins::kSoftmaxPluginName);
  *api = populated;
  return MTT_PLUGIN_STATUS_OK;
}

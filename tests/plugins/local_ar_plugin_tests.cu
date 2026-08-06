#include "local_ar_plugins.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using magpie_tts_rt::plugins::kLocalArAudioEosId;
using magpie_tts_rt::plugins::kLocalArCodebooks;
using magpie_tts_rt::plugins::kLocalArEmbeddingWidth;
using magpie_tts_rt::plugins::kLocalArFeedForwardWidth;
using magpie_tts_rt::plugins::kLocalArFrames;
using magpie_tts_rt::plugins::kLocalArPositions;
using magpie_tts_rt::plugins::kLocalArVocabularySize;
using magpie_tts_rt::plugins::kMainDecoderBatch;
using magpie_tts_rt::plugins::kMainDecoderCacheCapacity;
using magpie_tts_rt::plugins::kMainDecoderCrossAttentionWidth;
using magpie_tts_rt::plugins::kMainDecoderPrefillLength;
using magpie_tts_rt::plugins::kAttentionHeads;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_cuda(const cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        operation + ": " + std::string(cudaGetErrorString(status)));
  }
}

template <typename Element>
class ManagedBuffer final {
 public:
  explicit ManagedBuffer(const std::size_t count) : count_(count) {
    require_cuda(
        cudaMallocManaged(
            reinterpret_cast<void**>(&data_), count_ * sizeof(Element)),
        "cudaMallocManaged");
  }

  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  ~ManagedBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  [[nodiscard]] Element* data() noexcept { return data_; }
  [[nodiscard]] const Element* data() const noexcept { return data_; }
  [[nodiscard]] Element& operator[](const std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] const Element& operator[](
      const std::size_t index) const noexcept {
    return data_[index];
  }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  Element* data_{nullptr};
  std::size_t count_;
};

template <typename Element>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    require_cuda(
        cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(Element)),
        "cudaMalloc");
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  [[nodiscard]] Element* data() noexcept { return data_; }
  [[nodiscard]] const Element* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  Element* data_{nullptr};
  std::size_t count_;
};

void fill_bf16(ManagedBuffer<__nv_bfloat16>& values, const float value) {
  std::fill(
      values.data(),
      values.data() + values.size(),
      __float2bfloat16_rn(value));
}

struct SamplingBuffers final {
  ManagedBuffer<__nv_bfloat16> logits{
      2U * static_cast<std::size_t>(kLocalArVocabularySize)};
  ManagedBuffer<__nv_bfloat16> embedding{
      static_cast<std::size_t>(kLocalArVocabularySize) *
      static_cast<std::size_t>(kLocalArEmbeddingWidth)};
  ManagedBuffer<bool> unfinished{1};
  ManagedBuffer<bool> finished{1};
  ManagedBuffer<bool> forbid_eos{1};
  ManagedBuffer<std::int64_t> seed{1};
  ManagedBuffer<std::int64_t> counter{1};
  ManagedBuffer<std::int64_t> sampled{1};
  ManagedBuffer<__nv_bfloat16> next_embedding{
      2U * static_cast<std::size_t>(kLocalArEmbeddingWidth)};
  ManagedBuffer<std::int64_t> updated_counter{1};
  ManagedBuffer<std::int32_t> invalid_rows{1};
};

void reset_sampling(SamplingBuffers& buffers) {
  fill_bf16(buffers.logits, 0.0F);
  for (std::size_t index = 0; index < buffers.embedding.size(); ++index) {
    buffers.embedding[index] =
        __float2bfloat16_rn(static_cast<float>(index % 251U));
  }
  buffers.unfinished[0] = true;
  buffers.finished[0] = false;
  buffers.forbid_eos[0] = true;
  buffers.seed[0] = 20260729;
  buffers.counter[0] = 0;
  buffers.sampled[0] = -1;
  buffers.updated_counter[0] = -1;
  buffers.invalid_rows[0] = -1;
}

void run_sampling(SamplingBuffers& buffers) {
  require(
      magpie_tts_rt::plugins::launch_local_ar_sampling(
          buffers.logits.data(),
          buffers.unfinished.data(),
          buffers.finished.data(),
          buffers.forbid_eos.data(),
          buffers.seed.data(),
          buffers.counter.data(),
          buffers.embedding.data(),
          buffers.sampled.data(),
          buffers.next_embedding.data(),
          buffers.updated_counter.data(),
          buffers.invalid_rows.data(),
          nullptr) == 0,
      "sampling launch failed");
  require_cuda(cudaDeviceSynchronize(), "sampling synchronize");
}

void test_sampling_contract() {
  SamplingBuffers buffers;
  reset_sampling(buffers);
  run_sampling(buffers);
  require(buffers.sampled[0] == 1871, "inclusive top-k/Philox token mismatch");
  require(buffers.updated_counter[0] == 1, "counter did not advance");
  require(buffers.invalid_rows[0] == 0, "valid row marked invalid");
  for (std::int32_t width = 0; width < kLocalArEmbeddingWidth; ++width) {
    const std::size_t source =
        static_cast<std::size_t>(buffers.sampled[0]) *
            static_cast<std::size_t>(kLocalArEmbeddingWidth) +
        static_cast<std::size_t>(width);
    require(
        __bfloat16_as_ushort(buffers.next_embedding[width]) ==
            __bfloat16_as_ushort(buffers.embedding[source]),
        "conditional embedding gather mismatch");
    require(
        __bfloat16_as_ushort(
            buffers.next_embedding[kLocalArEmbeddingWidth + width]) ==
            __bfloat16_as_ushort(buffers.embedding[source]),
        "unconditional embedding gather mismatch");
  }

  reset_sampling(buffers);
  fill_bf16(buffers.logits, -1000.0F);
  constexpr std::int32_t dominant_token = 7;
  buffers.logits[dominant_token] = __float2bfloat16_rn(1000.0F);
  buffers.logits[kLocalArVocabularySize + dominant_token] =
      __float2bfloat16_rn(1000.0F);
  run_sampling(buffers);
  require(buffers.sampled[0] == dominant_token, "dominant token mismatch");

  reset_sampling(buffers);
  buffers.finished[0] = true;
  buffers.unfinished[0] = false;
  buffers.forbid_eos[0] = false;
  run_sampling(buffers);
  require(
      buffers.sampled[0] == kLocalArAudioEosId,
      "finished row did not force EOS");
  require(buffers.invalid_rows[0] == 0, "forced EOS row marked invalid");

  reset_sampling(buffers);
  buffers.finished[0] = true;
  run_sampling(buffers);
  require(
      buffers.invalid_rows[0] != 0,
      "overlapping finished/unfinished/forbid status was accepted");

  reset_sampling(buffers);
  buffers.seed[0] = -1;
  run_sampling(buffers);
  require(buffers.invalid_rows[0] != 0, "negative seed was accepted");
  require(
      buffers.updated_counter[0] == 1,
      "valid counter must advance even when seed is invalid");

  reset_sampling(buffers);
  constexpr std::int64_t maximum_counter =
      (std::numeric_limits<std::int64_t>::max() - 2047) / 2048;
  buffers.counter[0] = maximum_counter + 1;
  run_sampling(buffers);
  require(buffers.invalid_rows[0] != 0, "overflowing counter was accepted");
  require(
      buffers.updated_counter[0] == maximum_counter + 1,
      "overflowing counter must remain unchanged");

  reset_sampling(buffers);
  fill_bf16(buffers.logits, std::numeric_limits<float>::quiet_NaN());
  run_sampling(buffers);
  require(buffers.invalid_rows[0] != 0, "NaN logits were accepted");

  require(
      magpie_tts_rt::plugins::launch_local_ar_sampling(
          nullptr,
          buffers.unfinished.data(),
          buffers.finished.data(),
          buffers.forbid_eos.data(),
          buffers.seed.data(),
          buffers.counter.data(),
          buffers.embedding.data(),
          buffers.sampled.data(),
          buffers.next_embedding.data(),
          buffers.updated_counter.data(),
          buffers.invalid_rows.data(),
          nullptr) == -1,
      "null sampling input was accepted");
}

void test_clamp_endpoints() {
  ManagedBuffer<float> uniform{4};
  ManagedBuffer<float> gumbel{4};
  uniform[0] = 0.0F;
  uniform[1] = 0.00000006F;
  uniform[2] = 1.0F;
  uniform[3] = 0.99999994F;
  require(
      magpie_tts_rt::plugins::launch_test_clamped_gumbel(
          uniform.data(), gumbel.data(), uniform.size(), nullptr) == 0,
      "clamped Gumbel launch failed");
  require_cuda(cudaDeviceSynchronize(), "clamped Gumbel synchronize");
  require(
      gumbel[0] == gumbel[1],
      "lower uniform endpoint was not clamped exactly");
  require(
      gumbel[2] == gumbel[3],
      "upper uniform endpoint was not clamped exactly");
  require(
      std::isfinite(gumbel[0]) && std::isfinite(gumbel[2]),
      "clamped Gumbel endpoint is not finite");
}

void test_oracle_math_plugins() {
  constexpr std::size_t normalization_rows = 3U;
  constexpr std::size_t normalization_elements =
      normalization_rows *
      static_cast<std::size_t>(kLocalArEmbeddingWidth);
  ManagedBuffer<__nv_bfloat16> normalization_input{
      normalization_elements};
  ManagedBuffer<__nv_bfloat16> normalization_weight{
      static_cast<std::size_t>(kLocalArEmbeddingWidth)};
  ManagedBuffer<__nv_bfloat16> normalization_output{
      normalization_elements};
  for (std::size_t index = 0; index < normalization_input.size(); ++index) {
    normalization_input[index] =
        __float2bfloat16_rn(index % 2U == 0U ? -1.0F : 1.0F);
  }
  fill_bf16(normalization_weight, 1.0F);
  fill_bf16(normalization_output, 9.0F);
  require(
      magpie_tts_rt::plugins::launch_layer_norm(
          normalization_input.data(),
          normalization_weight.data(),
          normalization_output.data(),
          static_cast<std::int32_t>(normalization_rows),
          nullptr) == 0,
      "LayerNorm launch failed");
  require_cuda(cudaDeviceSynchronize(), "LayerNorm synchronize");
  for (std::size_t index = 0; index < normalization_output.size(); ++index) {
    const __nv_bfloat16 expected =
        __float2bfloat16_rn(index % 2U == 0U ? -1.0F : 1.0F);
    require(
        __bfloat16_as_ushort(normalization_output[index]) ==
            __bfloat16_as_ushort(expected),
        "LayerNorm Welford output mismatch");
  }
  require(
      magpie_tts_rt::plugins::launch_layer_norm(
          nullptr,
          normalization_weight.data(),
          normalization_output.data(),
          static_cast<std::int32_t>(normalization_rows),
          nullptr) == -1,
      "null LayerNorm input was accepted");

  constexpr std::size_t gelu_elements =
      3U * static_cast<std::size_t>(kLocalArFeedForwardWidth);
  ManagedBuffer<__nv_bfloat16> gelu_input{gelu_elements};
  ManagedBuffer<__nv_bfloat16> gelu_output{gelu_elements};
  constexpr float beta = 0.7978845608028654F;
  constexpr float kappa = 0.044715F;
  for (std::size_t index = 0; index < gelu_input.size(); ++index) {
    const float input = static_cast<float>(
        static_cast<std::int32_t>(index % 9U) - 4);
    gelu_input[index] = __float2bfloat16_rn(input);
    gelu_output[index] = __float2bfloat16_rn(9.0F);
  }
  require(
      magpie_tts_rt::plugins::launch_gelu_tanh(
          gelu_input.data(),
          gelu_output.data(),
          gelu_elements,
          nullptr) == 0,
      "GELU launch failed");
  require_cuda(cudaDeviceSynchronize(), "GELU synchronize");
  for (std::size_t index = 0; index < gelu_output.size(); ++index) {
    const float input = __bfloat162float(gelu_input[index]);
    const float input_cube = input * input * input;
    const float inner = beta * (input + kappa * input_cube);
    const __nv_bfloat16 expected = __float2bfloat16_rn(
        0.5F * input * (1.0F + std::tanh(inner)));
    require(
        __bfloat16_as_ushort(gelu_output[index]) ==
            __bfloat16_as_ushort(expected),
        "GELU tanh output mismatch");
  }
  require(
      magpie_tts_rt::plugins::launch_gelu_tanh(
          nullptr, gelu_output.data(), gelu_elements, nullptr) == -1,
      "null GELU input was accepted");

  constexpr std::array<std::int32_t, 10> softmax_lengths{
      1, 2, 3, 16, 17, 128, 129, 218, 467, 512};
  constexpr std::int32_t softmax_batches = 3;
  ManagedBuffer<__nv_bfloat16> softmax_input{
      static_cast<std::size_t>(softmax_batches) *
      static_cast<std::size_t>(softmax_lengths.back())};
  ManagedBuffer<__nv_bfloat16> softmax_output{
      static_cast<std::size_t>(softmax_batches) *
      static_cast<std::size_t>(softmax_lengths.back())};
  for (const std::int32_t length : softmax_lengths) {
    fill_bf16(softmax_input, 0.0F);
    fill_bf16(softmax_output, 9.0F);
    require(
        magpie_tts_rt::plugins::launch_softmax(
            softmax_input.data(),
            softmax_output.data(),
            softmax_batches,
            length,
            nullptr) == 0,
        "Softmax launch failed");
    require_cuda(cudaDeviceSynchronize(), "Softmax synchronize");
    const __nv_bfloat16 expected =
        __float2bfloat16_rn(1.0F / static_cast<float>(length));
    const std::size_t count =
        static_cast<std::size_t>(softmax_batches) *
        static_cast<std::size_t>(length);
    for (std::size_t index = 0; index < count; ++index) {
      require(
          __bfloat16_as_ushort(softmax_output[index]) ==
              __bfloat16_as_ushort(expected),
          "PersistentSoftmax uniform output mismatch");
    }
  }
  require(
      magpie_tts_rt::plugins::launch_softmax(
          nullptr,
          softmax_output.data(),
          softmax_batches,
          16,
          nullptr) == -1,
      "null Softmax input was accepted");
  require(
      magpie_tts_rt::plugins::launch_softmax(
          softmax_input.data(),
          softmax_output.data(),
          softmax_batches,
          513,
          nullptr) == -1,
      "oversized Softmax input was accepted");
}

void test_main_cross_attention_softmax() {
  constexpr std::int32_t query_length = 1;
  constexpr std::int32_t memory_length = 2;
  ManagedBuffer<__nv_bfloat16> query{
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(query_length) *
      static_cast<std::size_t>(kMainDecoderCrossAttentionWidth)};
  ManagedBuffer<__nv_bfloat16> key{
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(memory_length) *
      static_cast<std::size_t>(kMainDecoderCrossAttentionWidth)};
  ManagedBuffer<bool> memory_mask{
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(memory_length)};
  ManagedBuffer<std::byte> workspace{
      magpie_tts_rt::plugins::
          main_cross_attention_softmax_workspace_size()};
  ManagedBuffer<__nv_bfloat16> probabilities{
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(query_length) *
      static_cast<std::size_t>(memory_length)};
  fill_bf16(query, 0.0F);
  fill_bf16(key, 0.0F);
  fill_bf16(probabilities, 9.0F);
  query[0] = __float2bfloat16_rn(2.0F);
  query[kMainDecoderCrossAttentionWidth] =
      __float2bfloat16_rn(-1.0F);
  key[0] = __float2bfloat16_rn(3.0F);
  key[
      3U * static_cast<std::size_t>(
               kMainDecoderCrossAttentionWidth)] =
      __float2bfloat16_rn(5.0F);
  memory_mask[0] = true;
  memory_mask[1] = false;
  memory_mask[2] = false;
  memory_mask[3] = true;
  cublasHandle_t cublas_handle = nullptr;
  require(
      cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main cross-attention Softmax cuBLAS handle creation failed");
  require(
      cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH) ==
          CUBLAS_STATUS_SUCCESS,
      "Main cross-attention Softmax cuBLAS math mode failed");
  require(
      magpie_tts_rt::plugins::launch_main_cross_attention_softmax(
          cublas_handle,
          query.data(),
          key.data(),
          memory_mask.data(),
          workspace.data(),
          probabilities.data(),
          query_length,
          memory_length,
          nullptr) == 0,
      "Main cross-attention Softmax launch failed");
  require_cuda(
      cudaDeviceSynchronize(),
      "Main cross-attention Softmax synchronize");
  const auto* scores =
      reinterpret_cast<const __nv_bfloat16*>(workspace.data());
  constexpr float scale = 0.08838834764831845F;
  const __nv_bfloat16 expected_positive =
      __float2bfloat16_rn(6.0F * scale);
  const __nv_bfloat16 expected_negative =
      __float2bfloat16_rn(-5.0F * scale);
  require(
      __bfloat16_as_ushort(scores[0]) ==
          __bfloat16_as_ushort(expected_positive),
      "Main cross-attention positive dot mismatch");
  require(
      std::isinf(__bfloat162float(scores[1])) &&
          __bfloat162float(scores[1]) < 0.0F &&
          std::isinf(__bfloat162float(scores[2])) &&
          __bfloat162float(scores[2]) < 0.0F,
      "Main cross-attention mask did not emit negative infinity");
  require(
      __bfloat16_as_ushort(scores[3]) ==
          __bfloat16_as_ushort(expected_negative),
      "Main cross-attention negative dot mismatch");
  const __nv_bfloat16 one = __float2bfloat16_rn(1.0F);
  const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
  for (const std::size_t index : {0U, 3U}) {
    require(
        __bfloat16_as_ushort(probabilities[index]) ==
            __bfloat16_as_ushort(one),
        "Main cross-attention valid probability mismatch");
  }
  for (const std::size_t index : {1U, 2U}) {
    require(
        __bfloat16_as_ushort(probabilities[index]) ==
            __bfloat16_as_ushort(zero),
        "Main cross-attention masked probability mismatch");
  }
  require(
      cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main cross-attention Softmax cuBLAS handle destruction failed");
}

void test_main_self_attention_context() {
  constexpr std::int32_t query_length = kMainDecoderPrefillLength;
  constexpr std::int32_t key_length = kMainDecoderPrefillLength;
  constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  constexpr std::int32_t batch_heads =
      kMainDecoderBatch * kAttentionHeads;
  ManagedBuffer<__nv_bfloat16> probabilities{
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(query_length) *
      static_cast<std::size_t>(key_length)};
  ManagedBuffer<__nv_bfloat16> value{
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(key_length) *
      static_cast<std::size_t>(head_width)};
  ManagedBuffer<__nv_bfloat16> context{
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(query_length) *
      static_cast<std::size_t>(head_width)};
  fill_bf16(probabilities, 0.0F);
  fill_bf16(value, 0.0F);
  fill_bf16(context, -9.0F);

  for (std::int32_t batch_head = 0; batch_head < batch_heads;
       ++batch_head) {
    const std::size_t probabilities_base =
        static_cast<std::size_t>(batch_head) *
        static_cast<std::size_t>(query_length) *
        static_cast<std::size_t>(key_length);
    const std::size_t value_base =
        static_cast<std::size_t>(batch_head) *
        static_cast<std::size_t>(key_length) *
        static_cast<std::size_t>(head_width);
    for (std::int32_t query = 0; query < query_length; ++query) {
      probabilities[
          probabilities_base +
          static_cast<std::size_t>(query) * key_length] =
          __float2bfloat16_rn(1.0F);
    }
    for (std::int32_t column = 0; column < head_width; ++column) {
      value[value_base + static_cast<std::size_t>(column)] =
          __float2bfloat16_rn(
              static_cast<float>((batch_head + column) % 17) / 16.0F);
    }
  }

  require(
      magpie_tts_rt::plugins::launch_main_self_attention_context(
          probabilities.data(),
          value.data(),
          context.data(),
          query_length,
          key_length,
          nullptr) == 0,
      "Main self-attention context launch failed");
  require_cuda(
      cudaDeviceSynchronize(),
      "Main self-attention context synchronize");

  for (std::int32_t batch_head = 0; batch_head < batch_heads;
       ++batch_head) {
    const std::size_t context_base =
        static_cast<std::size_t>(batch_head) *
        static_cast<std::size_t>(query_length) *
        static_cast<std::size_t>(head_width);
    for (std::int32_t query = 0; query < query_length; ++query) {
      for (std::int32_t column = 0; column < head_width; ++column) {
        const __nv_bfloat16 expected = __float2bfloat16_rn(
            static_cast<float>((batch_head + column) % 17) / 16.0F);
        const std::size_t index =
            context_base +
            static_cast<std::size_t>(query) * head_width +
            static_cast<std::size_t>(column);
        require(
            __bfloat16_as_ushort(context[index]) ==
                __bfloat16_as_ushort(expected),
            "Main self-attention context value mismatch");
      }
    }
  }

  require(
      magpie_tts_rt::plugins::launch_main_self_attention_context(
          probabilities.data(),
          value.data(),
          context.data(),
          2,
          key_length,
          nullptr) == -1,
      "unsupported Main self-attention shape was accepted");

  require(
      magpie_tts_rt::plugins::launch_main_self_attention_context(
          probabilities.data(),
          value.data(),
          context.data(),
          1,
          kMainDecoderCacheCapacity,
          nullptr) == -1,
      "prefill mode accepted a one-step self-attention shape");
}

void test_main_self_attention_step_context() {
  constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  constexpr std::int32_t batch_heads =
      kMainDecoderBatch * kAttentionHeads;
  ManagedBuffer<__nv_bfloat16> value{
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(kMainDecoderCacheCapacity) *
      static_cast<std::size_t>(head_width)};
  ManagedBuffer<__nv_bfloat16> context{
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(head_width)};
  DeviceBuffer<std::byte> workspace{
      magpie_tts_rt::plugins::
          main_self_attention_step_context_workspace_size()};
  fill_bf16(value, 0.0F);
  fill_bf16(context, -9.0F);
  for (std::int32_t batch_head = 0; batch_head < batch_heads;
       ++batch_head) {
    const std::size_t value_base =
        static_cast<std::size_t>(batch_head) *
        static_cast<std::size_t>(kMainDecoderCacheCapacity) *
        static_cast<std::size_t>(head_width);
    for (std::int32_t column = 0; column < head_width; ++column) {
      value[value_base + static_cast<std::size_t>(column)] =
          __float2bfloat16_rn(
              static_cast<float>((batch_head + column) % 17) / 16.0F);
    }
  }
  cublasHandle_t cublas_handle = nullptr;
  require(
      cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main self-attention step context cuBLAS handle creation failed");
  for (std::int32_t active_length =
           kMainDecoderPrefillLength + 1;
       active_length <= kMainDecoderCacheCapacity;
       ++active_length) {
    ManagedBuffer<__nv_bfloat16> probabilities{
        static_cast<std::size_t>(batch_heads) *
        static_cast<std::size_t>(active_length)};
    fill_bf16(probabilities, 0.0F);
    for (std::int32_t batch_head = 0; batch_head < batch_heads;
         ++batch_head) {
      probabilities[
          static_cast<std::size_t>(batch_head) *
          static_cast<std::size_t>(active_length)] =
          __float2bfloat16_rn(1.0F);
    }
    fill_bf16(context, -9.0F);
    require(
        magpie_tts_rt::plugins::
            launch_main_self_attention_step_context(
                cublas_handle,
                probabilities.data(),
                value.data(),
                workspace.data(),
                context.data(),
                active_length,
                nullptr) == 0,
        "Main self-attention step context launch failed");
    require_cuda(
        cudaDeviceSynchronize(),
        "Main self-attention step context synchronize");
    for (std::int32_t batch_head = 0; batch_head < batch_heads;
         ++batch_head) {
      const std::size_t context_base =
          static_cast<std::size_t>(batch_head) *
          static_cast<std::size_t>(head_width);
      for (std::int32_t column = 0; column < head_width; ++column) {
        const __nv_bfloat16 expected = __float2bfloat16_rn(
            static_cast<float>((batch_head + column) % 17) / 16.0F);
        require(
            __bfloat16_as_ushort(
                context[
                    context_base +
                    static_cast<std::size_t>(column)]) ==
                __bfloat16_as_ushort(expected),
            "Main self-attention step context value mismatch");
      }
    }
  }
  require(
      magpie_tts_rt::plugins::launch_main_self_attention_step_context(
          cublas_handle,
          value.data(),
          value.data(),
          workspace.data(),
          context.data(),
          kMainDecoderPrefillLength,
          nullptr) == -1,
      "Main self-attention step context accepted a pre-prefix position");
  require(
      magpie_tts_rt::plugins::launch_main_self_attention_step_context(
          cublas_handle,
          value.data(),
          value.data(),
          workspace.data(),
          context.data(),
          kMainDecoderCacheCapacity + 1,
          nullptr) == -1,
      "Main self-attention step context accepted a cache overflow");
  require(
      cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main self-attention step context cuBLAS handle destruction failed");
}

void test_main_cross_attention_context() {
  constexpr std::int32_t memory_length = 3;
  constexpr std::int32_t output_width =
      kMainDecoderCrossAttentionWidth;
  ManagedBuffer<__nv_bfloat16> value{
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(memory_length) *
      static_cast<std::size_t>(output_width)};
  fill_bf16(value, 0.0F);
  cublasHandle_t cublas_handle = nullptr;
  require(
      cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main cross-attention context cuBLAS handle creation failed");

  for (std::int32_t batch = 0; batch < kMainDecoderBatch; ++batch) {
    const std::size_t value_base =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(memory_length) *
        static_cast<std::size_t>(output_width);
    for (std::int32_t column = 0; column < output_width; ++column) {
      value[
          value_base + static_cast<std::size_t>(output_width) +
          static_cast<std::size_t>(column)] =
          __float2bfloat16_rn(
              static_cast<float>((batch + column) % 19) / 18.0F);
    }
  }

  for (const std::int32_t query_length :
       {1, kMainDecoderPrefillLength}) {
    ManagedBuffer<__nv_bfloat16> probabilities{
        static_cast<std::size_t>(kMainDecoderBatch) *
        static_cast<std::size_t>(query_length) *
        static_cast<std::size_t>(memory_length)};
    ManagedBuffer<__nv_bfloat16> context{
        static_cast<std::size_t>(kMainDecoderBatch) *
        static_cast<std::size_t>(query_length) *
        static_cast<std::size_t>(output_width)};
    fill_bf16(probabilities, 0.0F);
    fill_bf16(context, -9.0F);
    for (std::int32_t batch = 0;
         batch < kMainDecoderBatch;
         ++batch) {
      const std::size_t probabilities_base =
          static_cast<std::size_t>(batch) *
          static_cast<std::size_t>(query_length) *
          static_cast<std::size_t>(memory_length);
      for (std::int32_t query = 0;
           query < query_length;
           ++query) {
        probabilities[
            probabilities_base +
            static_cast<std::size_t>(query) * memory_length + 1U] =
            __float2bfloat16_rn(1.0F);
      }
    }

    require(
        magpie_tts_rt::plugins::launch_main_cross_attention_context(
            cublas_handle,
            probabilities.data(),
            value.data(),
            context.data(),
            query_length,
            memory_length,
            nullptr) == 0,
        "Main cross-attention context launch failed");
    require_cuda(
        cudaDeviceSynchronize(),
        "Main cross-attention context synchronize");

    for (std::int32_t batch = 0;
         batch < kMainDecoderBatch;
         ++batch) {
      const std::size_t context_base =
          static_cast<std::size_t>(batch) *
          static_cast<std::size_t>(query_length) *
          static_cast<std::size_t>(output_width);
      for (std::int32_t query = 0;
           query < query_length;
           ++query) {
        for (std::int32_t column = 0;
             column < output_width;
             ++column) {
          const __nv_bfloat16 expected = __float2bfloat16_rn(
              static_cast<float>((batch + column) % 19) / 18.0F);
          const std::size_t index =
              context_base +
              static_cast<std::size_t>(query) * output_width +
              static_cast<std::size_t>(column);
          require(
              __bfloat16_as_ushort(context[index]) ==
                  __bfloat16_as_ushort(expected),
              "Main cross-attention context value mismatch");
        }
      }
    }
  }
  require(
      cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main cross-attention context cuBLAS handle destruction failed");
}

void test_main_self_attention_step_scores() {
  constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  constexpr std::int32_t batch_heads =
      kMainDecoderBatch * kAttentionHeads;
  constexpr std::int32_t selected_key_column = 3;
  ManagedBuffer<__nv_bfloat16> query{
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(head_width)};
  fill_bf16(query, 0.0F);
  for (std::int32_t batch_head = 0;
       batch_head < batch_heads;
       ++batch_head) {
    query[
        static_cast<std::size_t>(batch_head) * head_width +
        selected_key_column] = __float2bfloat16_rn(1.0F);
  }

  cublasHandle_t cublas_handle = nullptr;
  require(
      cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main self-attention step scores cuBLAS handle creation failed");
  for (std::int32_t active_length =
           kMainDecoderPrefillLength + 1;
       active_length <= kMainDecoderCacheCapacity;
       ++active_length) {
    ManagedBuffer<__nv_bfloat16> key_transposed{
        static_cast<std::size_t>(batch_heads) *
        static_cast<std::size_t>(head_width) *
        static_cast<std::size_t>(active_length)};
    fill_bf16(key_transposed, 0.0F);
    for (std::int32_t batch_head = 0;
         batch_head < batch_heads;
         ++batch_head) {
      const std::size_t key_base =
          static_cast<std::size_t>(batch_head) *
          static_cast<std::size_t>(head_width) *
          static_cast<std::size_t>(active_length) +
          static_cast<std::size_t>(selected_key_column) *
          static_cast<std::size_t>(active_length);
      for (std::int32_t cache_position = 0;
           cache_position < active_length;
           ++cache_position) {
        key_transposed[
            key_base + static_cast<std::size_t>(cache_position)] =
            __float2bfloat16_rn(
                static_cast<float>(
                    (batch_head + cache_position) % 31) /
                30.0F);
      }
    }
    ManagedBuffer<__nv_bfloat16> scores{
        static_cast<std::size_t>(batch_heads) *
        static_cast<std::size_t>(active_length)};
    fill_bf16(scores, -9.0F);
    require(
        magpie_tts_rt::plugins::
            launch_main_self_attention_step_scores(
                cublas_handle,
                query.data(),
                key_transposed.data(),
                scores.data(),
                active_length,
                nullptr) == 0,
        "Main self-attention step scores launch failed");
    require_cuda(
        cudaDeviceSynchronize(),
        "Main self-attention step scores synchronize");
    for (std::int32_t batch_head = 0;
         batch_head < batch_heads;
         ++batch_head) {
      const std::size_t score_base =
          static_cast<std::size_t>(batch_head) *
          static_cast<std::size_t>(active_length);
      for (std::int32_t cache_position = 0;
           cache_position < active_length;
           ++cache_position) {
        const __nv_bfloat16 expected = __float2bfloat16_rn(
            static_cast<float>(
                (batch_head + cache_position) % 31) /
            30.0F);
        require(
            __bfloat16_as_ushort(
                scores[
                    score_base +
                    static_cast<std::size_t>(cache_position)]) ==
                __bfloat16_as_ushort(expected),
            "Main self-attention step scores value mismatch");
      }
    }
  }
  require(
      cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS,
      "Main self-attention step scores cuBLAS handle destruction failed");
}

void test_main_attention_prior_normalization() {
  constexpr std::int32_t memory_length = 53;
  ManagedBuffer<__nv_bfloat16> probabilities{
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(memory_length)};
  ManagedBuffer<__nv_bfloat16> attention_prior{probabilities.size()};
  ManagedBuffer<__nv_bfloat16> output{probabilities.size()};
  for (std::int32_t batch = 0;
       batch < kMainDecoderBatch;
       ++batch) {
    for (std::int32_t index = 0; index < memory_length; ++index) {
      const std::size_t offset =
          static_cast<std::size_t>(batch) * memory_length +
          static_cast<std::size_t>(index);
      probabilities[offset] = __float2bfloat16_rn(
          static_cast<float>((index * 7 + batch * 3) % 31 + 1) /
          512.0F);
      attention_prior[offset] = __float2bfloat16_rn(
          static_cast<float>((index * 11 + batch * 5) % 37 + 1) /
          128.0F);
    }
  }
  fill_bf16(output, -9.0F);

  require(
      magpie_tts_rt::plugins::launch_main_attention_prior_normalization(
          probabilities.data(),
          attention_prior.data(),
          output.data(),
          memory_length,
          nullptr) == 0,
      "Main attention-prior normalization launch failed");
  require_cuda(
      cudaDeviceSynchronize(),
      "Main attention-prior normalization synchronize");
  for (std::int32_t batch = 0;
       batch < kMainDecoderBatch;
       ++batch) {
    std::array<__nv_bfloat16, memory_length> numerators{};
    float denominator = 0.0F;
    for (std::int32_t index = 0; index < memory_length; ++index) {
      const std::size_t offset =
          static_cast<std::size_t>(batch) * memory_length +
          static_cast<std::size_t>(index);
      const __nv_bfloat16 rounded_prior = __float2bfloat16_rn(
          __bfloat162float(attention_prior[offset]) + 0x1p-126F);
      numerators[static_cast<std::size_t>(index)] =
          __float2bfloat16_rn(
              __bfloat162float(probabilities[offset]) *
              __bfloat162float(rounded_prior));
      denominator += __bfloat162float(
          numerators[static_cast<std::size_t>(index)]);
    }
    const __nv_bfloat16 rounded_denominator =
        __float2bfloat16_rn(denominator);
    for (std::int32_t index = 0; index < memory_length; ++index) {
      const std::size_t offset =
          static_cast<std::size_t>(batch) * memory_length +
          static_cast<std::size_t>(index);
      const __nv_bfloat16 expected = __float2bfloat16_rn(
          __bfloat162float(
              numerators[static_cast<std::size_t>(index)]) /
          __bfloat162float(rounded_denominator));
      require(
          __bfloat16_as_ushort(output[offset]) ==
              __bfloat16_as_ushort(expected),
          "Main attention-prior normalization value mismatch");
    }
  }
  require(
      magpie_tts_rt::plugins::launch_main_attention_prior_normalization(
          nullptr,
          attention_prior.data(),
          output.data(),
          memory_length,
          nullptr) == -1,
      "null Main attention probabilities were accepted");
}

void test_main_alignment_mean() {
  constexpr std::int32_t text_length = 53;
  constexpr std::size_t element_count =
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(text_length);
  ManagedBuffer<__nv_bfloat16> alignment_0{element_count};
  ManagedBuffer<__nv_bfloat16> alignment_1{element_count};
  ManagedBuffer<__nv_bfloat16> alignment_2{element_count};
  ManagedBuffer<__nv_bfloat16> alignment_3{element_count};
  ManagedBuffer<__nv_bfloat16> output{element_count};
  for (std::size_t index = 0; index < element_count; ++index) {
    const float base =
        static_cast<float>(static_cast<std::int32_t>(index % 29U) - 14) /
        128.0F;
    alignment_0[index] = __float2bfloat16_rn(base);
    alignment_1[index] = __float2bfloat16_rn(base * 0.5F);
    alignment_2[index] = __float2bfloat16_rn(base * -0.25F);
    alignment_3[index] = __float2bfloat16_rn(base * 0.125F);
  }
  fill_bf16(output, -9.0F);
  require(
      magpie_tts_rt::plugins::launch_main_alignment_mean(
          alignment_0.data(),
          alignment_1.data(),
          alignment_2.data(),
          alignment_3.data(),
          output.data(),
          text_length,
          nullptr) == 0,
      "Main alignment mean launch failed");
  require_cuda(
      cudaDeviceSynchronize(),
      "Main alignment mean synchronize");
  for (std::size_t index = 0; index < element_count; ++index) {
    float sum = __bfloat162float(alignment_0[index]);
    sum += __bfloat162float(alignment_1[index]);
    sum += __bfloat162float(alignment_2[index]);
    sum += __bfloat162float(alignment_3[index]);
    const __nv_bfloat16 expected = __float2bfloat16_rn(sum * 0.25F);
    require(
        __bfloat16_as_ushort(output[index]) ==
            __bfloat16_as_ushort(expected),
        "Main alignment mean value mismatch");
  }
  require(
      magpie_tts_rt::plugins::launch_main_alignment_mean(
          nullptr,
          alignment_1.data(),
          alignment_2.data(),
          alignment_3.data(),
          output.data(),
          text_length,
          nullptr) == -1,
      "null Main alignment input was accepted");
}

struct EosBuffers final {
  ManagedBuffer<__nv_bfloat16> hidden{
      2U * static_cast<std::size_t>(kLocalArEmbeddingWidth)};
  ManagedBuffer<std::int64_t> codes{
      static_cast<std::size_t>(kLocalArCodebooks) *
      static_cast<std::size_t>(kLocalArFrames)};
  ManagedBuffer<bool> unfinished{1};
  ManagedBuffer<bool> finished{1};
  ManagedBuffer<bool> forbid_eos{1};
  DeviceBuffer<__nv_bfloat16> final_weight{
      static_cast<std::size_t>(kLocalArPositions) *
      static_cast<std::size_t>(kLocalArVocabularySize) *
      static_cast<std::size_t>(kLocalArEmbeddingWidth)};
  DeviceBuffer<__nv_bfloat16> final_bias{
      static_cast<std::size_t>(kLocalArPositions) *
      static_cast<std::size_t>(kLocalArVocabularySize)};
  DeviceBuffer<std::byte> workspace{
      magpie_tts_rt::plugins::local_ar_eos_workspace_size()};
  ManagedBuffer<std::int32_t> end_frame{1};
};

void reset_eos(EosBuffers& buffers) {
  std::fill(
      buffers.hidden.data(),
      buffers.hidden.data() + buffers.hidden.size(),
      __float2bfloat16_rn(0.0F));
  std::fill(buffers.codes.data(), buffers.codes.data() + buffers.codes.size(), 0);
  require_cuda(
      cudaMemset(
          buffers.final_weight.data(),
          0,
          buffers.final_weight.size() * sizeof(__nv_bfloat16)),
      "clear EOS weight");
  require_cuda(
      cudaMemset(
          buffers.final_bias.data(),
          0,
          buffers.final_bias.size() * sizeof(__nv_bfloat16)),
      "clear EOS bias");
  buffers.unfinished[0] = false;
  buffers.finished[0] = false;
  buffers.forbid_eos[0] = false;
  buffers.end_frame[0] = -2;
}

void run_eos(EosBuffers& buffers) {
  require(
      magpie_tts_rt::plugins::launch_local_ar_eos(
          buffers.hidden.data(),
          buffers.codes.data(),
          buffers.unfinished.data(),
          buffers.finished.data(),
          buffers.forbid_eos.data(),
          buffers.final_weight.data(),
          buffers.final_bias.data(),
          buffers.workspace.data(),
          buffers.end_frame.data(),
          nullptr) == 0,
      "EOS launch failed");
  require_cuda(cudaDeviceSynchronize(), "EOS synchronize");
}

void test_eos_contract() {
  EosBuffers buffers;
  reset_eos(buffers);
  run_eos(buffers);
  require(buffers.end_frame[0] == -1, "zero projection produced EOS");

  reset_eos(buffers);
  buffers.unfinished[0] = true;
  run_eos(buffers);
  require(buffers.end_frame[0] == -1, "unfinished row produced EOS");

  reset_eos(buffers);
  buffers.forbid_eos[0] = true;
  run_eos(buffers);
  require(buffers.end_frame[0] == -1, "forbidden EOS was emitted");

  reset_eos(buffers);
  buffers.finished[0] = true;
  run_eos(buffers);
  require(buffers.end_frame[0] == 0, "finished row did not force frame zero");

  reset_eos(buffers);
  buffers.codes[1] = kLocalArAudioEosId;
  run_eos(buffers);
  require(buffers.end_frame[0] == 1, "sampled EOS frame was not detected");

  reset_eos(buffers);
  const std::size_t eos_bias_index =
      static_cast<std::size_t>(kLocalArAudioEosId);
  const __nv_bfloat16 one = __float2bfloat16_rn(1.0F);
  require_cuda(
      cudaMemcpy(
          buffers.final_bias.data() + eos_bias_index,
          &one,
          sizeof(one),
          cudaMemcpyHostToDevice),
      "write EOS bias");
  run_eos(buffers);
  require(buffers.end_frame[0] == 0, "argmax EOS was not detected");
}

}  // namespace

int main() {
  try {
    test_sampling_contract();
    test_clamp_endpoints();
    test_oracle_math_plugins();
    test_main_cross_attention_softmax();
    test_main_self_attention_context();
    test_main_self_attention_step_context();
    test_main_cross_attention_context();
    test_main_self_attention_step_scores();
    test_main_attention_prior_normalization();
    test_main_alignment_mean();
    test_eos_contract();
  } catch (const std::exception& error) {
    std::cerr << "Local AR plugin test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

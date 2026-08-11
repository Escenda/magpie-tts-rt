#include "local_ar_plugins.hpp"
#include "magpie_tts_rt/magpie_tts_rt_plugin.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

struct MainDevicePositionBuffers final {
  static constexpr std::int32_t head_width =
      kMainDecoderCrossAttentionWidth / 2;
  static constexpr std::int32_t batch_heads =
      kMainDecoderBatch * kAttentionHeads;
  static constexpr std::size_t cache_elements =
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(kMainDecoderCacheCapacity) *
      static_cast<std::size_t>(head_width);
  static constexpr std::size_t output_elements =
      static_cast<std::size_t>(batch_heads) *
      static_cast<std::size_t>(head_width);
  static constexpr std::size_t mask_elements =
      static_cast<std::size_t>(kMainDecoderBatch) *
      static_cast<std::size_t>(kMainDecoderCacheCapacity);

  ManagedBuffer<__nv_bfloat16> query{output_elements};
  ManagedBuffer<__nv_bfloat16> key{cache_elements};
  ManagedBuffer<__nv_bfloat16> value{cache_elements};
  ManagedBuffer<bool> mask_a{mask_elements};
  ManagedBuffer<bool> mask_b{mask_elements};
  ManagedBuffer<std::int64_t> position_a{1};
  ManagedBuffer<std::int64_t> position_b{1};
  ManagedBuffer<std::int32_t> execution_status_input{1};
  ManagedBuffer<std::int32_t> execution_status_output{1};
  DeviceBuffer<std::byte> bank_workspace{
      magpie_tts_rt::plugins::
          main_self_attention_device_position_workspace_size()};
  DeviceBuffer<std::byte> reference_workspace{
      magpie_tts_rt::plugins::
          main_self_attention_device_position_workspace_size()};
  ManagedBuffer<__nv_bfloat16> bank_output{output_elements};
  ManagedBuffer<__nv_bfloat16> reference_output{output_elements};
};

[[nodiscard]] constexpr std::int32_t main_device_position_status(
    const std::uint32_t category,
    const std::uint32_t layer_index,
    const std::uint32_t operation,
    const std::uint32_t detail) noexcept {
  return static_cast<std::int32_t>(
      (category << MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_SHIFT) |
      (layer_index << MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_SHIFT) |
      (operation << MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SHIFT) |
      (detail & MTT_MAIN_DEVICE_POSITION_STATUS_DETAIL_MASK));
}

void initialize_main_device_position_buffers(
    MainDevicePositionBuffers& buffers,
    const std::int32_t salt) {
  for (std::size_t index = 0; index < buffers.query.size(); ++index) {
    buffers.query[index] = __float2bfloat16_rn(
        static_cast<float>(
            static_cast<std::int32_t>((index * 17U + salt) % 41U) - 20) /
        32.0F);
  }
  for (std::size_t index = 0; index < buffers.key.size(); ++index) {
    buffers.key[index] = __float2bfloat16_rn(
        static_cast<float>(
            static_cast<std::int32_t>((index * 13U + salt) % 37U) - 18) /
        64.0F);
    buffers.value[index] = __float2bfloat16_rn(
        static_cast<float>(
            static_cast<std::int32_t>((index * 19U + salt) % 43U) - 21) /
        64.0F);
  }
}

void set_main_device_position_case(
    MainDevicePositionBuffers& buffers,
    const std::int32_t position,
    const bool use_second_inputs) {
  ManagedBuffer<bool>& mask =
      use_second_inputs ? buffers.mask_b : buffers.mask_a;
  std::fill(mask.data(), mask.data() + mask.size(), false);
  const std::int32_t active_k = position + 1;
  for (std::int32_t batch = 0; batch < kMainDecoderBatch; ++batch) {
    for (std::int32_t cache = 0; cache < active_k; ++cache) {
      const bool hole =
          use_second_inputs && cache > 0 &&
          ((cache + batch * 3) % 29 == 0);
      mask[
          static_cast<std::size_t>(batch) *
              static_cast<std::size_t>(kMainDecoderCacheCapacity) +
          static_cast<std::size_t>(cache)] = !hole;
    }
  }
  mask[0] = true;
  mask[static_cast<std::size_t>(kMainDecoderCacheCapacity)] = true;
  ManagedBuffer<std::int64_t>& position_buffer =
      use_second_inputs ? buffers.position_b : buffers.position_a;
  position_buffer[0] = position;
}

[[nodiscard]] const bool* selected_main_device_position_mask(
    const MainDevicePositionBuffers& buffers,
    const bool use_second_inputs) {
  return use_second_inputs ? buffers.mask_b.data() : buffers.mask_a.data();
}

[[nodiscard]] const std::int64_t* selected_main_device_position(
    const MainDevicePositionBuffers& buffers,
    const bool use_second_inputs) {
  return use_second_inputs ? buffers.position_b.data()
                           : buffers.position_a.data();
}

void compare_main_device_position_output(
    const MainDevicePositionBuffers& buffers,
    const std::string& stage) {
  for (std::size_t index = 0; index < buffers.bank_output.size(); ++index) {
    require(
        __bfloat16_as_ushort(buffers.bank_output[index]) ==
            __bfloat16_as_ushort(buffers.reference_output[index]),
        stage + " output mismatch at " + std::to_string(index));
  }
}

void run_main_device_position_reference(
    MainDevicePositionBuffers& buffers,
    const bool use_second_inputs,
    cudaStream_t stream) {
  require(
      magpie_tts_rt::plugins::launch_main_self_attention_device_position(
          buffers.query.data(),
          buffers.key.data(),
          buffers.value.data(),
          selected_main_device_position_mask(buffers, use_second_inputs),
          selected_main_device_position(buffers, use_second_inputs),
          buffers.reference_workspace.data(),
          buffers.reference_output.data(),
          stream) == 0,
      "Main device-position reference launch failed");
}

void run_main_device_position_standalone_case(
    magpie_tts_rt::plugins::MainDevicePositionBankTestState* state,
    MainDevicePositionBuffers& buffers,
    const std::int32_t position,
    const bool use_second_inputs,
    cudaStream_t stream) {
  set_main_device_position_case(buffers, position, use_second_inputs);
  buffers.execution_status_input[0] = 0;
  buffers.execution_status_output[0] = -1;
  fill_bf16(buffers.bank_output, -9.0F);
  fill_bf16(buffers.reference_output, 9.0F);
  require(
      magpie_tts_rt::plugins::enqueue_main_device_position_bank_test_state(
          state,
          buffers.query.data(),
          buffers.key.data(),
          buffers.value.data(),
          selected_main_device_position_mask(buffers, use_second_inputs),
          selected_main_device_position(buffers, use_second_inputs),
          buffers.execution_status_input.data(),
          buffers.execution_status_output.data(),
          buffers.bank_workspace.data(),
          buffers.bank_output.data(),
          stream) == 0,
      "Main device-position standalone enqueue failed");
  run_main_device_position_reference(buffers, use_second_inputs, stream);
  require_cuda(
      cudaStreamSynchronize(stream),
      "Main device-position standalone synchronize");
  require(
      buffers.execution_status_output[0] == 0,
      "Main device-position standalone returned non-zero status");
  compare_main_device_position_output(
      buffers, "Main device-position standalone");
}

cudaGraphExec_t capture_main_device_position_outer_graph(
    magpie_tts_rt::plugins::MainDevicePositionBankTestState* state,
    MainDevicePositionBuffers& buffers,
    const bool use_second_inputs,
    cudaStream_t stream) {
  buffers.execution_status_input[0] = 0;
  buffers.execution_status_output[0] = -1;
  require_cuda(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin Main device-position outer capture");
  require(
      magpie_tts_rt::plugins::enqueue_main_device_position_bank_test_state(
          state,
          buffers.query.data(),
          buffers.key.data(),
          buffers.value.data(),
          selected_main_device_position_mask(buffers, use_second_inputs),
          selected_main_device_position(buffers, use_second_inputs),
          buffers.execution_status_input.data(),
          buffers.execution_status_output.data(),
          buffers.bank_workspace.data(),
          buffers.bank_output.data(),
          stream) == 0,
      "inject Main device-position bank into outer capture");
  cudaGraph_t graph = nullptr;
  require_cuda(
      cudaStreamEndCapture(stream, &graph),
      "end Main device-position outer capture");
  require(graph != nullptr, "Main device-position source graph is null");
  cudaGraphExec_t executable = nullptr;
  require_cuda(
      cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
      "instantiate Main device-position outer graph");
  require_cuda(
      cudaGraphDestroy(graph),
      "destroy Main device-position source graph");
  require(executable != nullptr, "Main device-position executable is null");
  return executable;
}

void run_main_device_position_outer_case(
    const cudaGraphExec_t executable,
    MainDevicePositionBuffers& buffers,
    const std::int32_t position,
    const bool use_second_inputs,
    cudaStream_t stream) {
  set_main_device_position_case(buffers, position, use_second_inputs);
  buffers.execution_status_input[0] = 0;
  buffers.execution_status_output[0] = -1;
  fill_bf16(buffers.bank_output, -9.0F);
  fill_bf16(buffers.reference_output, 9.0F);
  require_cuda(
      cudaGraphLaunch(executable, stream),
      "launch Main device-position outer graph");
  run_main_device_position_reference(buffers, use_second_inputs, stream);
  require_cuda(
      cudaStreamSynchronize(stream),
      "Main device-position outer synchronize");
  require(
      buffers.execution_status_output[0] == 0,
      "Main device-position outer returned non-zero status");
  compare_main_device_position_output(buffers, "Main device-position outer");
}

void test_main_device_position_production_bank() {
  mtt_main_device_position_class_table_v1_t before{};
  before.struct_size = sizeof(before);
  before.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  require(
      mtt_plugin_get_main_device_position_class_table_v1(&before) ==
          MTT_PLUGIN_STATUS_NOT_READY,
      "Main device-position class table was ready before bank discovery");

  cudaStream_t stream = nullptr;
  require_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create Main device-position test stream");
  MainDevicePositionBuffers first;
  MainDevicePositionBuffers second;
  initialize_main_device_position_buffers(first, 3);
  initialize_main_device_position_buffers(second, 11);
  auto* first_state =
      magpie_tts_rt::plugins::create_main_device_position_bank_test_state(0);
  auto* second_state =
      magpie_tts_rt::plugins::create_main_device_position_bank_test_state(7);
  require(first_state != nullptr && second_state != nullptr,
          "create Main device-position test state");

  constexpr std::array<std::int32_t, 6> positions{
      218, 219, 255, 256, 300, 466};
  for (std::size_t index = 0; index < positions.size(); ++index) {
    run_main_device_position_standalone_case(
        first_state,
        first,
        positions[index],
        index % 2U != 0U,
        stream);
  }

  mtt_main_device_position_class_table_v1_t first_table{};
  first_table.struct_size = sizeof(first_table);
  first_table.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  require(
      mtt_plugin_get_main_device_position_class_table_v1(&first_table) ==
          MTT_PLUGIN_STATUS_OK,
      "Main device-position class table unavailable after discovery");
  require(
      first_table.class_count ==
          MTT_MAIN_DEVICE_POSITION_CLASS_COUNT_V1 &&
          first_table.k_count == MTT_MAIN_DEVICE_POSITION_K_COUNT_V1,
      "Main device-position class table counts mismatch");
  for (std::size_t index = 0; index < first_table.class_count; ++index) {
    require(
        first_table.classes[index].function_name[0] != '\0',
        "Main device-position class has no stable function name");
  }

  run_main_device_position_standalone_case(
      second_state, second, 300, true, stream);
  mtt_main_device_position_class_table_v1_t second_table{};
  second_table.struct_size = sizeof(second_table);
  second_table.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  require(
      mtt_plugin_get_main_device_position_class_table_v1(&second_table) ==
          MTT_PLUGIN_STATUS_OK &&
          std::memcmp(&first_table, &second_table, sizeof(first_table)) == 0,
      "independent Main device-position state changed the class table");

  set_main_device_position_case(first, 218, false);
  cudaGraphExec_t outer = capture_main_device_position_outer_graph(
      first_state, first, false, stream);
  for (const std::int32_t position : positions) {
    run_main_device_position_outer_case(
        outer, first, position, false, stream);
  }
  require_cuda(
      cudaGraphExecDestroy(outer),
      "destroy first Main device-position outer executable");

  set_main_device_position_case(first, 218, true);
  outer = capture_main_device_position_outer_graph(
      first_state, first, true, stream);
  run_main_device_position_outer_case(
      outer, first, 466, true, stream);
  require_cuda(
      cudaGraphExecDestroy(outer),
      "destroy recaptured Main device-position outer executable");

  magpie_tts_rt::plugins::destroy_main_device_position_bank_test_state(
      second_state);

  // A prior layer's first error is immutable. Every variant is disabled and
  // the context output remains untouched instead of doing speculative work.
  constexpr std::int32_t sticky_status = main_device_position_status(
      MTT_MAIN_DEVICE_POSITION_STATUS_CUDA_GRAPH_UPDATE,
      5,
      MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_PV,
      static_cast<std::uint32_t>(cudaErrorInvalidValue));
  set_main_device_position_case(first, 300, false);
  first.execution_status_input[0] = sticky_status;
  first.execution_status_output[0] = -1;
  fill_bf16(first.bank_output, -9.0F);
  require(
      magpie_tts_rt::plugins::enqueue_main_device_position_bank_test_state(
          first_state,
          first.query.data(),
          first.key.data(),
          first.value.data(),
          first.mask_a.data(),
          first.position_a.data(),
          first.execution_status_input.data(),
          first.execution_status_output.data(),
          first.bank_workspace.data(),
          first.bank_output.data(),
          stream) == 0,
      "sticky status enqueue failed");
  require_cuda(
      cudaStreamSynchronize(stream),
      "sticky Main device-position status synchronize");
  require(
      first.execution_status_output[0] == sticky_status,
      "later layer overwrote the first execution error");
  for (std::size_t index = 0; index < first.bank_output.size(); ++index) {
    require(
        __bfloat16_as_ushort(first.bank_output[index]) ==
            __bfloat16_as_ushort(__float2bfloat16_rn(-9.0F)),
        "sticky execution error admitted a context output");
  }

  // Invalid K disables every selectable cuBLAS node and reports a typed
  // device status. The stream remains usable so the existing batch boundary
  // can copy and diagnose the status before codec/publication.
  set_main_device_position_case(
      first, kMainDecoderPrefillLength - 1, false);
  first.execution_status_input[0] = 0;
  first.execution_status_output[0] = -1;
  fill_bf16(first.bank_output, -9.0F);
  require(
      magpie_tts_rt::plugins::enqueue_main_device_position_bank_test_state(
          first_state,
          first.query.data(),
          first.key.data(),
          first.value.data(),
          first.mask_a.data(),
          first.position_a.data(),
          first.execution_status_input.data(),
          first.execution_status_output.data(),
          first.bank_workspace.data(),
          first.bank_output.data(),
          stream) == 0,
      "invalid selector enqueue failed on the host");
  require_cuda(
      cudaStreamSynchronize(stream),
      "invalid selector status synchronize");
  require(
      first.execution_status_output[0] == main_device_position_status(
          MTT_MAIN_DEVICE_POSITION_STATUS_INVALID_K,
          0,
          MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SELECTOR,
          0),
      "invalid selector did not produce its canonical status");

  // Corrupt one captured node reference to force a device graph-update API
  // failure. The updater still attempts to disable every remaining variant
  // and reports the exact CUDA status category/operation without trapping.
  require(
      magpie_tts_rt::plugins::
          invalidate_main_device_position_qk_node_test_state(first_state, 0),
      "failed to inject device graph-update fault");
  set_main_device_position_case(first, 300, false);
  first.execution_status_input[0] = 0;
  first.execution_status_output[0] = -1;
  require(
      magpie_tts_rt::plugins::enqueue_main_device_position_bank_test_state(
          first_state,
          first.query.data(),
          first.key.data(),
          first.value.data(),
          first.mask_a.data(),
          first.position_a.data(),
          first.execution_status_input.data(),
          first.execution_status_output.data(),
          first.bank_workspace.data(),
          first.bank_output.data(),
          stream) == 0,
      "device graph-update fault enqueue failed on the host");
  require_cuda(
      cudaStreamSynchronize(stream),
      "device graph-update fault synchronize");
  const std::uint32_t update_status = static_cast<std::uint32_t>(
      first.execution_status_output[0]);
  require(
      ((update_status >> MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_SHIFT) &
       MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_MASK) ==
              MTT_MAIN_DEVICE_POSITION_STATUS_CUDA_GRAPH_UPDATE &&
          ((update_status >> MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_SHIFT) &
           MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_MASK) == 0 &&
          ((update_status >>
            MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SHIFT) &
           MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_MASK) ==
              MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_QK &&
          (update_status & MTT_MAIN_DEVICE_POSITION_STATUS_DETAIL_MASK) != 0,
      "device graph-update failure lost its layer/operation/CUDA status");
  magpie_tts_rt::plugins::destroy_main_device_position_bank_test_state(
      first_state);
  require_cuda(
      cudaStreamDestroy(stream),
      "destroy Main device-position test stream");
}

void test_main_device_position_scalar_callback_shapes() {
  require(
      magpie_tts_rt::plugins::
          validate_main_device_position_scalar_callback_shapes_test(),
      "Main device-position callback scalar lowering contract failed");
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
    test_main_cross_attention_context();
    test_main_attention_prior_normalization();
    test_main_alignment_mean();
    test_eos_contract();
    test_main_device_position_scalar_callback_shapes();
    test_main_device_position_production_bank();
  } catch (const std::exception& error) {
    std::cerr << "Local AR plugin test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

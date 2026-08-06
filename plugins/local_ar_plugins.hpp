#pragma once

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace magpie_tts_rt::plugins {

inline constexpr std::int32_t kLocalArVocabularySize = 2024;
inline constexpr std::int32_t kLocalArCodebookSize = 2016;
inline constexpr std::int32_t kLocalArAudioEosId = 2017;
inline constexpr std::int32_t kLocalArEmbeddingWidth = 768;
inline constexpr std::int32_t kLocalArPositions = 16;
inline constexpr std::int32_t kLocalArCodebooks = 8;
inline constexpr std::int32_t kLocalArFrames = 2;
inline constexpr std::int32_t kLocalArFeedForwardWidth = 3072;
inline constexpr std::int32_t kAttentionHeads = 12;
inline constexpr std::int32_t kMaximumTextSequenceLength = 512;
inline constexpr std::int32_t kMainDecoderBatch = 2;
inline constexpr std::int32_t kMainDecoderPrefillLength = 218;
inline constexpr std::int32_t kMainDecoderCacheCapacity = 467;
inline constexpr std::int32_t kMainDecoderCrossAttentionHeads = 1;
inline constexpr std::int32_t kMainDecoderCrossAttentionWidth = 128;
inline constexpr std::int32_t kMaximumNormalizationRows =
    kMainDecoderBatch * kMaximumTextSequenceLength;

[[nodiscard]] int launch_local_ar_sampling(
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
    cudaStream_t stream) noexcept;

[[nodiscard]] std::size_t local_ar_eos_workspace_size() noexcept;

[[nodiscard]] int launch_local_ar_eos(
    const void* decoder_hidden_bf16,
    const std::int64_t* codec_tokens,
    const bool* unfinished,
    const bool* finished,
    const bool* forbid_eos,
    const void* final_weight_bf16,
    const void* final_bias_bf16,
    void* workspace,
    std::int32_t* end_frame_index,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_layer_norm(
    const void* input_bf16,
    const void* weight_bf16,
    void* output_bf16,
    std::int32_t row_count,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_gelu_tanh(
    const void* input_bf16,
    void* output_bf16,
    std::size_t element_count,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_softmax(
    const void* input_bf16,
    void* output_bf16,
    std::int32_t batch_count,
    std::int32_t element_count,
    cudaStream_t stream) noexcept;

[[nodiscard]] std::size_t
main_cross_attention_softmax_workspace_size() noexcept;

[[nodiscard]] int launch_main_cross_attention_softmax(
    cublasHandle_t cublas_handle,
    const void* query_bf16,
    const void* key_bf16,
    const bool* memory_mask,
    void* workspace,
    void* output_bf16,
    std::int32_t query_length,
    std::int32_t memory_length,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_main_self_attention_context(
    const void* probabilities_bf16,
    const void* value_bf16,
    void* output_bf16,
    std::int32_t query_length,
    std::int32_t key_length,
    cudaStream_t stream) noexcept;

[[nodiscard]] std::size_t
main_self_attention_step_context_workspace_size() noexcept;

[[nodiscard]] int launch_main_self_attention_step_context(
    cublasHandle_t cublas_handle,
    const void* probabilities_bf16,
    const void* value_bf16,
    void* workspace,
    void* output_bf16,
    std::int32_t active_length,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_main_self_attention_step_scores(
    cublasHandle_t cublas_handle,
    const void* query_bf16,
    const void* key_transposed_bf16,
    void* output_bf16,
    std::int32_t active_length,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_main_cross_attention_context(
    cublasHandle_t cublas_handle,
    const void* probabilities_bf16,
    const void* value_bf16,
    void* output_bf16,
    std::int32_t query_length,
    std::int32_t memory_length,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_main_attention_prior_normalization(
    const void* probabilities_bf16,
    const void* attention_prior_bf16,
    void* output_bf16,
    std::int32_t memory_length,
    cudaStream_t stream) noexcept;

[[nodiscard]] int launch_main_alignment_mean(
    const void* alignment_0_bf16,
    const void* alignment_1_bf16,
    const void* alignment_2_bf16,
    const void* alignment_3_bf16,
    void* output_bf16,
    std::int32_t text_length,
    cudaStream_t stream) noexcept;

#if defined(MAGPIE_TTS_RT_PLUGIN_TESTING)
[[nodiscard]] int launch_test_clamped_gumbel(
    const float* uniform,
    float* gumbel,
    std::size_t count,
    cudaStream_t stream) noexcept;
#endif

}  // namespace magpie_tts_rt::plugins

#include "runtime/synthesis_kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace magpie_tts_rt {
namespace {

constexpr std::uint32_t kModelWidth = 768;
constexpr std::uint32_t kCodebooks = 8;
constexpr std::uint32_t kFramesPerDecoderStep = 2;
constexpr std::uint32_t kSteadyFrames = 8;
constexpr std::uint32_t kThreads = 256;

__global__ void advance_decoder_position_kernel(
    std::int64_t* decoder_position) {
  ++decoder_position[0];
}

__global__ void prepare_cfg_inputs_kernel(
    const __nv_bfloat16* text_condition,
    const bool* text_mask,
    const std::uint32_t text_token_count,
    __nv_bfloat16* cfg_condition,
    bool* condition_mask) {
  const std::uint64_t condition_elements =
      static_cast<std::uint64_t>(text_token_count) * kModelWidth;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < condition_elements * 2U;
       index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    cfg_condition[index] =
        index < condition_elements
            ? text_condition[index]
            : __float2bfloat16_rn(0.0F);
  }
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < text_token_count * 2U;
       index += static_cast<std::uint32_t>(gridDim.x) * blockDim.x) {
    if (index < text_token_count) {
      condition_mask[index] = text_mask[index];
    } else {
      condition_mask[index] = index == text_token_count;
    }
  }
}

__global__ void finalize_local_step_kernel(
    const std::int64_t* step_codec_tokens,
    std::int64_t* aggregate_codec_tokens,
    const std::uint32_t frame_offset,
    const std::int32_t* canonical_invalid_rows,
    const std::int32_t* canonical_end_frame_index,
    std::int32_t* step_invalid_rows,
    std::int32_t* step_end_frame_index,
    const std::int64_t* updated_rng_counter,
    std::int64_t* rng_counter,
    bool* generation_finished) {
  const std::uint32_t index =
      static_cast<std::uint32_t>(threadIdx.x);
  if (index < kCodebooks * kFramesPerDecoderStep) {
    const std::uint32_t codebook =
        index / kFramesPerDecoderStep;
    const std::uint32_t frame =
        index % kFramesPerDecoderStep;
    aggregate_codec_tokens[
        codebook * kSteadyFrames + frame_offset + frame] =
        step_codec_tokens[index];
  }
  if (index == 0) {
    const std::int32_t invalid_rows = canonical_invalid_rows[0];
    const std::int32_t end_frame_index =
        canonical_end_frame_index[0];
    step_invalid_rows[0] = invalid_rows;
    step_end_frame_index[0] = end_frame_index;
    rng_counter[0] = updated_rng_counter[0];
    if (end_frame_index == 0 || end_frame_index == 1) {
      generation_finished[0] = true;
    }
  }
}

__global__ void pack_codec_frames_kernel(
    const std::int64_t* aggregate_codec_tokens,
    std::int64_t* packed_codec_tokens,
    const std::uint32_t frame_count) {
  const std::uint32_t index =
      static_cast<std::uint32_t>(threadIdx.x);
  const std::uint32_t elements = kCodebooks * frame_count;
  if (index >= elements) {
    return;
  }
  const std::uint32_t codebook = index / frame_count;
  const std::uint32_t frame = index % frame_count;
  packed_codec_tokens[index] =
      aggregate_codec_tokens[
          codebook * kSteadyFrames + frame];
}

__global__ void pack_generation_diagnostics_kernel(
    const GenerationDiagnosticSources sources,
    const std::uint32_t step_count,
    GenerationBatchDiagnostics* const output) {
  const std::uint32_t step =
      static_cast<std::uint32_t>(threadIdx.x);
  if (step == 0U) {
    output->step_count = step_count;
    output->main_decoder_execution_status =
        sources.main_decoder_execution_status[0];
  }
  if (step >= kMaximumDecoderStepsPerEmission) {
    return;
  }
  GenerationStepDiagnostics& destination = output->steps[step];
  if (step < step_count) {
    destination.attended_token_index =
        sources.attended_token_indices[step][0];
    destination.alignment_invalid =
        sources.alignment_invalid_steps[step][0];
    destination.local_invalid_rows =
        sources.local_invalid_rows[step][0];
    destination.end_frame_index =
        sources.end_frame_indices[step][0];
  } else {
    destination.attended_token_index = -1;
    destination.alignment_invalid = 0;
    destination.local_invalid_rows = 0;
    destination.end_frame_index = -1;
  }
  destination.reserved = 0;
}

}  // namespace

cudaError_t launch_advance_decoder_position(
    std::int64_t* const decoder_position,
    const cudaStream_t stream) noexcept {
  if (decoder_position == nullptr) {
    return cudaErrorInvalidValue;
  }
  advance_decoder_position_kernel<<<1, 1, 0, stream>>>(
      decoder_position);
  return cudaPeekAtLastError();
}

cudaError_t launch_prepare_cfg_inputs(
    const void* text_condition_bf16,
    const bool* text_mask,
    const std::uint32_t text_token_count,
    void* cfg_condition_bf16,
    bool* condition_mask,
    const cudaStream_t stream) noexcept {
  if (text_condition_bf16 == nullptr || text_mask == nullptr ||
      text_token_count == 0 || text_token_count > 512 ||
      cfg_condition_bf16 == nullptr || condition_mask == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t elements =
      static_cast<std::uint64_t>(text_token_count) *
      kModelWidth * 2U;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      (elements + kThreads - 1U) / kThreads);
  prepare_cfg_inputs_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(text_condition_bf16),
      text_mask,
      text_token_count,
      static_cast<__nv_bfloat16*>(cfg_condition_bf16),
      condition_mask);
  return cudaPeekAtLastError();
}

cudaError_t launch_finalize_local_step(
    const std::int64_t* step_codec_tokens,
    std::int64_t* aggregate_codec_tokens,
    const std::uint32_t frame_offset,
    const std::int32_t* canonical_invalid_rows,
    const std::int32_t* canonical_end_frame_index,
    std::int32_t* step_invalid_rows,
    std::int32_t* step_end_frame_index,
    const std::int64_t* updated_rng_counter,
    std::int64_t* rng_counter,
    bool* generation_finished,
    const cudaStream_t stream) noexcept {
  if (step_codec_tokens == nullptr ||
      aggregate_codec_tokens == nullptr ||
      frame_offset > kSteadyFrames - kFramesPerDecoderStep ||
      frame_offset % kFramesPerDecoderStep != 0 ||
      canonical_invalid_rows == nullptr ||
      canonical_end_frame_index == nullptr ||
      step_invalid_rows == nullptr ||
      step_end_frame_index == nullptr ||
      updated_rng_counter == nullptr || rng_counter == nullptr ||
      generation_finished == nullptr) {
    return cudaErrorInvalidValue;
  }
  finalize_local_step_kernel<<<
      1, kCodebooks * kFramesPerDecoderStep, 0, stream>>>(
      step_codec_tokens,
      aggregate_codec_tokens,
      frame_offset,
      canonical_invalid_rows,
      canonical_end_frame_index,
      step_invalid_rows,
      step_end_frame_index,
      updated_rng_counter,
      rng_counter,
      generation_finished);
  return cudaPeekAtLastError();
}

cudaError_t launch_pack_codec_frames(
    const std::int64_t* aggregate_codec_tokens,
    std::int64_t* packed_codec_tokens,
    const std::uint32_t frame_count,
    const cudaStream_t stream) noexcept {
  if (aggregate_codec_tokens == nullptr ||
      packed_codec_tokens == nullptr || frame_count == 0 ||
      frame_count > kSteadyFrames) {
    return cudaErrorInvalidValue;
  }
  pack_codec_frames_kernel<<<
      1, kCodebooks * frame_count, 0, stream>>>(
      aggregate_codec_tokens, packed_codec_tokens, frame_count);
  return cudaPeekAtLastError();
}

cudaError_t launch_pack_generation_diagnostics(
    const GenerationDiagnosticSources& sources,
    const std::uint32_t step_count,
    GenerationBatchDiagnostics* const output,
    const cudaStream_t stream) noexcept {
  if (validate_generation_diagnostic_pack_arguments(
          sources, step_count, output) !=
      GenerationDiagnosticPackArgumentError::none) {
    return cudaErrorInvalidValue;
  }
  pack_generation_diagnostics_kernel<<<
      1, kMaximumDecoderStepsPerEmission, 0, stream>>>(
      sources, step_count, output);
  return cudaPeekAtLastError();
}

}  // namespace magpie_tts_rt

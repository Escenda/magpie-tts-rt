#include "runtime/alignment_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace magpie_tts_rt {
namespace {

constexpr std::uint32_t kMaximumTextTokens = 512;
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaximumUint32 = 0xFFFFFFFFU;

__global__ void alignment_controller_kernel(
    const __nv_bfloat16* alignment_scores,
    const AlignmentKernelContract contract,
    std::uint32_t* counters,
    std::uint32_t* last_attended,
    __nv_bfloat16* next_prior,
    std::int64_t* attended_token_index,
    bool* unfinished_text,
    std::int32_t* invalid_state) {
  __shared__ std::uint32_t attended;
  __shared__ std::int32_t maximum_sink_position;
  __shared__ std::int32_t invalid;

  if (threadIdx.x == 0) {
    invalid = 0;
    attended = 0;
    maximum_sink_position = -1;
    const std::uint32_t text_length = contract.text_length;
    if (text_length == 0 || text_length > kMaximumTextTokens ||
        text_length <= contract.ignored_terminal_tokens ||
        contract.lookahead == 0 || contract.sink_threshold == 0) {
      invalid = 1;
    } else {
      const std::uint32_t bounded_last =
          min(*last_attended, text_length - 1U);
      std::uint32_t search_start = *last_attended;
      if (counters[bounded_last] >= contract.sink_threshold) {
        if (search_start == kMaximumUint32) {
          invalid = 1;
        } else {
          ++search_start;
        }
      }
      search_start = min(search_start, text_length);
      const std::uint32_t content_end =
          text_length - contract.ignored_terminal_tokens;
      const std::uint64_t requested_window_end =
          static_cast<std::uint64_t>(search_start) +
          contract.lookahead;
      const std::uint32_t window_end = min(
          static_cast<std::uint32_t>(min(
              requested_window_end,
              static_cast<std::uint64_t>(kMaximumUint32))),
          content_end);

      attended = text_length - 1U;
      if (window_end > search_start) {
        attended = search_start;
        float best =
            __bfloat162float(alignment_scores[search_start]);
        if (!isfinite(best)) {
          invalid = 1;
        }
        for (std::uint32_t index = search_start + 1U;
             index < window_end;
             ++index) {
          const float candidate =
              __bfloat162float(alignment_scores[index]);
          if (!isfinite(candidate)) {
            invalid = 1;
          } else if (candidate > best) {
            attended = index;
            best = candidate;
          }
        }
      }
      // Reject corruption outside the active search window as well.
      for (std::uint32_t index = 0; index < text_length; ++index) {
        if (!isfinite(
                __bfloat162float(alignment_scores[index]))) {
          invalid = 1;
        }
      }

      if (counters[attended] == kMaximumUint32) {
        invalid = 1;
      } else {
        ++counters[attended];
      }
      *last_attended = attended;
      *attended_token_index =
          static_cast<std::int64_t>(attended);
      *unfinished_text = attended < content_end;
      for (std::uint32_t index = 0; index < text_length; ++index) {
        if (counters[index] >= contract.sink_threshold) {
          maximum_sink_position =
              static_cast<std::int32_t>(index);
        }
      }
    }
    *invalid_state = invalid;
  }
  __syncthreads();

  const std::uint32_t text_length = contract.text_length;
  if (invalid != 0 || text_length == 0 ||
      text_length > kMaximumTextTokens) {
    return;
  }
  const __nv_bfloat16 epsilon = __float2bfloat16_rn(0.1F);
  for (std::uint32_t index =
           static_cast<std::uint32_t>(threadIdx.x);
       index < text_length * 2U;
       index += static_cast<std::uint32_t>(blockDim.x)) {
    next_prior[index] = epsilon;
  }
  __syncthreads();

  if (static_cast<std::uint32_t>(threadIdx.x) < text_length) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(threadIdx.x);
    bool one =
        text_length <= contract.short_text_no_prior_max_tokens;
    if (!one) {
      const std::uint32_t history_floor =
          min(1U, text_length - 1U);
      const std::uint32_t history =
          max(attended == 0 ? 0U : attended - 1U, history_floor);
      one = index == history || index == attended;
      for (std::uint32_t offset = 1;
           !one && offset <= contract.lookahead;
           ++offset) {
        const std::uint64_t requested =
            static_cast<std::uint64_t>(attended) + offset;
        const std::uint32_t lookahead_index =
            static_cast<std::uint32_t>(min(
                requested,
                static_cast<std::uint64_t>(text_length - 1U)));
        one = index == lookahead_index;
      }
    }
    if (one &&
        (maximum_sink_position < 0 ||
         static_cast<std::int32_t>(index) >
             maximum_sink_position)) {
      next_prior[index] = __float2bfloat16_rn(1.0F);
    }
  }
}

}  // namespace

cudaError_t launch_alignment_controller(
    const void* alignment_scores_bf16,
    const AlignmentKernelContract& contract,
    std::uint32_t* counters,
    std::uint32_t* last_attended,
    void* next_prior_bf16,
    std::int64_t* attended_token_index,
    bool* unfinished_text,
    std::int32_t* invalid_state,
    const cudaStream_t stream) noexcept {
  if (alignment_scores_bf16 == nullptr || counters == nullptr ||
      last_attended == nullptr || next_prior_bf16 == nullptr ||
      attended_token_index == nullptr || unfinished_text == nullptr ||
      invalid_state == nullptr) {
    return cudaErrorInvalidValue;
  }
  alignment_controller_kernel<<<1, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(alignment_scores_bf16),
      contract,
      counters,
      last_attended,
      static_cast<__nv_bfloat16*>(next_prior_bf16),
      attended_token_index,
      unfinished_text,
      invalid_state);
  return cudaPeekAtLastError();
}

}  // namespace magpie_tts_rt

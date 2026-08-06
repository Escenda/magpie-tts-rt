#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace magpie_tts_rt {

struct AlignmentKernelContract {
  std::uint32_t text_length;
  std::uint32_t ignored_terminal_tokens;
  std::uint32_t short_text_no_prior_max_tokens;
  std::uint32_t lookahead;
  std::uint32_t sink_threshold;
};

// All pointers are device pointers. alignment_scores_bf16 is [2,T] and
// next_prior_bf16 is [2,1,T]. State and outputs are single-utterance scalars.
// Returns the CUDA launch status without synchronizing the stream.
[[nodiscard]] cudaError_t launch_alignment_controller(
    const void* alignment_scores_bf16,
    const AlignmentKernelContract& contract,
    std::uint32_t* counters,
    std::uint32_t* last_attended,
    void* next_prior_bf16,
    std::int64_t* attended_token_index,
    bool* unfinished_text,
    std::int32_t* invalid_state,
    cudaStream_t stream) noexcept;

}  // namespace magpie_tts_rt

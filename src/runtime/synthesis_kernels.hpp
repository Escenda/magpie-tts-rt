#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace magpie_tts_rt {

// Expands Text Encoder [1,T,768] BF16 output into the exact two-row CFG
// contract and constructs the conditional/unconditional masks. No host
// synchronization is performed.
[[nodiscard]] cudaError_t launch_prepare_cfg_inputs(
    const void* text_condition_bf16,
    const bool* text_mask,
    std::uint32_t text_token_count,
    void* cfg_condition_bf16,
    bool* condition_mask,
    cudaStream_t stream) noexcept;

// Copies one Local AR [1,8,2] result into the codebook-major
// [1,8,8] codec submission buffer at frame_offset.
[[nodiscard]] cudaError_t launch_append_codec_step(
    const std::int64_t* step_codec_tokens,
    std::int64_t* aggregate_codec_tokens,
    std::uint32_t frame_offset,
    cudaStream_t stream) noexcept;

// Packs the first frame_count columns from the internal [1,8,8] stride into
// the contiguous [1,8,F] layout required by every NanoCodec route.
[[nodiscard]] cudaError_t launch_pack_codec_frames(
    const std::int64_t* aggregate_codec_tokens,
    std::int64_t* packed_codec_tokens,
    std::uint32_t frame_count,
    cudaStream_t stream) noexcept;

// Marks the remaining speculative Local AR steps as finished after the first
// EOS result. The next Local AR invocation will therefore emit forced EOS
// tokens instead of consuming another meaningful random branch.
[[nodiscard]] cudaError_t launch_latch_generation_finished(
    const std::int32_t* end_frame_index,
    bool* generation_finished,
    cudaStream_t stream) noexcept;

}  // namespace magpie_tts_rt

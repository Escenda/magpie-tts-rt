#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "runtime/generation_diagnostics.hpp"

namespace magpie_tts_rt {

// Advances the request-owned absolute Main Decoder position entirely on the
// generation stream. The same launch precedes eager warmup enqueues and is
// captured as the first node of both recurrent CUDA Graph directions.
[[nodiscard]] cudaError_t launch_advance_decoder_position(
    std::int64_t* decoder_position,
    cudaStream_t stream) noexcept;

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

// Commits one graph-backed Local AR result. The engine always writes its
// diagnostics and RNG counter to fixed canonical addresses; this kernel copies
// those values to the logical step, advances the canonical RNG input, appends
// the [1,8,2] codec result, and latches EOS in one ordered operation.
[[nodiscard]] cudaError_t launch_finalize_local_step(
    const std::int64_t* step_codec_tokens,
    std::int64_t* aggregate_codec_tokens,
    std::uint32_t frame_offset,
    const std::int32_t* canonical_invalid_rows,
    const std::int32_t* canonical_end_frame_index,
    std::int32_t* step_invalid_rows,
    std::int32_t* step_end_frame_index,
    const std::int64_t* updated_rng_counter,
    std::int64_t* rng_counter,
    bool* generation_finished,
    cudaStream_t stream) noexcept;

// Packs all host-observed generation diagnostics into one fixed record so a
// generation batch performs exactly one device-to-host transfer.
[[nodiscard]] cudaError_t launch_pack_generation_diagnostics(
    const GenerationDiagnosticSources& sources,
    std::uint32_t step_count,
    GenerationBatchDiagnostics* output,
    cudaStream_t stream) noexcept;

// Packs the first frame_count columns from the internal [1,8,8] stride into
// the contiguous [1,8,F] layout required by every NanoCodec route.
[[nodiscard]] cudaError_t launch_pack_codec_frames(
    const std::int64_t* aggregate_codec_tokens,
    std::int64_t* packed_codec_tokens,
    std::uint32_t frame_count,
    cudaStream_t stream) noexcept;

}  // namespace magpie_tts_rt

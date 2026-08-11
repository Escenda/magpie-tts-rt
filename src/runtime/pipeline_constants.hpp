#pragma once

#include <cstddef>

namespace magpie_tts_rt {

inline constexpr std::size_t kGenerationBatchSlotCount = 2;
inline constexpr std::size_t kMaximumDecoderStepsPerEmission = 4;

}  // namespace magpie_tts_rt

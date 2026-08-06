#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace magpie_tts_rt {

inline constexpr std::uint32_t kCodecFramesPerDecoderStep = 2;

// Local AR's EOS index identifies the first non-audio frame. The indexed EOS
// frame itself must never be retained or sent to NanoCodec.
[[nodiscard]] constexpr std::uint32_t retained_codec_frames_before_eos(
    const std::size_t decoder_step_index,
    const std::int32_t eos_frame_index) {
  if (eos_frame_index != 0 && eos_frame_index != 1) {
    throw std::invalid_argument("EOS frame index must be zero or one");
  }
  constexpr std::size_t maximum_step =
      (std::numeric_limits<std::uint32_t>::max() - 1U) /
      kCodecFramesPerDecoderStep;
  if (decoder_step_index > maximum_step) {
    throw std::overflow_error("retained codec frame count overflows uint32");
  }
  return static_cast<std::uint32_t>(
      decoder_step_index * kCodecFramesPerDecoderStep +
      static_cast<std::size_t>(eos_frame_index));
}

}  // namespace magpie_tts_rt

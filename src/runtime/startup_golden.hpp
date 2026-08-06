#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bundle/bundle.hpp"
#include "manifest/manifest.hpp"

namespace magpie_tts_rt {

inline constexpr std::uint64_t kMaximumStartupGoldenFixtureBytes =
    1024U * 1024U;

enum class StartupGoldenErrorCode {
  invalid_fixture,
  invalid_capture,
  count_mismatch,
  hash_mismatch,
};

[[nodiscard]] std::string_view to_string(
    StartupGoldenErrorCode code) noexcept;

class StartupGoldenError final : public std::runtime_error {
 public:
  StartupGoldenError(
      StartupGoldenErrorCode code,
      std::string detail);

  [[nodiscard]] StartupGoldenErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  StartupGoldenErrorCode code_;
  std::string detail_;
};

struct StartupGoldenExpected {
  std::string decoder_tokens_sha256;
  std::string codec_codes_sha256;
  std::uint64_t codec_frame_count;
  std::string pcm_f32le_sha256;
  std::uint64_t pcm_sample_count;
};

struct StartupGoldenFixture {
  std::uint32_t schema_version;
  std::string fixture_id;
  std::vector<std::int32_t> prepared_token_ids;
  std::uint32_t seed;
  std::string tokenizer_identity_sha256;
  std::string oracle_lock_sha256;
  std::string normalized_text_sha256;
  std::string token_ids_sha256;
  std::string baked_context_sha256;
  StartupGoldenExpected expected;
};

struct StartupGoldenActual {
  std::string decoder_tokens_sha256;
  std::string codec_codes_sha256;
  std::uint64_t codec_frame_count;
  std::string pcm_f32le_sha256;
  std::uint64_t pcm_sample_count;
};

// Parses the exact schema-v1 document and cross-checks every value that is
// duplicated in the authenticated runtime manifest. Unknown and duplicate
// fields fail closed.
[[nodiscard]] StartupGoldenFixture parse_startup_golden_fixture(
    std::string_view json_text,
    const RuntimeBundleManifest& manifest);

// Reads only the verifier-retained sealed snapshot. The diagnostic canonical
// path is deliberately never reopened.
[[nodiscard]] StartupGoldenFixture load_startup_golden_fixture(
    const VerifiedRuntimeBundle& bundle);

// Captures the complete deterministic sequence used by the mandatory startup
// gate. Decoder tokens are reassembled as one codebook-major [1,C,F] tensor,
// while codec codes preserve the scheduled per-engine [1,C,chunk] byte order.
class StartupGoldenCapture final {
 public:
  StartupGoldenCapture(
      std::uint32_t codebooks_per_frame,
      std::uint32_t codec_hop_length_samples,
      std::uint32_t maximum_audio_frames);

  StartupGoldenCapture(const StartupGoldenCapture&) = delete;
  StartupGoldenCapture& operator=(const StartupGoldenCapture&) = delete;

  void record_chunk(
      std::span<const std::int64_t> codec_codes,
      std::uint32_t codec_frame_count,
      std::span<const float> pcm);

  [[nodiscard]] StartupGoldenActual finish() const;

 private:
  std::uint32_t codebooks_per_frame_;
  std::uint32_t codec_hop_length_samples_;
  std::uint32_t maximum_audio_frames_;
  std::uint64_t codec_frame_count_{0};
  std::vector<std::vector<std::int64_t>> decoder_codebooks_;
  std::vector<std::int64_t> scheduled_codec_codes_;
  std::vector<float> pcm_;
};

void require_startup_golden_match(
    const StartupGoldenFixture& expected,
    const StartupGoldenActual& actual);

}  // namespace magpie_tts_rt

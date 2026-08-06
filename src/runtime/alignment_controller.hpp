#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "manifest/manifest.hpp"

namespace magpie_tts_rt {

enum class AlignmentControllerErrorCode {
  invalid_text_length,
  invalid_score_count,
  non_finite_score,
  state_overflow,
};

class AlignmentControllerError final : public std::runtime_error {
 public:
  AlignmentControllerError(
      AlignmentControllerErrorCode code,
      std::string detail);

  [[nodiscard]] AlignmentControllerErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  AlignmentControllerErrorCode code_;
  std::string detail_;
};

struct AlignmentStepResult {
  std::uint32_t attended_token_index;
  bool unfinished_text;
  // Row-major [conditional, unconditional], each row shaped [1, T].
  std::vector<float> next_prior;
};

// Exact single-utterance CPU reference for the accepted Magpie dynamic
// attention-prior controller. The runtime CUDA implementation is validated
// against this class and the locked oracle; this class is not a silent CPU
// inference fallback.
class AlignmentControllerReference final {
 public:
  AlignmentControllerReference(
      std::uint32_t text_length,
      const AlignmentManifest& contract);

  [[nodiscard]] AlignmentStepResult advance(
      const std::vector<float>& conditional_alignment_scores);

  [[nodiscard]] std::uint32_t last_attended_token_index() const noexcept;
  [[nodiscard]] const std::vector<std::uint32_t>& counters() const noexcept;

 private:
  std::uint32_t text_length_;
  double prior_epsilon_;
  std::uint32_t ignored_terminal_tokens_;
  std::uint32_t short_text_no_prior_max_tokens_;
  std::uint32_t lookahead_;
  std::uint32_t sink_threshold_;
  std::uint32_t last_attended_;
  std::vector<std::uint32_t> counters_;
};

}  // namespace magpie_tts_rt

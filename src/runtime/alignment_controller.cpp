#include "runtime/alignment_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const AlignmentControllerErrorCode code,
    const std::string_view detail) {
  std::string code_name;
  switch (code) {
    case AlignmentControllerErrorCode::invalid_text_length:
      code_name = "invalid_text_length";
      break;
    case AlignmentControllerErrorCode::invalid_score_count:
      code_name = "invalid_score_count";
      break;
    case AlignmentControllerErrorCode::non_finite_score:
      code_name = "non_finite_score";
      break;
    case AlignmentControllerErrorCode::state_overflow:
      code_name = "state_overflow";
      break;
  }
  return "alignment controller failed [code=" + code_name +
         "]: " + std::string(detail);
}

[[noreturn]] void fail(
    const AlignmentControllerErrorCode code,
    const std::string& detail) {
  throw AlignmentControllerError(code, detail);
}

}  // namespace

AlignmentControllerError::AlignmentControllerError(
    const AlignmentControllerErrorCode code,
    std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

AlignmentControllerErrorCode AlignmentControllerError::code() const noexcept {
  return code_;
}

const std::string& AlignmentControllerError::detail() const noexcept {
  return detail_;
}

AlignmentControllerReference::AlignmentControllerReference(
    const std::uint32_t text_length,
    const AlignmentManifest& contract)
    : text_length_(text_length),
      prior_epsilon_(contract.prior_epsilon),
      ignored_terminal_tokens_(contract.ignored_terminal_tokens),
      short_text_no_prior_max_tokens_(
          contract.short_text_no_prior_max_tokens),
      lookahead_(contract.lookahead),
      sink_threshold_(contract.sink_threshold),
      last_attended_(contract.initial_attended),
      counters_(text_length, 0) {
  if (text_length_ == 0 ||
      text_length_ <= ignored_terminal_tokens_ ||
      last_attended_ >= text_length_) {
    fail(
        AlignmentControllerErrorCode::invalid_text_length,
        "text must contain the initial attended token and all ignored terminal tokens");
  }
}

AlignmentStepResult AlignmentControllerReference::advance(
    const std::vector<float>& conditional_alignment_scores) {
  if (conditional_alignment_scores.size() != text_length_) {
    fail(
        AlignmentControllerErrorCode::invalid_score_count,
        "alignment score length differs from the prepared token length");
  }
  for (const float score : conditional_alignment_scores) {
    if (!std::isfinite(score)) {
      fail(
          AlignmentControllerErrorCode::non_finite_score,
          "alignment scores must all be finite");
    }
  }

  const std::uint32_t last =
      std::min(last_attended_, text_length_ - 1U);
  std::uint32_t search_start = last_attended_;
  if (counters_.at(last) >= sink_threshold_) {
    if (search_start == std::numeric_limits<std::uint32_t>::max()) {
      fail(
          AlignmentControllerErrorCode::state_overflow,
          "alignment search position overflowed");
    }
    ++search_start;
  }
  search_start = std::min(search_start, text_length_);
  const std::uint32_t content_end =
      text_length_ - ignored_terminal_tokens_;
  const std::uint64_t requested_window_end =
      static_cast<std::uint64_t>(search_start) + lookahead_;
  const std::uint32_t window_end = std::min(
      static_cast<std::uint32_t>(
          std::min<std::uint64_t>(
              requested_window_end,
              std::numeric_limits<std::uint32_t>::max())),
      content_end);

  std::uint32_t attended = text_length_ - 1U;
  if (window_end > search_start) {
    attended = search_start;
    float best = conditional_alignment_scores.at(search_start);
    for (std::uint32_t index = search_start + 1U;
         index < window_end;
         ++index) {
      const float candidate =
          conditional_alignment_scores.at(index);
      if (candidate > best) {
        attended = index;
        best = candidate;
      }
    }
  }
  last_attended_ = attended;
  if (counters_.at(attended) ==
      std::numeric_limits<std::uint32_t>::max()) {
    fail(
        AlignmentControllerErrorCode::state_overflow,
        "alignment sink counter overflowed");
  }
  ++counters_.at(attended);

  const float epsilon = static_cast<float>(prior_epsilon_);
  std::vector<float> prior(
      static_cast<std::size_t>(text_length_) * 2U, epsilon);
  if (text_length_ <= short_text_no_prior_max_tokens_) {
    std::fill(
        prior.begin(),
        prior.begin() + static_cast<std::ptrdiff_t>(text_length_),
        1.0F);
  } else {
    const std::uint32_t history_floor =
        std::min<std::uint32_t>(1U, text_length_ - 1U);
    const std::uint32_t history =
        std::clamp(
            attended == 0 ? 0U : attended - 1U,
            history_floor,
            text_length_ - 1U);
    prior.at(history) = 1.0F;
    prior.at(attended) = 1.0F;
    for (std::uint32_t offset = 1; offset <= lookahead_; ++offset) {
      const std::uint64_t requested =
          static_cast<std::uint64_t>(attended) + offset;
      const std::uint32_t index = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(requested, text_length_ - 1U));
      prior.at(index) = 1.0F;
    }
  }

  std::int64_t maximum_sink_position = -1;
  for (std::uint32_t index = 0; index < text_length_; ++index) {
    if (counters_.at(index) >= sink_threshold_) {
      maximum_sink_position = index;
    }
  }
  if (maximum_sink_position >= 0) {
    std::fill(
        prior.begin(),
        prior.begin() + maximum_sink_position + 1,
        epsilon);
  }

  return AlignmentStepResult{
      .attended_token_index = attended,
      .unfinished_text = attended < content_end,
      .next_prior = std::move(prior),
  };
}

std::uint32_t
AlignmentControllerReference::last_attended_token_index() const noexcept {
  return last_attended_;
}

const std::vector<std::uint32_t>&
AlignmentControllerReference::counters() const noexcept {
  return counters_;
}

}  // namespace magpie_tts_rt

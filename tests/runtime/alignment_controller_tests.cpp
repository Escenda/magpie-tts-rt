#include "runtime/alignment_controller.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using magpie_tts_rt::AlignmentControllerError;
using magpie_tts_rt::AlignmentControllerErrorCode;
using magpie_tts_rt::AlignmentControllerReference;
using magpie_tts_rt::AlignmentManifest;
using magpie_tts_rt::TensorDataType;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] AlignmentManifest contract() {
  return AlignmentManifest{
      .dtype = TensorDataType::bf16,
      .source_decoder_layers = {4, 5, 8, 9},
      .prefill_output_binding = "alignment",
      .step_prior_input_binding = "alignment_prior",
      .step_alignment_output_binding = "alignment",
      .prior_epsilon = 0.1,
      .initial_attended = 1,
      .ignored_terminal_tokens = 3,
      .short_text_no_prior_max_tokens = 5,
      .lookahead = 6,
      .sink_threshold = 4,
      .source_position_policy = "exact_frontend_span_only",
  };
}

void test_window_sink_and_prior() {
  AlignmentControllerReference controller(12, contract());
  std::vector<float> scores(12, 0.0F);
  scores.at(4) = 2.0F;
  const auto first = controller.advance(scores);
  require(first.attended_token_index == 4, "first attended token");
  require(first.unfinished_text, "content ended too early");
  require(first.next_prior.size() == 24, "CFG prior shape");
  require(first.next_prior.at(3) == 1.0F, "history weight");
  require(first.next_prior.at(4) == 1.0F, "attended weight");
  require(first.next_prior.at(9) == 1.0F, "lookahead weight");
  require(first.next_prior.at(12) == 0.1F, "unconditional prior");

  scores.assign(12, 0.0F);
  scores.at(4) = 3.0F;
  for (std::uint32_t repeat = 0; repeat < 3; ++repeat) {
    const auto result = controller.advance(scores);
    require(result.attended_token_index == 4, "sink repeat");
  }
  scores.at(5) = 4.0F;
  const auto advanced = controller.advance(scores);
  require(advanced.attended_token_index == 5, "sink did not advance");
  for (std::uint32_t index = 0; index <= 4; ++index) {
    require(
        advanced.next_prior.at(index) == 0.1F,
        "sunk history was not masked");
  }
}

void test_short_text_and_terminal_window() {
  AlignmentControllerReference short_controller(5, contract());
  const auto short_result =
      short_controller.advance({0.0F, 3.0F, 1.0F, 0.0F, 0.0F});
  for (std::uint32_t index = 0; index < 5; ++index) {
    require(
        short_result.next_prior.at(index) == 1.0F,
        "short text prior must be all ones");
  }

  AlignmentControllerReference terminal_controller(8, contract());
  std::vector<float> scores(8, 0.0F);
  scores.at(4) = 2.0F;
  require(
      terminal_controller.advance(scores).attended_token_index == 4,
      "terminal precondition");
  scores.assign(8, 0.0F);
  scores.at(4) = 1.0F;
  const auto result = terminal_controller.advance(scores);
  require(result.attended_token_index == 4, "terminal window selection");
  require(result.unfinished_text, "token four should remain unfinished");
}

void test_invalid_input_fails_closed() {
  try {
    AlignmentControllerReference invalid(3, contract());
    static_cast<void>(invalid);
    throw std::runtime_error("short invalid text was accepted");
  } catch (const AlignmentControllerError& error) {
    require(
        error.code() ==
            AlignmentControllerErrorCode::invalid_text_length,
        "unexpected short-text error");
  }

  AlignmentControllerReference controller(8, contract());
  try {
    static_cast<void>(controller.advance({1.0F}));
    throw std::runtime_error("wrong score count was accepted");
  } catch (const AlignmentControllerError& error) {
    require(
        error.code() ==
            AlignmentControllerErrorCode::invalid_score_count,
        "unexpected score-count error");
  }
}

}  // namespace

int main() {
  try {
    test_window_sink_and_prior();
    test_short_text_and_terminal_window();
    test_invalid_input_fails_closed();
  } catch (const std::exception& error) {
    std::cerr << "alignment controller test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "runtime/pipeline_constants.hpp"

namespace magpie_tts_rt {

// One fixed decoder-step diagnostic record copied from the generation stream.
// reserved is written as zero by the packing kernel and checked after the D2H
// transfer so a partial or layout-incompatible copy fails closed.
struct GenerationStepDiagnostics final {
  std::int64_t attended_token_index;
  std::int32_t alignment_invalid;
  std::int32_t local_invalid_rows;
  std::int32_t end_frame_index;
  std::int32_t reserved;
};

// This is the sole device-to-host diagnostic payload for one generation batch.
// It deliberately retains every value that was previously copied separately.
struct GenerationBatchDiagnostics final {
  std::uint32_t step_count;
  std::int32_t main_decoder_execution_status;
  GenerationStepDiagnostics steps[kMaximumDecoderStepsPerEmission];
};

// Device addresses consumed by the packing kernel. The fixed four-step shape
// makes every source explicit and prevents an untyped pointer table from
// crossing the host/device boundary.
struct GenerationDiagnosticSources final {
  const std::int32_t* main_decoder_execution_status;
  const std::int64_t*
      attended_token_indices[kMaximumDecoderStepsPerEmission];
  const std::int32_t*
      alignment_invalid_steps[kMaximumDecoderStepsPerEmission];
  const std::int32_t*
      local_invalid_rows[kMaximumDecoderStepsPerEmission];
  const std::int32_t*
      end_frame_indices[kMaximumDecoderStepsPerEmission];
};

enum class GenerationDiagnosticPackArgumentError {
  none,
  invalid_step_count,
  missing_output,
  missing_main_decoder_status,
  missing_attended_token_index,
  missing_alignment_invalid,
  missing_local_invalid_rows,
  missing_end_frame_index,
};

[[nodiscard]] constexpr GenerationDiagnosticPackArgumentError
validate_generation_diagnostic_pack_arguments(
    const GenerationDiagnosticSources& sources,
    const std::uint32_t step_count,
    const GenerationBatchDiagnostics* const output) noexcept {
  if (step_count == 0U ||
      step_count > kMaximumDecoderStepsPerEmission) {
    return GenerationDiagnosticPackArgumentError::invalid_step_count;
  }
  if (output == nullptr) {
    return GenerationDiagnosticPackArgumentError::missing_output;
  }
  if (sources.main_decoder_execution_status == nullptr) {
    return GenerationDiagnosticPackArgumentError::
        missing_main_decoder_status;
  }
  for (std::size_t step = 0;
       step < kMaximumDecoderStepsPerEmission;
       ++step) {
    if (sources.attended_token_indices[step] == nullptr) {
      return GenerationDiagnosticPackArgumentError::
          missing_attended_token_index;
    }
    if (sources.alignment_invalid_steps[step] == nullptr) {
      return GenerationDiagnosticPackArgumentError::
          missing_alignment_invalid;
    }
    if (sources.local_invalid_rows[step] == nullptr) {
      return GenerationDiagnosticPackArgumentError::
          missing_local_invalid_rows;
    }
    if (sources.end_frame_indices[step] == nullptr) {
      return GenerationDiagnosticPackArgumentError::
          missing_end_frame_index;
    }
  }
  return GenerationDiagnosticPackArgumentError::none;
}

enum class GenerationDiagnosticPayloadError {
  none,
  invalid_expected_step_count,
  packed_step_count_mismatch,
  nonzero_reserved_field,
  noncanonical_unused_step,
};

struct GenerationDiagnosticPayloadValidation final {
  GenerationDiagnosticPayloadError code;
  std::size_t step;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return code == GenerationDiagnosticPayloadError::none;
  }
};

[[nodiscard]] constexpr GenerationDiagnosticPayloadValidation
validate_generation_diagnostic_payload(
    const GenerationBatchDiagnostics& diagnostics,
    const std::size_t expected_step_count) noexcept {
  if (expected_step_count == 0U ||
      expected_step_count > kMaximumDecoderStepsPerEmission) {
    return GenerationDiagnosticPayloadValidation{
        .code = GenerationDiagnosticPayloadError::
            invalid_expected_step_count,
        .step = 0U,
    };
  }
  if (diagnostics.step_count != expected_step_count) {
    return GenerationDiagnosticPayloadValidation{
        .code = GenerationDiagnosticPayloadError::
            packed_step_count_mismatch,
        .step = 0U,
    };
  }
  for (std::size_t step = 0;
       step < kMaximumDecoderStepsPerEmission;
       ++step) {
    const GenerationStepDiagnostics& current = diagnostics.steps[step];
    if (current.reserved != 0) {
      return GenerationDiagnosticPayloadValidation{
          .code = GenerationDiagnosticPayloadError::
              nonzero_reserved_field,
          .step = step,
      };
    }
    if (step >= expected_step_count &&
        (current.attended_token_index != -1 ||
         current.alignment_invalid != 0 ||
         current.local_invalid_rows != 0 ||
         current.end_frame_index != -1)) {
      return GenerationDiagnosticPayloadValidation{
          .code = GenerationDiagnosticPayloadError::
              noncanonical_unused_step,
          .step = step,
      };
    }
  }
  return GenerationDiagnosticPayloadValidation{
      .code = GenerationDiagnosticPayloadError::none,
      .step = 0U,
  };
}

static_assert(std::is_standard_layout_v<GenerationStepDiagnostics>);
static_assert(std::is_trivially_copyable_v<GenerationStepDiagnostics>);
static_assert(sizeof(GenerationStepDiagnostics) == 24U);
static_assert(std::is_standard_layout_v<GenerationBatchDiagnostics>);
static_assert(std::is_trivially_copyable_v<GenerationBatchDiagnostics>);
static_assert(sizeof(GenerationBatchDiagnostics) == 104U);
static_assert(std::is_standard_layout_v<GenerationDiagnosticSources>);
static_assert(std::is_trivially_copyable_v<GenerationDiagnosticSources>);

}  // namespace magpie_tts_rt

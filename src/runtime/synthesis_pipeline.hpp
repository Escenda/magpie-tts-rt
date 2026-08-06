#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "manifest/manifest.hpp"
#include "runtime/async_pipeline_contract.hpp"
#include "runtime/request_state.hpp"
#include "runtime/session_resources.hpp"
#include "runtime/startup_golden.hpp"

namespace magpie_tts_rt {

enum class SynthesisPipelineErrorCode {
  cuda_failure,
  engine_failure,
  alignment_failure,
  local_ar_failure,
  codec_failure,
  context_exhausted,
  invariant_failure,
};

[[nodiscard]] std::string_view to_string(
    SynthesisPipelineErrorCode code) noexcept;

class SynthesisPipelineError final : public std::runtime_error {
 public:
  SynthesisPipelineError(
      SynthesisPipelineErrorCode code,
      std::string detail);

  [[nodiscard]] SynthesisPipelineErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  SynthesisPipelineErrorCode code_;
  std::string detail_;
};

// Runs one utterance on a session's dedicated execution contexts. This method
// is blocking for its caller and is intended to run on the C ABI request
// worker thread. CUDA synchronization is stream/event scoped; it never calls
// cudaDeviceSynchronize on the healthy path.
void run_synthesis_pipeline(
    const RuntimeBundleManifest& manifest,
    SessionResources& resources,
    const std::vector<std::int32_t>& text_token_ids,
    std::uint32_t random_seed,
    StreamingRequestState& request_state,
    StartupGoldenCapture* startup_capture,
    PipelineSynchronizationMetrics* synchronization_metrics = nullptr);

}  // namespace magpie_tts_rt

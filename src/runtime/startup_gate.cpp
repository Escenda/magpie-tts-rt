#include "runtime/startup_gate.hpp"

#include <cstdint>

#include "runtime/request_state.hpp"
#include "runtime/startup_gate_drain.hpp"
#include "runtime/synthesis_pipeline.hpp"

namespace magpie_tts_rt {

void run_startup_golden_gate(
    const RuntimeBundleManifest& manifest,
    const StartupGoldenFixture& fixture,
    SessionResources& resources) {
  resources.begin_cuda_graph_memory_accounting();
  LeaseIdSequence lease_ids;
  StreamingRequestState request_state(
      fixture.prepared_token_ids.size(),
      manifest.limits.pcm_ring_capacity_frames,
      lease_ids);
  StartupGoldenCapture capture(
      manifest.local_ar.codebooks_per_frame,
      manifest.codec.hop_length_samples,
      manifest.limits.maximum_audio_frames);
  run_startup_producer_until_drained(
      request_state,
      [&]() {
      run_synthesis_pipeline(
          manifest,
          resources,
          fixture.prepared_token_ids,
          fixture.seed,
          request_state,
          &capture);
      },
      [&](auto&& producer_finished) {
        drain_startup_request_until_finished(
            request_state,
            std::forward<decltype(producer_finished)>(
                producer_finished));
      });

  resources.finalize_cuda_graph_memory_accounting();
  const RequestStateSnapshot terminal = request_state.snapshot();
  if (!resources.main_decoder_graphs_ready() ||
      !resources.local_ar_graph_ready() ||
      !resources.nanocodec_graphs_ready() ||
      !resources.cuda_graph_memory_accounted()) {
    throw StartupGoldenError(
        StartupGoldenErrorCode::invalid_capture,
        "startup synthesis did not accept every required Main Decoder, "
        "Local AR, and NanoCodec CUDA graph");
  }
  if (terminal.state != RequestLifecycleState::completed ||
      terminal.available_audio_leases != 0 ||
      request_state.has_live_leases()) {
    throw StartupGoldenError(
        StartupGoldenErrorCode::invalid_capture,
        "startup synthesis did not finish as a fully drained completed request");
  }
  const StartupGoldenActual actual = capture.finish();
  require_startup_golden_match(fixture, actual);
}

}  // namespace magpie_tts_rt

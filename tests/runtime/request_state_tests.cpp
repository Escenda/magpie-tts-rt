#include "runtime/request_state.hpp"
#include "runtime/eos_contract.hpp"
#include "runtime/startup_gate_drain.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using magpie_tts_rt::AlignmentProgress;
using magpie_tts_rt::AudioChunk;
using magpie_tts_rt::LeaseIdSequence;
using magpie_tts_rt::PreparedTokenErrorCode;
using magpie_tts_rt::RequestLifecycleState;
using magpie_tts_rt::RequestStateError;
using magpie_tts_rt::RequestStateErrorCode;
using magpie_tts_rt::RequestStateSnapshot;
using magpie_tts_rt::StreamingRequestState;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_error(
    Function&& function,
    const RequestStateErrorCode expected) {
  try {
    function();
  } catch (const RequestStateError& error) {
    require(error.code() == expected, "unexpected request-state error");
    return;
  }
  throw std::runtime_error("expected RequestStateError");
}

AudioChunk chunk(
    const std::uint64_t sequence,
    const std::uint64_t first_sample,
    const std::uint32_t frames,
    const bool first,
    const bool final,
    std::vector<AlignmentProgress> events = {}) {
  const bool alignment_valid = !events.empty();
  const std::uint64_t committed_text_tokens =
      events.empty() ? 0 : events.back().committed_text_tokens;
  return AudioChunk{
      .samples =
          std::vector<float>(
              static_cast<std::size_t>(frames) *
                  magpie_tts_rt::kCodecSamplesPerFrame,
              static_cast<float>(sequence + 1)),
      .first_sample_index = first_sample,
      .sequence = sequence,
      .codec_frame_count = frames,
      .first = first,
      .final = final,
      .alignment_valid = alignment_valid,
      .committed_text_tokens = committed_text_tokens,
      .alignment_events = std::move(events),
  };
}

void require_publish(
    StreamingRequestState& state,
    AudioChunk audio_chunk) {
  require(
      state.publish(std::move(audio_chunk)),
      "healthy publication was rejected");
}

void test_prepared_token_sequence_contract() {
  const std::vector<std::int64_t> valid{0, 3356, 3358};
  require(
      magpie_tts_rt::validate_prepared_token_ids(
          valid, 3357, 3358)
          .valid(),
      "normal rows followed by EOS must be accepted");

  for (const auto& [tokens, expected] :
       std::vector<std::pair<
           std::vector<std::int64_t>,
           PreparedTokenErrorCode>>{
           {{}, PreparedTokenErrorCode::empty},
           {{0}, PreparedTokenErrorCode::final_token_is_not_eos},
           {{0, 3358, 3358},
            PreparedTokenErrorCode::
                non_final_token_is_not_a_normal_row},
           {{0, 3357, 3358},
            PreparedTokenErrorCode::
                non_final_token_is_not_a_normal_row},
           {{0, 3359, 3358},
            PreparedTokenErrorCode::
                non_final_token_is_not_a_normal_row},
           {{-1, 3358},
            PreparedTokenErrorCode::
                non_final_token_is_not_a_normal_row},
       }) {
    require(
        magpie_tts_rt::validate_prepared_token_ids(
            tokens, 3357, 3358)
                .code == expected,
        "invalid prepared sequence did not fail with the expected code");
  }
}

void test_streaming_schedule_and_lease_lifetime() {
  LeaseIdSequence ids;
  StreamingRequestState state(39, 12, ids);
  require_publish(state, chunk(
      0,
      0,
      4,
      true,
      false,
      {{2048, 2}, {4096, 5}}));
  require(state.can_publish(8), "steady chunk should fit");
  require_publish(state, chunk(
      1,
      4096,
      8,
      false,
      false,
      {{6144, 7}, {8192, 9}, {10240, 12}, {12288, 15}}));
  require(!state.can_publish(1), "ring must apply backpressure");

  auto first = state.acquire_audio();
  require(first.lease_id == 1, "unexpected first lease identifier");
  require(first.samples != nullptr, "first lease PCM is null");
  require(first.sample_count == 4096, "first lease sample count");
  require(first.alignment_event_count == 2, "first alignment count");
  require(first.committed_text_tokens == 5, "first committed tokens");
  const float* retained_samples = first.samples;
  const float retained_value = retained_samples[0];
  require(!state.can_publish(1), "live lease must retain ring capacity");
  state.release_audio(first.lease_id);
  require(retained_value == 1.0F, "unexpected retained PCM value");
  require(state.can_publish(4), "release must restore ring capacity");

  auto steady = state.acquire_audio();
  require(steady.lease_id == 2, "unexpected steady lease identifier");
  state.release_audio(steady.lease_id);

  require_publish(state, chunk(
      2,
      12288,
      3,
      false,
      true,
      {{14336, 18}, {15360, 20}}));
  const RequestStateSnapshot terminal = state.snapshot();
  require(
      terminal.state == RequestLifecycleState::completed,
      "final publish must complete request");
  require(terminal.generated_codec_frames == 15, "generated frame count");
  require(terminal.published_samples == 15360, "published sample count");
  require(terminal.committed_text_tokens == 20, "terminal text progress");
  auto tail = state.acquire_audio();
  require(tail.final, "tail must carry FINAL");
  state.release_audio(tail.lease_id);
  require(!state.has_live_leases(), "all leases must be released");
}

void test_exact_eight_frame_ring_blocks_pinned_slot_reuse() {
  LeaseIdSequence ids;
  StreamingRequestState state(39, 8, ids);
  require_publish(
      state,
      chunk(0, 0, 4, true, false, {{4096, 5}}));
  const auto first = state.acquire_audio();
  const float* const first_pcm = first.samples;
  const float first_sample = first_pcm[0];
  require(
      !state.can_publish(8),
      "a live first lease admitted reuse by a steady decode");
  require_error(
      [&]() {
        static_cast<void>(
            state.publish(
                chunk(1, 4096, 8, false, false)));
      },
      RequestStateErrorCode::backpressure);
  require(
      first_pcm[0] == first_sample,
      "rejected pinned-slot reuse changed live lease PCM");

  state.release_audio(first.lease_id);
  require(
      state.can_publish(8),
      "releasing the first lease did not admit the steady decode");
  require_publish(
      state,
      chunk(1, 4096, 8, false, false));
  const auto steady = state.acquire_audio();
  require(
      !state.can_publish(1),
      "a live steady lease admitted pinned-slot reuse");
  state.release_audio(steady.lease_id);
  require(
      state.occupied_codec_frames() == 0,
      "releasing the steady lease did not free the pinned slot");
}

void test_eos_frame_is_excluded_from_audio() {
  require(
      magpie_tts_rt::retained_codec_frames_before_eos(0, 0) == 0,
      "frame-zero EOS retained an audio frame");
  require(
      magpie_tts_rt::retained_codec_frames_before_eos(0, 1) == 1,
      "frame-one EOS did not retain exactly frame zero");
  require(
      magpie_tts_rt::retained_codec_frames_before_eos(3, 0) == 6,
      "later frame-zero EOS retained the wrong prefix");
  require(
      magpie_tts_rt::retained_codec_frames_before_eos(3, 1) == 7,
      "later frame-one EOS retained the wrong prefix");
  try {
    static_cast<void>(
        magpie_tts_rt::retained_codec_frames_before_eos(0, -1));
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid EOS index did not fail closed");
}

void test_invalid_chunks_fail_without_advancing_state() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 8, ids);
  const RequestStateSnapshot initial = state.snapshot();
  require_error(
      [&]() {
        static_cast<void>(
            state.publish(chunk(0, 0, 3, true, false)));
      },
      RequestStateErrorCode::invalid_audio_chunk);
  require(state.snapshot().revision == initial.revision, "invalid publish revision");
  require_error(
      [&]() {
        static_cast<void>(state.publish(chunk(
            0,
            0,
            4,
            true,
            false,
            {{0, 1}})));
      },
      RequestStateErrorCode::invalid_audio_chunk);
  require_error(
      [&]() {
        static_cast<void>(state.publish(chunk(
            0,
            0,
            4,
            true,
            false,
            {{2048, 2}, {4096, 2}})));
      },
      RequestStateErrorCode::invalid_audio_chunk);
  require_error(
      [&]() {
        static_cast<void>(state.publish(chunk(
            0,
            0,
            4,
            true,
            false,
            {{2048, 11}})));
      },
      RequestStateErrorCode::invalid_audio_chunk);
}

void test_zero_frame_final_control_marker() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 8, ids);
  require_publish(state, chunk(0, 0, 4, true, false));
  auto first = state.acquire_audio();
  state.release_audio(first.lease_id);
  require(
      !first.alignment_valid,
      "request-state lease invented alignment validity");

  AudioChunk marker = chunk(1, 4096, 0, false, true);
  marker.alignment_valid = true;
  require_publish(state, std::move(marker));
  const RequestStateSnapshot terminal = state.snapshot();
  require(
      terminal.state == RequestLifecycleState::completed,
      "zero-frame FINAL marker did not complete the request");
  require(
      terminal.generated_codec_frames == 4 &&
          terminal.published_samples == 4096,
      "zero-frame FINAL marker changed audio accounting");
  const auto final = state.acquire_audio();
  require(final.samples == nullptr, "zero-frame marker exposed a PCM pointer");
  require(final.sample_count == 0, "zero-frame marker exposed PCM samples");
  require(final.first_sample_index == 4096, "zero-frame marker moved sample offset");
  require(final.sequence == 1, "zero-frame marker sequence is not contiguous");
  require(!final.first && final.final, "zero-frame marker flags are invalid");
  require(final.alignment_valid, "zero-frame marker lost alignment validity");
  state.release_audio(final.lease_id);
  require(
      state.occupied_codec_frames() == 0,
      "zero-frame marker consumed ring capacity");

  StreamingRequestState invalid(10, 8, ids);
  require_error(
      [&]() {
        static_cast<void>(
            invalid.publish(chunk(0, 0, 0, false, true)));
      },
      RequestStateErrorCode::invalid_audio_chunk);
  require_publish(invalid, chunk(0, 0, 4, true, false));
  const auto initial = invalid.acquire_audio();
  invalid.release_audio(initial.lease_id);
  require_error(
      [&]() {
        static_cast<void>(
            invalid.publish(chunk(1, 4096, 0, false, false)));
      },
      RequestStateErrorCode::invalid_audio_chunk);
}

void test_cancellation_discards_only_unleased_audio() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 12, ids);
  require_publish(state, chunk(0, 0, 4, true, false));
  auto lease = state.acquire_audio();
  require_publish(state, chunk(1, 4096, 8, false, false));
  require(
      state.request_cancellation(),
      "running cancellation was not accepted");
  require(
      state.snapshot().state == RequestLifecycleState::running,
      "cancellation became terminal before a drained boundary");
  require(state.has_live_leases(), "cancel invalidated a live lease");
  require(
      state.occupied_codec_frames() == 4,
      "cancel did not retain exactly the live lease frames");
  require(
      state.complete_cancellation_after_drain(),
      "accepted cancellation was not observed at a boundary");
  state.release_audio(lease.lease_id);
  require(state.occupied_codec_frames() == 0, "release did not drain cancelled ring");

  StreamingRequestState cancellable(10, 8, ids);
  require_publish(
      cancellable, chunk(0, 0, 4, true, false));
  require(
      cancellable.request_cancellation(),
      "second cancellation was not accepted");
  require(
      cancellable.complete_cancellation_after_drain(),
      "second cancellation was not observed");
  const RequestStateSnapshot snapshot = cancellable.snapshot();
  require(
      snapshot.state == RequestLifecycleState::cancelled,
      "request is not cancelled");
  require(snapshot.available_audio_leases == 0, "cancelled queue was not discarded");
  require(cancellable.occupied_codec_frames() == 0, "cancelled ring remains occupied");
  require_error(
      [&]() { static_cast<void>(cancellable.acquire_audio()); },
      RequestStateErrorCode::no_audio);
}

void test_cancellation_closes_publication_gate() {
  LeaseIdSequence ids;
  StreamingRequestState cancelled(10, 8, ids);
  require(
      cancelled.request_cancellation(),
      "cancellation gate was not accepted");
  require(
      !cancelled.can_publish(4),
      "accepted cancellation left publication capacity open");
  require(
      !cancelled.publish(chunk(0, 0, 4, true, false)),
      "audio was published after cancellation acceptance");
  const RequestStateSnapshot cancelled_snapshot =
      cancelled.snapshot();
  require(
      cancelled_snapshot.state ==
          RequestLifecycleState::running,
      "rejected post-cancel publish exposed cancellation before drain");
  require(
      cancelled_snapshot.available_audio_leases == 0 &&
          cancelled_snapshot.published_samples == 0,
      "post-cancel audio became observable");
  require(
      cancelled.cancellation_requested(),
      "accepted cancellation was not retained through publication rejection");
  require(
      cancelled.complete_cancellation_after_drain(),
      "drained boundary did not publish accepted cancellation");
  require(
      cancelled.snapshot().state ==
          RequestLifecycleState::cancelled,
      "drained cancellation did not become terminal");
  require(
      cancelled.request_cancellation(),
      "cancelled request was not idempotent");

  StreamingRequestState completed(10, 8, ids);
  require_publish(
      completed, chunk(0, 0, 4, true, false));
  const auto initial = completed.acquire_audio();
  completed.release_audio(initial.lease_id);
  require_publish(
      completed, chunk(1, 4096, 1, false, true));
  require(
      !completed.request_cancellation(),
      "completion did not win the cancellation terminal race");
}

void test_drain_failure_wins_after_cancellation_acceptance() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 8, ids);
  require_publish(state, chunk(0, 0, 4, true, false));
  require(
      state.request_cancellation(),
      "cancellation was not accepted before simulated drain failure");
  require(
      state.cancellation_requested(),
      "accepted cancellation was not visible while draining");

  state.fail(9, 8, "CUDA event drain failed");
  const RequestStateSnapshot failed = state.snapshot();
  require(
      failed.state == RequestLifecycleState::failed,
      "drain failure was hidden by premature cancellation");
  require(
      failed.terminal_status == 9 &&
          failed.terminal_error_stage == 8 &&
          failed.terminal_error_message ==
              "CUDA event drain failed",
      "drain failure diagnostic was not retained");
  require(
      !state.cancellation_requested(),
      "terminal failure still reported a pending cancellation");
  require(
      !state.complete_cancellation_after_drain(),
      "failed request accepted a late cancellation completion");
  require(
      state.snapshot().state ==
          RequestLifecycleState::failed,
      "late cancellation boundary replaced the drain failure");
}

void test_cancellation_and_final_publish_are_serialized() {
  LeaseIdSequence ids;
  for (std::uint32_t iteration = 0; iteration < 64; ++iteration) {
    StreamingRequestState state(10, 8, ids);
    require_publish(
        state, chunk(0, 0, 4, true, false));
    const auto initial = state.acquire_audio();
    state.release_audio(initial.lease_id);

    std::barrier start(3);
    bool published = false;
    bool cancellation_accepted = false;
    std::thread publisher([&]() {
      start.arrive_and_wait();
      published =
          state.publish(chunk(1, 4096, 1, false, true));
    });
    std::thread canceller([&]() {
      start.arrive_and_wait();
      cancellation_accepted = state.request_cancellation();
    });
    start.arrive_and_wait();
    publisher.join();
    canceller.join();

    require(
        published != cancellation_accepted,
        "final publication and cancellation were both accepted");
    const RequestStateSnapshot terminal = state.snapshot();
    if (cancellation_accepted) {
      require(
          terminal.state == RequestLifecycleState::running &&
              terminal.available_audio_leases == 0 &&
              terminal.published_samples == 4096,
          "accepted cancellation became terminal before a drained boundary");
      require(
          state.complete_cancellation_after_drain(),
          "final-publication race lost its accepted cancellation");
      require(
          state.snapshot().state ==
              RequestLifecycleState::cancelled,
          "drained final-publication race did not become cancelled");
    } else {
      require(
          terminal.state == RequestLifecycleState::completed &&
              terminal.available_audio_leases == 1 &&
              terminal.published_samples == 5120,
          "winning final publication was not retained");
      const auto tail = state.acquire_audio();
      state.release_audio(tail.lease_id);
    }
  }
}

void test_cancellation_and_zero_final_are_serialized() {
  LeaseIdSequence ids;
  for (std::uint32_t iteration = 0; iteration < 64; ++iteration) {
    StreamingRequestState state(10, 8, ids);
    require_publish(
        state, chunk(0, 0, 4, true, false));
    const auto initial = state.acquire_audio();
    state.release_audio(initial.lease_id);

    std::barrier start(3);
    bool published = false;
    bool cancellation_accepted = false;
    std::thread publisher([&]() {
      start.arrive_and_wait();
      AudioChunk marker = chunk(1, 4096, 0, false, true);
      marker.alignment_valid = true;
      published = state.publish(std::move(marker));
    });
    std::thread canceller([&]() {
      start.arrive_and_wait();
      cancellation_accepted = state.request_cancellation();
    });
    start.arrive_and_wait();
    publisher.join();
    canceller.join();

    require(
        published != cancellation_accepted,
        "zero-frame FINAL and cancellation were both accepted");
    const RequestStateSnapshot terminal = state.snapshot();
    if (cancellation_accepted) {
      require(
          terminal.state == RequestLifecycleState::running &&
              terminal.available_audio_leases == 0 &&
              terminal.published_samples == 4096,
          "accepted cancellation became terminal before a drained boundary");
      require(
          state.complete_cancellation_after_drain(),
          "zero-frame race lost its accepted cancellation");
      require(
          state.snapshot().state ==
              RequestLifecycleState::cancelled,
          "drained zero-frame race did not become cancelled");
    } else {
      require(
          terminal.state == RequestLifecycleState::completed &&
              terminal.available_audio_leases == 1 &&
              terminal.published_samples == 4096 &&
              terminal.generated_codec_frames == 4,
          "winning zero-frame FINAL changed audio accounting");
      const auto marker = state.acquire_audio();
      require(
          marker.final && marker.sample_count == 0,
          "winning zero-frame FINAL lease is malformed");
      state.release_audio(marker.lease_id);
    }
  }
}

void test_failure_diagnostic_is_persistent() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 8, ids);
  state.fail(8, 7, "TensorRT execution failed");
  const RequestStateSnapshot first = state.snapshot();
  const RequestStateSnapshot second = state.snapshot();
  require(first.state == RequestLifecycleState::failed, "request is not failed");
  require(first.terminal_status == 8, "failed status was not retained");
  require(first.terminal_error_stage == 7, "failed stage was not retained");
  require(
      first.terminal_error_message == "TensorRT execution failed",
      "failed message was not retained");
  require(
      second.terminal_error_message == first.terminal_error_message,
      "failed diagnostic changed");
}

void test_revision_wait_and_timeout() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 8, ids);
  const std::uint64_t revision = state.snapshot().revision;
  RequestStateSnapshot output{};
  require(
      !state.wait_for_revision(
          revision,
          std::chrono::nanoseconds::zero(),
          output),
      "zero timeout returned stale snapshot");
  std::thread publisher([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require_publish(state, chunk(0, 0, 4, true, false));
  });
  require(
      state.wait_for_revision(
          revision,
          std::chrono::seconds(1),
          output),
      "wait did not observe publish");
  publisher.join();
  require(output.revision > revision, "wait returned stale revision");
}

void test_startup_drain_rechecks_after_completion_observation() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 8, ids);
  require_publish(state, chunk(0, 0, 4, true, false));

  std::uint32_t completion_checks = 0;
  magpie_tts_rt::drain_startup_request_until_finished(
      state,
      [&]() {
        require(
            completion_checks == 0,
            "startup completion predicate was called after completion");
        ++completion_checks;
        require(
            state.snapshot().available_audio_leases == 0,
            "startup consumer did not drain the first chunk");
        require_publish(
            state,
            chunk(1, 4096, 1, false, true));
        return true;
      });

  const RequestStateSnapshot terminal = state.snapshot();
  require(
      completion_checks == 1,
      "startup completion was not observed exactly once");
  require(
      terminal.state == RequestLifecycleState::completed,
      "startup request did not remain completed");
  require(
      terminal.available_audio_leases == 0,
      "final startup lease was not drained after completion");
  require(
      state.occupied_codec_frames() == 0,
      "final startup lease retained ring capacity");
  require(!state.has_live_leases(), "startup drain retained a live lease");
}

void test_startup_drain_failure_cancels_and_joins_producer() {
  LeaseIdSequence ids;
  StreamingRequestState state(10, 8, ids);
  std::atomic<bool> simulated_gpu_drained{false};
  std::atomic<bool> producer_join_ready{false};
  bool injected_error_observed = false;

  try {
    magpie_tts_rt::run_startup_producer_until_drained(
        state,
        [&]() {
          while (!state.cancellation_requested()) {
            const RequestStateSnapshot snapshot = state.snapshot();
            RequestStateSnapshot changed{};
            static_cast<void>(state.wait_for_revision(
                snapshot.revision,
                std::chrono::milliseconds(10),
                changed));
          }
          // This is the test double for run_synthesis_pipeline's
          // settle_inflight() boundary.
          simulated_gpu_drained.store(true, std::memory_order_release);
          require(
              state.complete_cancellation_after_drain(),
              "startup cancellation was not completed after the GPU drain");
          producer_join_ready.store(true, std::memory_order_release);
        },
        [&](auto&&) {
          throw std::runtime_error("injected startup drain failure");
        });
  } catch (const std::runtime_error& error) {
    injected_error_observed =
        std::string(error.what()) ==
        "injected startup drain failure";
  }

  require(
      injected_error_observed,
      "startup drain fault was not propagated after cleanup");
  require(
      simulated_gpu_drained.load(std::memory_order_acquire),
      "startup drain fault returned before the simulated GPU drain");
  require(
      producer_join_ready.load(std::memory_order_acquire),
      "startup drain fault returned before producer join readiness");
  require(
      state.snapshot().state == RequestLifecycleState::cancelled,
      "startup drain fault did not terminate the private request");
}

void test_lease_sequence_exhaustion_fails_closed() {
  LeaseIdSequence ids(std::numeric_limits<std::uint64_t>::max());
  StreamingRequestState state(10, 8, ids);
  require_publish(state, chunk(0, 0, 4, true, false));
  require_error(
      [&]() { static_cast<void>(state.acquire_audio()); },
      RequestStateErrorCode::lease_sequence_exhausted);
  require(
      state.snapshot().available_audio_leases == 1,
      "failed lease allocation consumed queued audio");
}

}  // namespace

int main() {
  try {
    test_prepared_token_sequence_contract();
    test_streaming_schedule_and_lease_lifetime();
    test_exact_eight_frame_ring_blocks_pinned_slot_reuse();
    test_eos_frame_is_excluded_from_audio();
    test_invalid_chunks_fail_without_advancing_state();
    test_zero_frame_final_control_marker();
    test_cancellation_discards_only_unleased_audio();
    test_cancellation_closes_publication_gate();
    test_drain_failure_wins_after_cancellation_acceptance();
    test_cancellation_and_final_publish_are_serialized();
    test_cancellation_and_zero_final_are_serialized();
    test_failure_diagnostic_is_persistent();
    test_revision_wait_and_timeout();
    test_startup_drain_rechecks_after_completion_observation();
    test_startup_drain_failure_cancels_and_joins_producer();
    test_lease_sequence_exhaustion_fails_closed();
  } catch (const std::exception& error) {
    std::cerr << "request-state test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

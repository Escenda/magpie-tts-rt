#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <thread>
#include <utility>

#include "runtime/request_state.hpp"

namespace magpie_tts_rt {

// Drains the private startup request until its producer has stopped. Observing
// producer completion establishes that no later publication is possible, but a
// final publication may have raced between the preceding empty snapshot and
// that observation. Therefore completion always triggers one additional full
// drain pass before this function returns.
template <typename CompletionPredicate>
void drain_startup_request_until_finished(
    StreamingRequestState& request_state,
    CompletionPredicate&& producer_finished) {
  bool completion_observed = false;
  while (true) {
    RequestStateSnapshot snapshot = request_state.snapshot();
    while (snapshot.available_audio_leases != 0) {
      const AudioLeaseView lease = request_state.acquire_audio();
      request_state.release_audio(lease.lease_id);
      snapshot = request_state.snapshot();
    }

    if (completion_observed) {
      return;
    }
    if (producer_finished()) {
      completion_observed = true;
      continue;
    }

    RequestStateSnapshot changed{};
    static_cast<void>(request_state.wait_for_revision(
        snapshot.revision,
        std::chrono::milliseconds(10),
        changed));
  }
}

// Runs the startup producer and its normal lease-drain path as one
// exception-safe operation. If the normal drain throws, cancellation closes
// publication and releases queued capacity before this function joins the
// producer. The producer is responsible for draining every armed GPU boundary
// before it reports cancellation complete, exactly as run_synthesis_pipeline
// does. The original drain exception is rethrown only after that join.
template <typename Producer, typename Drain>
void run_startup_producer_until_drained(
    StreamingRequestState& request_state,
    Producer&& producer,
    Drain&& drain) {
  std::atomic<bool> producer_done{false};
  std::exception_ptr producer_error;
  std::jthread producer_thread(
      [&request_state,
       &producer,
       &producer_done,
       &producer_error]() {
        try {
          std::forward<Producer>(producer)();
        } catch (...) {
          producer_error = std::current_exception();
        }
        producer_done.store(true, std::memory_order_release);
      });

  std::exception_ptr drain_error;
  try {
    std::forward<Drain>(drain)(
        [&producer_done]() {
          return producer_done.load(std::memory_order_acquire);
        });
  } catch (...) {
    drain_error = std::current_exception();
    // request_cancellation() is allocation-free and discards every queued
    // startup chunk. A running producer blocked by the bounded ring is thereby
    // released and observes cancellation before its next engine/codec frame.
    static_cast<void>(request_state.request_cancellation());
  }

  producer_thread.join();
  if (drain_error != nullptr) {
    std::rethrow_exception(drain_error);
  }
  if (producer_error != nullptr) {
    std::rethrow_exception(producer_error);
  }
}

}  // namespace magpie_tts_rt

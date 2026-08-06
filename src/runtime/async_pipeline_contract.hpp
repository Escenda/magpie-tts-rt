#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "runtime/pipeline_constants.hpp"

namespace magpie_tts_rt {

enum class EventPollStatus {
  complete,
  pending,
  failed,
};

enum class HostWaitBoundary {
  terminal_diagnostics,
  pcm_publication,
};

struct PipelineSynchronizationMetrics {
  std::uint64_t generation_batches{0};
  std::uint64_t codec_batches{0};
  std::uint64_t generation_to_codec_stream_waits{0};
  std::uint64_t codec_to_generation_reuse_waits{0};
  std::uint64_t generation_codec_overlap_opportunities{0};
  std::uint64_t host_terminal_waits{0};
  std::uint64_t host_pcm_waits{0};
  std::uint64_t host_event_queries{0};
  std::uint64_t host_event_pending_observations{0};
  // Healthy request execution must leave this at zero. Teardown stream
  // synchronization is outside this per-request contract.
  std::uint64_t host_cuda_event_synchronizations{0};
};

template <typename Query, typename Backoff>
[[nodiscard]] bool await_event_completion(
    Query&& query,
    Backoff&& backoff,
    const HostWaitBoundary boundary,
    PipelineSynchronizationMetrics& metrics) {
  if (boundary == HostWaitBoundary::terminal_diagnostics) {
    ++metrics.host_terminal_waits;
  } else {
    ++metrics.host_pcm_waits;
  }
  while (true) {
    ++metrics.host_event_queries;
    switch (query()) {
      case EventPollStatus::complete:
        return true;
      case EventPollStatus::pending:
        ++metrics.host_event_pending_observations;
        backoff();
        break;
      case EventPollStatus::failed:
        return false;
    }
  }
}

template <
    typename CanPublish,
    typename CancellationRequested,
    typename SnapshotRevision,
    typename WaitForRevision>
[[nodiscard]] bool await_output_capacity(
    CanPublish&& can_publish,
    CancellationRequested&& cancellation_requested,
    SnapshotRevision&& snapshot_revision,
    WaitForRevision&& wait_for_revision) {
  while (!can_publish()) {
    if (cancellation_requested()) {
      return false;
    }

    const auto revision = snapshot_revision();

    // Capacity may have been released immediately before snapshot_revision()
    // captured the post-release revision. Rechecking here prevents sleeping
    // until the timeout when no further revision is required to arrive.
    if (can_publish()) {
      continue;
    }
    if (cancellation_requested()) {
      return false;
    }

    // A release or cancellation after the snapshot changes the revision, so
    // the implementation's revision wait must return immediately in that
    // race. Its timeout only bounds how long cancellation and capacity go
    // unchecked when no state transition occurs.
    wait_for_revision(revision);
  }
  return !cancellation_requested();
}

class AsyncPipelineContract final {
 public:
  void schedule_generation(const std::size_t slot) {
    BatchSlotState& state = state_for(slot);
    if (state != BatchSlotState::unused &&
        state != BatchSlotState::reuse_dependency_queued) {
      throw std::logic_error(
          "generation batch slot reused without codec-consumed dependency");
    }
    state = BatchSlotState::generation_pending;
    ++metrics_.generation_batches;
  }

  void mark_generation_ready(const std::size_t slot) {
    BatchSlotState& state = state_for(slot);
    if (state != BatchSlotState::generation_pending) {
      throw std::logic_error(
          "generation completion did not match a pending batch");
    }
    state = BatchSlotState::generation_ready;
  }

  void schedule_codec(const std::size_t slot) {
    BatchSlotState& state = state_for(slot);
    if (state != BatchSlotState::generation_ready) {
      throw std::logic_error(
          "codec batch scheduled before generation diagnostics completed");
    }
    state = BatchSlotState::codec_pending;
    ++metrics_.codec_batches;
    ++metrics_.generation_to_codec_stream_waits;
  }

  void queue_reuse_dependency(const std::size_t slot) {
    BatchSlotState& state = state_for(slot);
    if (state != BatchSlotState::codec_pending) {
      throw std::logic_error(
          "generation slot reuse lacks a preceding codec pack");
    }
    state = BatchSlotState::reuse_dependency_queued;
    ++metrics_.codec_to_generation_reuse_waits;
  }

  void record_overlap_opportunity() noexcept {
    ++metrics_.generation_codec_overlap_opportunities;
  }

  [[nodiscard]] PipelineSynchronizationMetrics& metrics() noexcept {
    return metrics_;
  }

  [[nodiscard]] const PipelineSynchronizationMetrics& metrics()
      const noexcept {
    return metrics_;
  }

 private:
  enum class BatchSlotState {
    unused,
    generation_pending,
    generation_ready,
    codec_pending,
    reuse_dependency_queued,
  };

  [[nodiscard]] BatchSlotState& state_for(const std::size_t slot) {
    if (slot >= states_.size()) {
      throw std::out_of_range(
          "generation batch slot must be 0 or 1");
    }
    return states_.at(slot);
  }

  std::array<BatchSlotState, kGenerationBatchSlotCount> states_{};
  PipelineSynchronizationMetrics metrics_{};
};

}  // namespace magpie_tts_rt

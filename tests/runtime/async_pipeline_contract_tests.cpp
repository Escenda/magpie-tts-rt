#include "runtime/async_pipeline_contract.hpp"
#include "runtime/request_failure_policy.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using magpie_tts_rt::AsyncPipelineContract;
using magpie_tts_rt::HostWaitBoundary;
using magpie_tts_rt::PipelineSynchronizationMetrics;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_logic_error(Function&& function) {
  try {
    function();
  } catch (const std::logic_error&) {
    return;
  }
  throw std::runtime_error("expected ownership contract failure");
}

void test_blocking_host_synchronization_counts() {
  AsyncPipelineContract contract;
  contract.record_host_event_synchronization(
      HostWaitBoundary::pcm_publication);
  contract.record_host_event_synchronization(
      HostWaitBoundary::terminal_diagnostics);
  const PipelineSynchronizationMetrics& metrics = contract.metrics();
  require(metrics.host_pcm_waits == 1, "PCM wait count mismatch");
  require(
      metrics.host_terminal_waits == 1,
      "terminal wait count mismatch");
  require(
      metrics.host_cuda_event_synchronizations == 2,
      "blocking CUDA event synchronization count mismatch");
}

void test_double_buffer_ownership_and_overlap_counts() {
  AsyncPipelineContract contract;

  contract.schedule_generation(0);
  contract.mark_generation_ready(0);
  contract.schedule_codec(0);

  // The initial codec batch is published before the next generation batch is
  // admitted, preserving the shortest first-PCM path.
  contract.schedule_generation(1);
  contract.mark_generation_ready(1);
  contract.schedule_codec(1);

  // From the second chunk onward, gen(n+1) is admitted behind the
  // codes-consumed dependency while codec(n) is still running.
  require_logic_error([&]() {
    contract.schedule_generation(0);
  });
  contract.queue_reuse_dependency(0);
  contract.schedule_generation(0);
  contract.record_overlap_opportunity();
  contract.mark_generation_ready(0);

  const PipelineSynchronizationMetrics& metrics = contract.metrics();
  require(metrics.generation_batches == 3, "generation batch count");
  require(metrics.codec_batches == 2, "codec batch count");
  require(
      metrics.generation_to_codec_dependencies == 2,
      "generation-to-codec dependency count");
  require(
      metrics.codec_to_generation_reuse_waits == 1,
      "codec-to-generation reuse wait count");
  require(
      metrics.generation_codec_overlap_opportunities == 1,
      "overlap opportunity count");
  require(
      metrics.host_cuda_event_synchronizations == 0,
      "ownership schedule introduced host event synchronization");
}

void test_output_capacity_release_before_snapshot_does_not_wait() {
  bool capacity_available = false;
  std::uint64_t revision = 41;
  std::uint32_t snapshot_count = 0;
  std::uint32_t wait_count = 0;

  const bool admitted = magpie_tts_rt::await_output_capacity(
      [&]() {
        return capacity_available;
      },
      []() {
        return false;
      },
      [&]() {
        // Deterministically model a lease release immediately before the
        // state snapshot captures the resulting revision.
        capacity_available = true;
        ++revision;
        ++snapshot_count;
        return revision;
      },
      [&](const std::uint64_t) {
        ++wait_count;
      });

  require(admitted, "release before snapshot did not admit output");
  require(snapshot_count == 1, "release-before snapshot count mismatch");
  require(
      wait_count == 0,
      "release before snapshot incorrectly entered the revision wait");
}

void test_output_capacity_release_after_snapshot_wakes_revision_wait() {
  bool capacity_available = false;
  std::uint64_t revision = 73;
  std::uint32_t capacity_check_count = 0;
  std::uint32_t wait_count = 0;

  const bool admitted = magpie_tts_rt::await_output_capacity(
      [&]() {
        ++capacity_check_count;
        const bool available = capacity_available;
        if (capacity_check_count == 2) {
          // The second capacity check still observes the full ring. Release
          // happens immediately afterwards, before wait_for_revision().
          capacity_available = true;
          ++revision;
        }
        return available;
      },
      []() {
        return false;
      },
      [&]() {
        return revision;
      },
      [&](const std::uint64_t after_revision) {
        ++wait_count;
        require(
            revision > after_revision,
            "release after snapshot did not advance the revision");
      });

  require(admitted, "release after snapshot did not admit output");
  require(wait_count == 1, "release-after snapshot wait count mismatch");
}

void test_unknown_inflight_and_allocation_failures_poison_session() {
  using magpie_tts_rt::RequestWorkerFailureClass;
  require(
      magpie_tts_rt::request_failure_requires_session_poison(
          RequestWorkerFailureClass::
              host_allocation_after_possible_enqueue),
      "host allocation failure incorrectly permits session reuse");
  require(
      magpie_tts_rt::request_failure_requires_session_poison(
          RequestWorkerFailureClass::execution_state_unknown),
      "unknown execution state incorrectly permits session reuse");
  require(
      !magpie_tts_rt::request_failure_requires_session_poison(
          RequestWorkerFailureClass::
              context_exhausted_at_proven_quiescent_boundary),
      "pre-enqueue context exhaustion incorrectly poisons the session");
}

}  // namespace

int main() {
  try {
    test_blocking_host_synchronization_counts();
    test_double_buffer_ownership_and_overlap_counts();
    test_output_capacity_release_before_snapshot_does_not_wait();
    test_output_capacity_release_after_snapshot_wakes_revision_wait();
    test_unknown_inflight_and_allocation_failures_poison_session();
    std::cout
        << "host_cuda_event_synchronizations=blocking-boundaries "
        << "generation_slots=2 "
        << "first_pcm_overlap_opportunities=0 "
        << "steady_overlap_opportunities=1\n";
  } catch (const std::exception& error) {
    std::cerr << "async pipeline contract test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

#include "runtime/cuda_graph_memory_accounting.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_immutable_baseline_retains_fixed_graph_charge() {
  const magpie_tts_rt::CudaGraphMemorySnapshot aggregate_baseline{
      .free_bytes = 1'000,
      .graph_used_bytes = 100,
      .graph_reserved_bytes = 200,
      .graph_used_high_bytes = 100,
      .graph_reserved_high_bytes = 200,
  };
  const magpie_tts_rt::CudaGraphMemorySnapshot after_startup{
      .free_bytes = 760,
      .graph_used_bytes = 310,
      .graph_reserved_bytes = 430,
      .graph_used_high_bytes = 320,
      .graph_reserved_high_bytes = 440,
  };
  require(
      magpie_tts_rt::observed_cuda_graph_memory_growth(
          aggregate_baseline, after_startup) == 240,
      "startup aggregate graph charge is incorrect");

  // A later request has replaced only its Main graphs. Measuring from the
  // request-local state below would report 70 bytes and omit the persistent
  // Local/Nano charge. The immutable aggregate baseline must report 260.
  const magpie_tts_rt::CudaGraphMemorySnapshot request_local_before{
      .free_bytes = 810,
      .graph_used_bytes = 270,
      .graph_reserved_bytes = 390,
      .graph_used_high_bytes = 270,
      .graph_reserved_high_bytes = 390,
  };
  const magpie_tts_rt::CudaGraphMemorySnapshot after_larger_request{
      .free_bytes = 740,
      .graph_used_bytes = 330,
      .graph_reserved_bytes = 450,
      .graph_used_high_bytes = 340,
      .graph_reserved_high_bytes = 460,
  };
  require(
      magpie_tts_rt::observed_cuda_graph_memory_growth(
          aggregate_baseline, after_larger_request) == 260,
      "request measurement dropped persistent graph memory");
  require(
      magpie_tts_rt::observed_cuda_graph_memory_growth(
          request_local_before, after_larger_request) == 70,
      "test fixture does not distinguish an invalid request-local baseline");
}

void test_high_water_and_free_memory_are_both_fail_closed() {
  const magpie_tts_rt::CudaGraphMemorySnapshot baseline{
      .free_bytes = 2'000,
      .graph_used_bytes = 500,
      .graph_reserved_bytes = 700,
      .graph_used_high_bytes = 500,
      .graph_reserved_high_bytes = 700,
  };
  const magpie_tts_rt::CudaGraphMemorySnapshot observed{
      .free_bytes = 1'950,
      .graph_used_bytes = 520,
      .graph_reserved_bytes = 710,
      .graph_used_high_bytes = 625,
      .graph_reserved_high_bytes = 790,
  };
  require(
      magpie_tts_rt::observed_cuda_graph_memory_growth(
          baseline, observed) == 125,
      "graph high-water growth was not retained");
}

void test_near_limit_budget_is_exact_and_does_not_underflow() {
  require(
      magpie_tts_rt::cuda_graph_memory_fits_budget(1'000, 975, 25),
      "an exact device-memory limit was rejected");
  require(
      !magpie_tts_rt::cuda_graph_memory_fits_budget(1'000, 975, 26),
      "one graph byte above the device-memory limit was accepted");
  require(
      !magpie_tts_rt::cuda_graph_memory_fits_budget(1'000, 1'001, 0),
      "explicit memory above the limit underflowed the remaining budget");
  require(
      !magpie_tts_rt::cuda_graph_memory_fits_budget(
          1'000, UINT64_MAX, UINT64_MAX),
      "saturated accounting values underflowed into an accepted budget");
}

}  // namespace

int main() {
  try {
    test_immutable_baseline_retains_fixed_graph_charge();
    test_high_water_and_free_memory_are_both_fail_closed();
    test_near_limit_budget_is_exact_and_does_not_underflow();
  } catch (const std::exception& error) {
    std::cerr << "CUDA graph memory accounting test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

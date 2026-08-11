#pragma once

#include <algorithm>
#include <cstdint>

namespace magpie_tts_rt {

struct CudaGraphMemorySnapshot {
  std::uint64_t free_bytes;
  std::uint64_t graph_used_bytes;
  std::uint64_t graph_reserved_bytes;
  std::uint64_t graph_used_high_bytes;
  std::uint64_t graph_reserved_high_bytes;
};

// Measures aggregate graph memory from one immutable pre-graph baseline.
// Reusing that baseline for later request-specific Main graph captures keeps
// the persistent Local AR and NanoCodec graphs in the charge. A per-request
// baseline would incorrectly report only the replacement Main graphs.
[[nodiscard]] inline std::uint64_t observed_cuda_graph_memory_growth(
    const CudaGraphMemorySnapshot& baseline,
    const CudaGraphMemorySnapshot& observed) noexcept {
  const auto positive_decrease = [](
                                     const std::uint64_t before,
                                     const std::uint64_t after) noexcept {
    return after < before ? before - after : 0U;
  };
  const auto positive_growth = [](
                                   const std::uint64_t before,
                                   const std::uint64_t after) noexcept {
    return after > before ? after - before : 0U;
  };
  return std::max(
      {positive_decrease(baseline.free_bytes, observed.free_bytes),
       positive_growth(
           baseline.graph_used_bytes, observed.graph_used_bytes),
       positive_growth(
           baseline.graph_reserved_bytes,
           observed.graph_reserved_bytes),
       positive_growth(
           baseline.graph_used_bytes,
           observed.graph_used_high_bytes),
       positive_growth(
           baseline.graph_reserved_bytes,
           observed.graph_reserved_high_bytes)});
}

// The subtraction is deliberately guarded. SessionWorkspace normally rejects
// an explicit allocation that reaches the manifest limit, but the final graph
// gate must remain fail-closed even if an upstream accounting regression hands
// it an already-over-budget value.
[[nodiscard]] inline bool cuda_graph_memory_fits_budget(
    const std::uint64_t maximum_bytes,
    const std::uint64_t explicit_session_bytes,
    const std::uint64_t observed_graph_bytes) noexcept {
  return explicit_session_bytes <= maximum_bytes &&
         observed_graph_bytes <=
             maximum_bytes - explicit_session_bytes;
}

}  // namespace magpie_tts_rt

#pragma once

namespace magpie_tts_rt {

enum class RequestWorkerFailureClass {
  context_exhausted_at_proven_quiescent_boundary,
  host_allocation_after_possible_enqueue,
  execution_state_unknown,
};

[[nodiscard]] constexpr bool request_failure_requires_session_poison(
    const RequestWorkerFailureClass failure) noexcept {
  return failure !=
         RequestWorkerFailureClass::
             context_exhausted_at_proven_quiescent_boundary;
}

}  // namespace magpie_tts_rt

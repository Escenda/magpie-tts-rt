#include "runtime/cuda_graph.hpp"

#include <utility>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const CudaGraphErrorCode code,
    const std::string_view detail) {
  return "CUDA graph failed [code=" +
         std::string(to_string(code)) + "]: " +
         std::string(detail);
}

[[noreturn]] void fail(
    const CudaGraphErrorCode code,
    const std::string& detail) {
  throw CudaGraphError(code, detail);
}

[[nodiscard]] std::string cuda_detail(
    const std::string_view operation,
    const cudaError_t status) {
  return std::string(operation) + ": " +
         cudaGetErrorString(status);
}

void close_failed_capture(const cudaStream_t stream) noexcept {
  cudaGraph_t abandoned = nullptr;
  static_cast<void>(cudaStreamEndCapture(stream, &abandoned));
  if (abandoned != nullptr) {
    static_cast<void>(cudaGraphDestroy(abandoned));
  }
}

#if defined(MAGPIE_TTS_RT_CUDA_GRAPH_TESTING)
testing::CudaGraphExecDestroyOperation graph_exec_destroy_operation =
    nullptr;
#endif

[[nodiscard]] cudaError_t destroy_graph_exec(
    const cudaGraphExec_t executable) noexcept {
#if defined(MAGPIE_TTS_RT_CUDA_GRAPH_TESTING)
  if (graph_exec_destroy_operation != nullptr) {
    return graph_exec_destroy_operation(executable);
  }
#endif
  return cudaGraphExecDestroy(executable);
}

}  // namespace

std::string_view to_string(
    const CudaGraphErrorCode code) noexcept {
  switch (code) {
    case CudaGraphErrorCode::invalid_state:
      return "invalid_state";
    case CudaGraphErrorCode::capture_begin_failed:
      return "capture_begin_failed";
    case CudaGraphErrorCode::captured_enqueue_failed:
      return "captured_enqueue_failed";
    case CudaGraphErrorCode::capture_end_failed:
      return "capture_end_failed";
    case CudaGraphErrorCode::instantiate_failed:
      return "instantiate_failed";
    case CudaGraphErrorCode::upload_failed:
      return "upload_failed";
    case CudaGraphErrorCode::launch_failed:
      return "launch_failed";
    case CudaGraphErrorCode::destroy_failed:
      return "destroy_failed";
  }
  return "unknown";
}

CudaGraphError::CudaGraphError(
    const CudaGraphErrorCode code,
    std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

CudaGraphErrorCode CudaGraphError::code() const noexcept {
  return code_;
}

const std::string& CudaGraphError::detail() const noexcept {
  return detail_;
}

CudaGraphExecutable::~CudaGraphExecutable() { reset(); }

bool CudaGraphExecutable::ready() const noexcept {
  return executable_ != nullptr;
}

void CudaGraphExecutable::capture_and_upload(
    const cudaStream_t stream,
    const EnqueueOperation& enqueue) {
  if (stream == nullptr || !enqueue) {
    fail(
        CudaGraphErrorCode::invalid_state,
        "a non-default stream and enqueue operation are required");
  }
  if (ready()) {
    fail(
        CudaGraphErrorCode::invalid_state,
        "an executable graph cannot be captured twice");
  }

  const cudaError_t begin_status = cudaStreamBeginCapture(
      stream, cudaStreamCaptureModeGlobal);
  if (begin_status != cudaSuccess) {
    fail(
        CudaGraphErrorCode::capture_begin_failed,
        cuda_detail("begin stream capture", begin_status));
  }

  bool enqueued = false;
  try {
    enqueued = enqueue(stream);
  } catch (...) {
    close_failed_capture(stream);
    throw;
  }
  if (!enqueued) {
    close_failed_capture(stream);
    fail(
        CudaGraphErrorCode::captured_enqueue_failed,
        "the captured enqueue operation returned false");
  }

  cudaGraph_t graph = nullptr;
  const cudaError_t end_status =
      cudaStreamEndCapture(stream, &graph);
  if (end_status != cudaSuccess || graph == nullptr) {
    if (graph != nullptr) {
      static_cast<void>(cudaGraphDestroy(graph));
    }
    fail(
        CudaGraphErrorCode::capture_end_failed,
        end_status == cudaSuccess
            ? "stream capture returned a null graph"
            : cuda_detail("end stream capture", end_status));
  }

  cudaGraphExec_t executable = nullptr;
  const cudaError_t instantiate_status =
      cudaGraphInstantiate(&executable, graph, 0);
  const cudaError_t destroy_status = cudaGraphDestroy(graph);
  if (instantiate_status != cudaSuccess || executable == nullptr) {
    if (executable != nullptr) {
      static_cast<void>(cudaGraphExecDestroy(executable));
    }
    fail(
        CudaGraphErrorCode::instantiate_failed,
        instantiate_status == cudaSuccess
            ? "graph instantiation returned a null executable"
            : cuda_detail(
                  "instantiate captured graph",
                  instantiate_status));
  }
  if (destroy_status != cudaSuccess) {
    static_cast<void>(cudaGraphExecDestroy(executable));
    fail(
        CudaGraphErrorCode::instantiate_failed,
        cuda_detail("destroy captured graph", destroy_status));
  }

  const cudaError_t upload_status =
      cudaGraphUpload(executable, stream);
  if (upload_status != cudaSuccess) {
    static_cast<void>(cudaGraphExecDestroy(executable));
    fail(
        CudaGraphErrorCode::upload_failed,
        cuda_detail("upload executable graph", upload_status));
  }
  executable_ = executable;
}

void CudaGraphExecutable::launch(
    const cudaStream_t stream) const {
  if (stream == nullptr || !ready()) {
    fail(
        CudaGraphErrorCode::invalid_state,
        "a ready executable graph and non-default stream are required");
  }
  const cudaError_t status =
      cudaGraphLaunch(executable_, stream);
  if (status != cudaSuccess) {
    fail(
        CudaGraphErrorCode::launch_failed,
        cuda_detail("launch executable graph", status));
  }
}

void CudaGraphExecutable::reset() noexcept {
  if (executable_ != nullptr) {
    static_cast<void>(destroy_graph_exec(executable_));
    executable_ = nullptr;
  }
}

void CudaGraphExecutable::reset_checked() {
  if (executable_ == nullptr) {
    return;
  }
  const cudaError_t status = destroy_graph_exec(executable_);
  if (status != cudaSuccess) {
    fail(
        CudaGraphErrorCode::destroy_failed,
        cuda_detail("destroy executable graph", status));
  }
  executable_ = nullptr;
}

#if defined(MAGPIE_TTS_RT_CUDA_GRAPH_TESTING)
namespace testing {

void set_cuda_graph_exec_destroy_operation(
    const CudaGraphExecDestroyOperation operation) noexcept {
  graph_exec_destroy_operation = operation;
}

}  // namespace testing
#endif

}  // namespace magpie_tts_rt

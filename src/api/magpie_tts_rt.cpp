#include "magpie_tts_rt/magpie_tts_rt.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <string_view>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace {

static_assert(
    NV_TENSORRT_MAJOR == 10,
    "MagpieTTS-RT ABI v1 requires TensorRT major version 10");
static_assert(
    CUDART_VERSION / 1000 == 13,
    "MagpieTTS-RT ABI v1 requires CUDA Runtime major version 13");

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity > Severity::kWARNING || message == nullptr) {
      return;
    }
    std::fprintf(stderr, "MagpieTTS-RT TensorRT: %s\n", message);
  }
};

}  // namespace

struct mtt_runtime {
  explicit mtt_runtime(const int32_t selected_device) : cuda_device_index(selected_device) {}

  TensorRtLogger logger;
  nvinfer1::IRuntime* tensorrt_runtime{nullptr};
  std::atomic<uint32_t> live_models{0};
  int32_t cuda_device_index;
};

namespace {

[[nodiscard]] bool has_valid_header(
    const uint32_t struct_size,
    const uint32_t abi_version,
    const uint32_t expected_size) noexcept {
  return struct_size == expected_size && abi_version == MTT_ABI_VERSION_1;
}

[[nodiscard]] bool error_is_usable(const mtt_error_v1_t* error) noexcept {
  return error == nullptr ||
         has_valid_header(error->struct_size, error->abi_version, sizeof(mtt_error_v1_t));
}

void write_error(
    mtt_error_v1_t* error,
    const mtt_status_t code,
    const mtt_error_stage_t stage,
    const std::string_view message) noexcept {
  if (error == nullptr ||
      !has_valid_header(error->struct_size, error->abi_version, sizeof(mtt_error_v1_t))) {
    return;
  }
  error->code = code;
  error->stage = stage;
  const auto copy_size =
      message.size() < MTT_ERROR_MESSAGE_CAPACITY - 1 ? message.size() : MTT_ERROR_MESSAGE_CAPACITY - 1;
  std::memcpy(error->message, message.data(), copy_size);
  error->message[copy_size] = '\0';
}

void clear_error(mtt_error_v1_t* error) noexcept {
  write_error(error, MTT_STATUS_OK, MTT_ERROR_STAGE_NONE, "");
}

[[nodiscard]] bool reserved_is_zero(const uint64_t (&values)[4]) noexcept {
  for (const uint64_t value : values) {
    if (value != 0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool is_valid_utf8(const std::string_view text) noexcept {
  std::size_t index = 0;
  while (index < text.size()) {
    const auto lead = static_cast<unsigned char>(text[index]);
    std::size_t continuation_count = 0;
    std::uint32_t code_point = 0;
    if (lead <= 0x7FU) {
      ++index;
      continue;
    }
    if ((lead & 0xE0U) == 0xC0U) {
      continuation_count = 1;
      code_point = lead & 0x1FU;
      if (code_point == 0) {
        return false;
      }
    } else if ((lead & 0xF0U) == 0xE0U) {
      continuation_count = 2;
      code_point = lead & 0x0FU;
    } else if ((lead & 0xF8U) == 0xF0U) {
      continuation_count = 3;
      code_point = lead & 0x07U;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index - 1) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const auto continuation =
          static_cast<unsigned char>(text[index + offset]);
      if ((continuation & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (continuation & 0x3FU);
    }
    if ((continuation_count == 1 && code_point < 0x80U) ||
        (continuation_count == 2 && code_point < 0x800U) ||
        (continuation_count == 3 && code_point < 0x10000U) ||
        code_point > 0x10FFFFU ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

[[nodiscard]] mtt_status_t reject_unusable_error(mtt_error_v1_t* error) noexcept {
  if (error_is_usable(error)) {
    return MTT_STATUS_OK;
  }
  return MTT_STATUS_ABI_MISMATCH;
}

[[nodiscard]] mtt_status_t unavailable(
    mtt_error_v1_t* error,
    const mtt_error_stage_t stage,
    const std::string_view message) noexcept {
  write_error(error, MTT_STATUS_UNAVAILABLE, stage, message);
  return MTT_STATUS_UNAVAILABLE;
}

[[nodiscard]] mtt_status_t runtime_create(
    const mtt_runtime_desc_v1_t* desc,
    mtt_runtime_t** runtime,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (runtime == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_RUNTIME, "runtime output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *runtime = nullptr;
  if (desc == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_RUNTIME, "runtime descriptor is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_runtime_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "runtime descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->cuda_device_index < 0 || desc->flags != 0 || !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_RUNTIME,
        "runtime descriptor requires a non-negative CUDA device and zero flags/reserved fields");
    return MTT_STATUS_INVALID_ARGUMENT;
  }

  auto* created = new (std::nothrow) mtt_runtime(desc->cuda_device_index);
  if (created == nullptr) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_RUNTIME,
        "failed to allocate runtime state");
    return MTT_STATUS_INTERNAL_ERROR;
  }

  int previous_cuda_device = 0;
  const cudaError_t get_device_status = cudaGetDevice(&previous_cuda_device);
  if (get_device_status != cudaSuccess) {
    delete created;
    write_error(
        error, MTT_STATUS_CUDA_ERROR, MTT_ERROR_STAGE_CUDA, cudaGetErrorString(get_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  const cudaError_t set_device_status = cudaSetDevice(desc->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    delete created;
    write_error(
        error, MTT_STATUS_CUDA_ERROR, MTT_ERROR_STAGE_CUDA, cudaGetErrorString(set_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  created->tensorrt_runtime = nvinfer1::createInferRuntime(created->logger);
  if (created->tensorrt_runtime == nullptr) {
    delete created;
    const cudaError_t restore_status = cudaSetDevice(previous_cuda_device);
    if (restore_status != cudaSuccess) {
      write_error(
          error,
          MTT_STATUS_CUDA_ERROR,
          MTT_ERROR_STAGE_CUDA,
          cudaGetErrorString(restore_status));
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_ENGINE_ERROR,
        MTT_ERROR_STAGE_TENSORRT,
        "TensorRT runtime creation failed");
    return MTT_STATUS_ENGINE_ERROR;
  }

  const cudaError_t restore_device_status = cudaSetDevice(previous_cuda_device);
  if (restore_device_status != cudaSuccess) {
    delete created->tensorrt_runtime;
    created->tensorrt_runtime = nullptr;
    delete created;
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(restore_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  *runtime = created;
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t runtime_destroy(
    mtt_runtime_t* runtime,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (runtime == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_RUNTIME, "runtime is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (runtime->live_models.load(std::memory_order_acquire) != 0) {
    write_error(
        error,
        MTT_STATUS_BUSY,
        MTT_ERROR_STAGE_RUNTIME,
        "runtime still owns live models");
    return MTT_STATUS_BUSY;
  }
  delete runtime->tensorrt_runtime;
  runtime->tensorrt_runtime = nullptr;
  delete runtime;
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t model_load(
    mtt_runtime_t* runtime,
    const mtt_model_desc_v1_t* desc,
    mtt_model_t** model,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (model == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_MODEL, "model output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *model = nullptr;
  if (runtime == nullptr || desc == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "runtime and model descriptor are required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_model_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "model descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->flags != 0 || desc->reserved_0 != 0 ||
      !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "model flags and reserved fields must be zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->bundle_path == nullptr || desc->bundle_path_length == 0) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "a non-empty bundle path is required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->bundle_path_length > MTT_MAX_BUNDLE_PATH_BYTES ||
      desc->bundle_path_length >
          static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "bundle path exceeds the ABI v1 byte limit");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  bool digest_is_all_zero = true;
  for (const uint8_t byte : desc->expected_manifest_sha256) {
    if (byte != 0) {
      digest_is_all_zero = false;
      break;
    }
  }
  if (digest_is_all_zero) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "expected manifest SHA-256 trust anchor must not be all zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  const std::string_view path(
      desc->bundle_path,
      static_cast<std::size_t>(desc->bundle_path_length));
  if (path.find('\0') != std::string_view::npos || !is_valid_utf8(path)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "bundle path must be valid UTF-8 without embedded NUL bytes");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  return unavailable(
      error,
      MTT_ERROR_STAGE_MODEL,
      "model loading is not implemented in the initial contract-only runtime");
}

[[nodiscard]] mtt_status_t model_destroy(
    mtt_model_t* model,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (model == nullptr) {
    write_error(error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_MODEL, "model is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  return unavailable(
      error,
      MTT_ERROR_STAGE_MODEL,
      "model destruction is unavailable because model loading is not implemented");
}

[[nodiscard]] mtt_status_t session_create(
    mtt_model_t* model,
    const mtt_session_desc_v1_t* desc,
    mtt_session_t** session,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (session == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_SESSION, "session output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *session = nullptr;
  if (model == nullptr || desc == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_SESSION,
        "model and session descriptor are required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_session_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "session descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->flags != 0 || desc->reserved_0 != 0 ||
      !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_SESSION,
        "session flags and reserved fields must be zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  return unavailable(
      error,
      MTT_ERROR_STAGE_SESSION,
      "session creation is not implemented in the initial contract-only runtime");
}

[[nodiscard]] mtt_status_t session_destroy(
    mtt_session_t* session,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (session == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_SESSION, "session is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  return unavailable(
      error,
      MTT_ERROR_STAGE_SESSION,
      "session destruction is unavailable because session creation is not implemented");
}

[[nodiscard]] mtt_status_t request_start(
    mtt_session_t* session,
    const mtt_request_desc_v1_t* desc,
    mtt_request_t** request,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (request == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_REQUEST, "request output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *request = nullptr;
  if (session == nullptr || desc == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "session and request descriptor are required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_request_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "request descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->text_token_ids == nullptr || desc->text_token_count == 0) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "a non-empty token sequence is required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->text_token_count > MTT_MAX_TEXT_TOKENS) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "token sequence exceeds the ABI v1 token limit");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->flags != 0 || desc->reserved_0 != 0 || !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "request flags and reserved fields must be zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  return unavailable(
      error,
      MTT_ERROR_STAGE_REQUEST,
      "request start is not implemented in the initial contract-only runtime");
}

[[nodiscard]] mtt_status_t request_operation_unavailable(
    mtt_request_t* request,
    mtt_error_v1_t* error,
    const std::string_view operation) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (request == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_REQUEST, "request is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  return unavailable(error, MTT_ERROR_STAGE_REQUEST, operation);
}

[[nodiscard]] mtt_status_t request_read_unavailable(
    mtt_request_t* request,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error,
    const std::string_view operation) noexcept {
  if (snapshot == nullptr) {
    if (error_is_usable(error)) {
      clear_error(error);
      write_error(
          error,
          MTT_STATUS_INVALID_ARGUMENT,
          MTT_ERROR_STAGE_REQUEST,
          "request snapshot is null");
    }
    return error_is_usable(error) ? MTT_STATUS_INVALID_ARGUMENT : MTT_STATUS_ABI_MISMATCH;
  }
  if (!has_valid_header(
          snapshot->struct_size, snapshot->abi_version, sizeof(mtt_request_snapshot_v1_t))) {
    if (error_is_usable(error)) {
      clear_error(error);
      write_error(
          error,
          MTT_STATUS_ABI_MISMATCH,
          MTT_ERROR_STAGE_ABI,
          "request snapshot does not match ABI version 1");
    }
    return MTT_STATUS_ABI_MISMATCH;
  }
  return request_operation_unavailable(request, error, operation);
}

[[nodiscard]] mtt_status_t request_poll(
    mtt_request_t* request,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error) noexcept {
  return request_read_unavailable(
      request, snapshot, error, "request poll");
}

[[nodiscard]] mtt_status_t request_wait(
    mtt_request_t* request,
    const uint64_t after_revision,
    const uint64_t timeout_nanoseconds,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error) noexcept {
  static_cast<void>(after_revision);
  static_cast<void>(timeout_nanoseconds);
  return request_read_unavailable(
      request, snapshot, error, "request wait");
}

[[nodiscard]] mtt_status_t request_cancel(
    mtt_request_t* request,
    mtt_error_v1_t* error) noexcept {
  return request_operation_unavailable(request, error, "request cancellation");
}

[[nodiscard]] mtt_status_t request_destroy(
    mtt_request_t* request,
    mtt_error_v1_t* error) noexcept {
  return request_operation_unavailable(request, error, "request destruction");
}

[[nodiscard]] mtt_status_t audio_acquire(
    mtt_request_t* request,
    mtt_audio_lease_v1_t* lease,
    mtt_error_v1_t* error) noexcept {
  if (lease == nullptr) {
    if (error_is_usable(error)) {
      clear_error(error);
      write_error(
          error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_REQUEST, "audio lease is null");
    }
    return error_is_usable(error) ? MTT_STATUS_INVALID_ARGUMENT : MTT_STATUS_ABI_MISMATCH;
  }
  if (!has_valid_header(lease->struct_size, lease->abi_version, sizeof(mtt_audio_lease_v1_t))) {
    if (error_is_usable(error)) {
      clear_error(error);
      write_error(
          error,
          MTT_STATUS_ABI_MISMATCH,
          MTT_ERROR_STAGE_ABI,
          "audio lease does not match ABI version 1");
    }
    return MTT_STATUS_ABI_MISMATCH;
  }
  return request_operation_unavailable(request, error, "audio acquisition");
}

[[nodiscard]] mtt_status_t audio_release(
    mtt_request_t* request,
    const uint64_t lease_id,
    mtt_error_v1_t* error) noexcept {
  if (lease_id == 0) {
    if (error_is_usable(error)) {
      clear_error(error);
      write_error(
          error,
          MTT_STATUS_INVALID_ARGUMENT,
          MTT_ERROR_STAGE_REQUEST,
          "audio lease identifier must be non-zero");
    }
    return error_is_usable(error) ? MTT_STATUS_INVALID_ARGUMENT : MTT_STATUS_ABI_MISMATCH;
  }
  return request_operation_unavailable(request, error, "audio release");
}

}  // namespace

extern "C" MTT_API mtt_status_t mtt_get_api(
    const uint32_t requested_abi_version,
    mtt_api_v1_t* api) {
  if (api == nullptr) {
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (requested_abi_version != MTT_ABI_VERSION_1 ||
      !has_valid_header(api->struct_size, api->abi_version, sizeof(mtt_api_v1_t))) {
    return MTT_STATUS_ABI_MISMATCH;
  }

  api->runtime_create = runtime_create;
  api->runtime_destroy = runtime_destroy;
  api->model_load = model_load;
  api->model_destroy = model_destroy;
  api->session_create = session_create;
  api->session_destroy = session_destroy;
  api->request_start = request_start;
  api->request_poll = request_poll;
  api->request_wait = request_wait;
  api->request_cancel = request_cancel;
  api->request_destroy = request_destroy;
  api->audio_acquire = audio_acquire;
  api->audio_release = audio_release;
  return MTT_STATUS_OK;
}

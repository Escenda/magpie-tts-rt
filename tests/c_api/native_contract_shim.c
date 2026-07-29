#include "magpie_tts_rt/magpie_tts_rt.h"

#include <stddef.h>
#include <string.h>

static int error_is_usable(const mtt_error_v1_t* error) {
  return error == NULL ||
         (error->struct_size == sizeof(*error) &&
          error->abi_version == MTT_ABI_VERSION_1);
}

static mtt_status_t write_unavailable(
    mtt_error_v1_t* error,
    mtt_error_stage_t stage) {
  static const char message[] = "native contract shim has no runtime";
  if (!error_is_usable(error)) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (error != NULL) {
    error->code = MTT_STATUS_UNAVAILABLE;
    error->stage = stage;
    memset(error->message, 0, sizeof(error->message));
    memcpy(error->message, message, sizeof(message));
  }
  return MTT_STATUS_UNAVAILABLE;
}

static mtt_status_t unavailable_runtime_create(
    const mtt_runtime_desc_v1_t* desc,
    mtt_runtime_t** runtime,
    mtt_error_v1_t* error) {
  (void)desc;
  if (!error_is_usable(error)) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (runtime != NULL) {
    *runtime = NULL;
  }
  return write_unavailable(error, MTT_ERROR_STAGE_RUNTIME);
}

static mtt_status_t unavailable_runtime_destroy(
    mtt_runtime_t* runtime,
    mtt_error_v1_t* error) {
  (void)runtime;
  return write_unavailable(error, MTT_ERROR_STAGE_RUNTIME);
}

static mtt_status_t unavailable_model_load(
    mtt_runtime_t* runtime,
    const mtt_model_desc_v1_t* desc,
    mtt_model_t** model,
    mtt_error_v1_t* error) {
  (void)runtime;
  (void)desc;
  if (!error_is_usable(error)) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (model != NULL) {
    *model = NULL;
  }
  return write_unavailable(error, MTT_ERROR_STAGE_MODEL);
}

static mtt_status_t unavailable_model_destroy(
    mtt_model_t* model,
    mtt_error_v1_t* error) {
  (void)model;
  return write_unavailable(error, MTT_ERROR_STAGE_MODEL);
}

static mtt_status_t unavailable_session_create(
    mtt_model_t* model,
    const mtt_session_desc_v1_t* desc,
    mtt_session_t** session,
    mtt_error_v1_t* error) {
  (void)model;
  (void)desc;
  if (!error_is_usable(error)) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (session != NULL) {
    *session = NULL;
  }
  return write_unavailable(error, MTT_ERROR_STAGE_SESSION);
}

static mtt_status_t unavailable_session_destroy(
    mtt_session_t* session,
    mtt_error_v1_t* error) {
  (void)session;
  return write_unavailable(error, MTT_ERROR_STAGE_SESSION);
}

static mtt_status_t unavailable_request_start(
    mtt_session_t* session,
    const mtt_request_desc_v1_t* desc,
    mtt_request_t** request,
    mtt_error_v1_t* error) {
  (void)session;
  (void)desc;
  if (!error_is_usable(error)) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (request != NULL) {
    *request = NULL;
  }
  return write_unavailable(error, MTT_ERROR_STAGE_REQUEST);
}

static mtt_status_t unavailable_request_poll(
    mtt_request_t* request,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error) {
  (void)request;
  (void)snapshot;
  return write_unavailable(error, MTT_ERROR_STAGE_REQUEST);
}

static mtt_status_t unavailable_request_wait(
    mtt_request_t* request,
    uint64_t after_revision,
    uint64_t timeout_nanoseconds,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error) {
  (void)request;
  (void)after_revision;
  (void)timeout_nanoseconds;
  (void)snapshot;
  return write_unavailable(error, MTT_ERROR_STAGE_REQUEST);
}

static mtt_status_t unavailable_request_operation(
    mtt_request_t* request,
    mtt_error_v1_t* error) {
  (void)request;
  return write_unavailable(error, MTT_ERROR_STAGE_REQUEST);
}

static mtt_status_t unavailable_audio_acquire(
    mtt_request_t* request,
    mtt_audio_lease_v1_t* lease,
    mtt_error_v1_t* error) {
  (void)request;
  (void)lease;
  return write_unavailable(error, MTT_ERROR_STAGE_REQUEST);
}

static mtt_status_t unavailable_audio_release(
    mtt_request_t* request,
    uint64_t lease_id,
    mtt_error_v1_t* error) {
  (void)request;
  (void)lease_id;
  return write_unavailable(error, MTT_ERROR_STAGE_REQUEST);
}

MTT_API mtt_status_t mtt_get_api(
    uint32_t requested_abi_version,
    mtt_api_v1_t* api) {
  if (api == NULL) {
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (requested_abi_version != MTT_ABI_VERSION_1 ||
      api->struct_size != sizeof(*api) ||
      api->abi_version != MTT_ABI_VERSION_1) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  api->runtime_create = unavailable_runtime_create;
  api->runtime_destroy = unavailable_runtime_destroy;
  api->model_load = unavailable_model_load;
  api->model_destroy = unavailable_model_destroy;
  api->session_create = unavailable_session_create;
  api->session_destroy = unavailable_session_destroy;
  api->request_start = unavailable_request_start;
  api->request_poll = unavailable_request_poll;
  api->request_wait = unavailable_request_wait;
  api->request_cancel = unavailable_request_operation;
  api->request_destroy = unavailable_request_operation;
  api->audio_acquire = unavailable_audio_acquire;
  api->audio_release = unavailable_audio_release;
  return MTT_STATUS_OK;
}

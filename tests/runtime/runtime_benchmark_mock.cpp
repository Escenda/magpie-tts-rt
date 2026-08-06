#include "magpie_tts_rt/magpie_tts_rt.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

struct mtt_runtime {
  std::uint32_t model_count = 0U;
};

struct mtt_model {
  mtt_runtime_t* runtime = nullptr;
  std::uint32_t session_count = 0U;
};

struct mtt_session {
  mtt_model_t* model = nullptr;
  std::uint32_t request_count = 0U;
};

struct mtt_request {
  mtt_session_t* session = nullptr;
  std::uint64_t token_count = 0U;
  std::uint64_t revision = 1U;
  std::uint64_t generated_frames = 4U;
  std::uint64_t committed_tokens = 0U;
  std::uint32_t next_chunk = 1U;
  mtt_request_state_t state = MTT_REQUEST_STATE_RUNNING;
  bool audio_available = true;
  bool lease_live = false;
  std::uint64_t live_lease_id = 0U;
  std::uint64_t sequence = 0U;
  std::uint64_t first_sample_index = 0U;
  std::uint32_t current_frames = 4U;
  bool current_first = true;
  bool current_final = false;
  std::vector<float> samples;
  std::vector<mtt_alignment_event_v1_t> alignment;
};

namespace {

std::uint64_t g_next_lease_id = 1U;

void set_error(
    mtt_error_v1_t* const error, const mtt_status_t status,
    const mtt_error_stage_t stage, const char* const message) {
  if (error == nullptr) {
    return;
  }
  error->code = status;
  error->stage = stage;
  std::memset(error->message, 0, sizeof(error->message));
  if (message != nullptr) {
    const auto length = std::min(
        std::strlen(message), sizeof(error->message) - 1U);
    std::memcpy(error->message, message, length);
  }
}

[[nodiscard]] mtt_status_t succeed(mtt_error_v1_t* const error) {
  set_error(error, MTT_STATUS_OK, MTT_ERROR_STAGE_NONE, nullptr);
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t request_control(
    mtt_error_v1_t* const error, const mtt_status_t status) {
  set_error(error, status, MTT_ERROR_STAGE_REQUEST, nullptr);
  return status;
}

[[nodiscard]] mtt_status_t invalid(
    mtt_error_v1_t* const error, const char* const message) {
  set_error(
      error, MTT_STATUS_INVALID_ARGUMENT,
      MTT_ERROR_STAGE_ABI, message);
  return MTT_STATUS_INVALID_ARGUMENT;
}

[[nodiscard]] bool valid_error(const mtt_error_v1_t* const error) {
  return error != nullptr &&
         error->struct_size == sizeof(mtt_error_v1_t) &&
         error->abi_version == MTT_ABI_VERSION_1;
}

void set_alignment(
    mtt_request_t* const request,
    const std::uint64_t lease_start,
    const std::uint32_t frame_count) {
  request->alignment.clear();
  const auto lease_end =
      lease_start +
      static_cast<std::uint64_t>(frame_count) * 1'024U;
  std::uint64_t boundary = lease_start + 2'048U;
  while (boundary <= lease_end &&
         request->committed_tokens < request->token_count) {
    ++request->committed_tokens;
    mtt_alignment_event_v1_t event{};
    event.struct_size = sizeof(event);
    event.abi_version = MTT_ABI_VERSION_1;
    event.sample_index = boundary;
    event.committed_text_tokens = request->committed_tokens;
    request->alignment.push_back(event);
    boundary += 2'048U;
  }
}

void prepare_chunk(
    mtt_request_t* const request, const std::uint32_t frame_count,
    const bool first, const bool final) {
  request->current_frames = frame_count;
  request->current_first = first;
  request->current_final = final;
  request->samples.assign(
      static_cast<std::size_t>(frame_count) * 1'024U, 0.125F);
  set_alignment(
      request, request->first_sample_index, frame_count);
  request->generated_frames +=
      first ? 0U : static_cast<std::uint64_t>(frame_count);
  request->audio_available = true;
  ++request->revision;
  if (final) {
    request->state = MTT_REQUEST_STATE_COMPLETED;
  }
}

void publish_next_chunk(mtt_request_t* const request) {
  if (request->state != MTT_REQUEST_STATE_RUNNING ||
      request->audio_available || request->lease_live) {
    return;
  }
  if (request->next_chunk == 1U) {
    prepare_chunk(request, 8U, false, false);
  } else {
    prepare_chunk(request, 1U, false, true);
  }
  ++request->next_chunk;
}

void fill_snapshot(
    const mtt_request_t* const request,
    mtt_request_snapshot_v1_t* const snapshot) {
  const auto caller_size = snapshot->struct_size;
  const auto caller_abi = snapshot->abi_version;
  *snapshot = {};
  snapshot->struct_size = caller_size;
  snapshot->abi_version = caller_abi;
  snapshot->revision = request->revision;
  snapshot->state = request->state;
  snapshot->available_audio_leases =
      request->audio_available ? 1U : 0U;
  snapshot->generated_codec_frames =
      request->generated_frames;
  snapshot->published_samples =
      request->generated_frames * 1'024U;
  snapshot->committed_text_tokens =
      request->committed_tokens;
  if (request->state == MTT_REQUEST_STATE_CANCELLED) {
    snapshot->terminal_status = MTT_STATUS_CANCELLED;
    snapshot->terminal_error_stage = MTT_ERROR_STAGE_REQUEST;
  } else {
    snapshot->terminal_status = MTT_STATUS_OK;
    snapshot->terminal_error_stage = MTT_ERROR_STAGE_NONE;
  }
}

[[nodiscard]] mtt_status_t runtime_create(
    const mtt_runtime_desc_v1_t* const desc,
    mtt_runtime_t** const runtime, mtt_error_v1_t* const error) {
  if (!valid_error(error) || desc == nullptr || runtime == nullptr ||
      desc->struct_size != sizeof(*desc) ||
      desc->abi_version != MTT_ABI_VERSION_1) {
    return invalid(error, "invalid runtime_create arguments");
  }
  *runtime = new (std::nothrow) mtt_runtime_t;
  if (*runtime == nullptr) {
    set_error(
        error, MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_RUNTIME, "allocation failed");
    return MTT_STATUS_INTERNAL_ERROR;
  }
  return succeed(error);
}

[[nodiscard]] mtt_status_t runtime_destroy(
    mtt_runtime_t* const runtime, mtt_error_v1_t* const error) {
  if (!valid_error(error) || runtime == nullptr ||
      runtime->model_count != 0U) {
    return invalid(error, "invalid runtime_destroy arguments");
  }
  delete runtime;
  return succeed(error);
}

[[nodiscard]] mtt_status_t model_load(
    mtt_runtime_t* const runtime,
    const mtt_model_desc_v1_t* const desc,
    mtt_model_t** const model, mtt_error_v1_t* const error) {
  if (!valid_error(error) || runtime == nullptr || desc == nullptr ||
      model == nullptr || desc->struct_size != sizeof(*desc) ||
      desc->abi_version != MTT_ABI_VERSION_1 ||
      desc->bundle_path == nullptr ||
      desc->bundle_path_length == 0U) {
    return invalid(error, "invalid model_load arguments");
  }
  *model = new (std::nothrow) mtt_model_t;
  if (*model == nullptr) {
    set_error(
        error, MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_MODEL, "allocation failed");
    return MTT_STATUS_INTERNAL_ERROR;
  }
  (*model)->runtime = runtime;
  ++runtime->model_count;
  return succeed(error);
}

[[nodiscard]] mtt_status_t model_destroy(
    mtt_model_t* const model, mtt_error_v1_t* const error) {
  if (!valid_error(error) || model == nullptr ||
      model->session_count != 0U) {
    return invalid(error, "invalid model_destroy arguments");
  }
  --model->runtime->model_count;
  delete model;
  return succeed(error);
}

[[nodiscard]] mtt_status_t model_get_info(
    mtt_model_t* const model, mtt_model_info_v1_t* const info,
    mtt_error_v1_t* const error) {
  if (!valid_error(error) || model == nullptr || info == nullptr ||
      info->struct_size != sizeof(*info) ||
      info->abi_version != MTT_ABI_VERSION_1) {
    return invalid(error, "invalid model_get_info arguments");
  }
  const auto caller_size = info->struct_size;
  const auto caller_abi = info->abi_version;
  *info = {};
  info->struct_size = caller_size;
  info->abi_version = caller_abi;
  info->tokenizer_vocabulary_size = 4'096U;
  info->text_embedding_rows = 4'098U;
  info->bos_token_id = 4'096U;
  info->eos_token_id = 4'097U;
  info->japanese_global_pad_token_id = 1U;
  info->maximum_text_tokens = 512U;
  info->maximum_audio_frames = 1'024U;
  info->sample_rate_hz = 22'050U;
  info->channels = 1U;
  info->pcm_format = MTT_PCM_FORMAT_F32_MONO;
  info->codec_frame_samples = 1'024U;
  info->initial_frames = 4U;
  info->steady_frames = 8U;
  info->tail_min_frames = 1U;
  info->tail_max_frames = 8U;
  std::fill(
      std::begin(info->tokenizer_identity_sha256),
      std::end(info->tokenizer_identity_sha256),
      static_cast<std::uint8_t>(0x11U));
  return succeed(error);
}

[[nodiscard]] mtt_status_t session_create(
    mtt_model_t* const model,
    const mtt_session_desc_v1_t* const desc,
    mtt_session_t** const session, mtt_error_v1_t* const error) {
  if (!valid_error(error) || model == nullptr || desc == nullptr ||
      session == nullptr || desc->struct_size != sizeof(*desc) ||
      desc->abi_version != MTT_ABI_VERSION_1) {
    return invalid(error, "invalid session_create arguments");
  }
  *session = new (std::nothrow) mtt_session_t;
  if (*session == nullptr) {
    set_error(
        error, MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_SESSION, "allocation failed");
    return MTT_STATUS_INTERNAL_ERROR;
  }
  (*session)->model = model;
  ++model->session_count;
  return succeed(error);
}

[[nodiscard]] mtt_status_t session_destroy(
    mtt_session_t* const session, mtt_error_v1_t* const error) {
  if (!valid_error(error) || session == nullptr ||
      session->request_count != 0U) {
    return invalid(error, "invalid session_destroy arguments");
  }
  --session->model->session_count;
  delete session;
  return succeed(error);
}

[[nodiscard]] mtt_status_t request_start(
    mtt_session_t* const session,
    const mtt_request_desc_v1_t* const desc,
    mtt_request_t** const request, mtt_error_v1_t* const error) {
  if (!valid_error(error) || session == nullptr || desc == nullptr ||
      request == nullptr || desc->struct_size != sizeof(*desc) ||
      desc->abi_version != MTT_ABI_VERSION_1 ||
      desc->text_token_ids == nullptr ||
      desc->text_token_count == 0U ||
      desc->text_token_count > MTT_MAX_TEXT_TOKENS) {
    return invalid(error, "invalid request_start arguments");
  }
  *request = new (std::nothrow) mtt_request_t;
  if (*request == nullptr) {
    set_error(
        error, MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST, "allocation failed");
    return MTT_STATUS_INTERNAL_ERROR;
  }
  (*request)->session = session;
  (*request)->token_count = desc->text_token_count;
  (*request)->samples.assign(4U * 1'024U, 0.25F);
  set_alignment(*request, 0U, 4U);
  ++session->request_count;
  return succeed(error);
}

[[nodiscard]] mtt_status_t request_poll(
    mtt_request_t* const request,
    mtt_request_snapshot_v1_t* const snapshot,
    mtt_error_v1_t* const error) {
  if (!valid_error(error) || request == nullptr ||
      snapshot == nullptr ||
      snapshot->struct_size != sizeof(*snapshot) ||
      snapshot->abi_version != MTT_ABI_VERSION_1) {
    return invalid(error, "invalid request_poll arguments");
  }
  fill_snapshot(request, snapshot);
  return succeed(error);
}

[[nodiscard]] mtt_status_t request_wait(
    mtt_request_t* const request, const std::uint64_t after_revision,
    const std::uint64_t,
    mtt_request_snapshot_v1_t* const snapshot,
    mtt_error_v1_t* const error) {
  if (!valid_error(error) || request == nullptr ||
      snapshot == nullptr ||
      snapshot->struct_size != sizeof(*snapshot) ||
      snapshot->abi_version != MTT_ABI_VERSION_1) {
    return invalid(error, "invalid request_wait arguments");
  }
  if (request->revision <= after_revision &&
      request->state == MTT_REQUEST_STATE_RUNNING) {
    publish_next_chunk(request);
  }
  if (request->revision <= after_revision) {
    return request_control(error, MTT_STATUS_TIMEOUT);
  }
  fill_snapshot(request, snapshot);
  return succeed(error);
}

[[nodiscard]] mtt_status_t request_cancel(
    mtt_request_t* const request, mtt_error_v1_t* const error) {
  if (!valid_error(error) || request == nullptr ||
      request->state != MTT_REQUEST_STATE_RUNNING) {
    return invalid(error, "invalid request_cancel arguments");
  }
  request->audio_available = false;
  request->state = MTT_REQUEST_STATE_CANCELLED;
  ++request->revision;
  return succeed(error);
}

[[nodiscard]] mtt_status_t request_destroy(
    mtt_request_t* const request, mtt_error_v1_t* const error) {
  if (!valid_error(error) || request == nullptr ||
      request->state == MTT_REQUEST_STATE_RUNNING ||
      request->lease_live) {
    return invalid(error, "invalid request_destroy arguments");
  }
  --request->session->request_count;
  delete request;
  return succeed(error);
}

[[nodiscard]] mtt_status_t audio_acquire(
    mtt_request_t* const request, mtt_audio_lease_v1_t* const lease,
    mtt_error_v1_t* const error) {
  if (!valid_error(error) || request == nullptr || lease == nullptr ||
      lease->struct_size != sizeof(*lease) ||
      lease->abi_version != MTT_ABI_VERSION_1 ||
      request->lease_live) {
    return invalid(error, "invalid audio_acquire arguments");
  }
  if (!request->audio_available) {
    return request_control(error, MTT_STATUS_WOULD_BLOCK);
  }
  if (g_next_lease_id ==
      std::numeric_limits<std::uint64_t>::max()) {
    set_error(
        error, MTT_STATUS_POISONED, MTT_ERROR_STAGE_REQUEST,
        "lease identifiers exhausted");
    return MTT_STATUS_POISONED;
  }
  const auto caller_size = lease->struct_size;
  const auto caller_abi = lease->abi_version;
  *lease = {};
  lease->struct_size = caller_size;
  lease->abi_version = caller_abi;
  lease->lease_id = g_next_lease_id++;
  lease->samples = request->samples.data();
  lease->sample_count = request->samples.size();
  lease->first_sample_index = request->first_sample_index;
  lease->sequence = request->sequence;
  lease->sample_rate_hz = 22'050U;
  lease->channels = 1U;
  lease->format = MTT_PCM_FORMAT_F32_MONO;
  lease->flags =
      (request->current_first ? MTT_AUDIO_FLAG_FIRST : 0U) |
      (request->current_final ? MTT_AUDIO_FLAG_FINAL : 0U);
  if (!request->alignment.empty()) {
    lease->flags |= MTT_AUDIO_FLAG_ALIGNMENT_VALID;
    lease->committed_text_tokens =
        request->committed_tokens;
    lease->alignment_events = request->alignment.data();
    lease->alignment_event_count = request->alignment.size();
  }
  request->audio_available = false;
  request->lease_live = true;
  request->live_lease_id = lease->lease_id;
  return succeed(error);
}

[[nodiscard]] mtt_status_t audio_release(
    mtt_request_t* const request, const std::uint64_t lease_id,
    mtt_error_v1_t* const error) {
  if (!valid_error(error) || request == nullptr ||
      !request->lease_live ||
      request->live_lease_id != lease_id) {
    return invalid(error, "invalid audio_release arguments");
  }
  request->lease_live = false;
  request->live_lease_id = 0U;
  request->first_sample_index += request->samples.size();
  ++request->sequence;
  ++request->revision;
  return succeed(error);
}

}  // namespace

extern "C" MTT_API mtt_status_t mtt_get_api(
    const std::uint32_t requested_abi_version,
    mtt_api_v1_t* const api) {
  if (requested_abi_version != MTT_ABI_VERSION_1 || api == nullptr ||
      api->struct_size != sizeof(*api) ||
      api->abi_version != MTT_ABI_VERSION_1) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  *api = {};
  api->struct_size = sizeof(*api);
  api->abi_version = MTT_ABI_VERSION_1;
  api->runtime_create = runtime_create;
  api->runtime_destroy = runtime_destroy;
  api->model_load = model_load;
  api->model_destroy = model_destroy;
  api->model_get_info = model_get_info;
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

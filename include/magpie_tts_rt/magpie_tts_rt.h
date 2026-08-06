#ifndef MAGPIE_TTS_RT_MAGPIE_TTS_RT_H_
#define MAGPIE_TTS_RT_MAGPIE_TTS_RT_H_

#include <stdint.h>

#if defined(_WIN32)
#if defined(MAGPIE_TTS_RT_BUILDING_LIBRARY)
#define MTT_API __declspec(dllexport)
#else
#define MTT_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define MTT_API __attribute__((visibility("default")))
#else
#define MTT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MTT_ABI_VERSION_1 UINT32_C(1)
#define MTT_ERROR_MESSAGE_CAPACITY UINT32_C(512)
#define MTT_SHA256_BYTES UINT32_C(32)
#define MTT_MAX_BUNDLE_PATH_BYTES UINT64_C(4096)
#define MTT_MAX_TEXT_TOKENS UINT64_C(512)

typedef struct mtt_runtime mtt_runtime_t;
typedef struct mtt_model mtt_model_t;
typedef struct mtt_session mtt_session_t;
typedef struct mtt_request mtt_request_t;

typedef int32_t mtt_status_t;

#define MTT_STATUS_OK INT32_C(0)
#define MTT_STATUS_INVALID_ARGUMENT INT32_C(1)
#define MTT_STATUS_ABI_MISMATCH INT32_C(2)
#define MTT_STATUS_BUSY INT32_C(3)
#define MTT_STATUS_IO_ERROR INT32_C(4)
#define MTT_STATUS_MANIFEST_ERROR INT32_C(5)
#define MTT_STATUS_RUNTIME_MISMATCH INT32_C(6)
#define MTT_STATUS_HASH_MISMATCH INT32_C(7)
#define MTT_STATUS_ENGINE_ERROR INT32_C(8)
#define MTT_STATUS_CUDA_ERROR INT32_C(9)
#define MTT_STATUS_CANCELLED INT32_C(10)
#define MTT_STATUS_WOULD_BLOCK INT32_C(11)
#define MTT_STATUS_TIMEOUT INT32_C(12)
#define MTT_STATUS_POISONED INT32_C(13)
#define MTT_STATUS_UNAVAILABLE INT32_C(14)
#define MTT_STATUS_INTERNAL_ERROR INT32_C(15)

typedef int32_t mtt_error_stage_t;

#define MTT_ERROR_STAGE_NONE INT32_C(0)
#define MTT_ERROR_STAGE_ABI INT32_C(1)
#define MTT_ERROR_STAGE_RUNTIME INT32_C(2)
#define MTT_ERROR_STAGE_MANIFEST INT32_C(3)
#define MTT_ERROR_STAGE_MODEL INT32_C(4)
#define MTT_ERROR_STAGE_SESSION INT32_C(5)
#define MTT_ERROR_STAGE_REQUEST INT32_C(6)
#define MTT_ERROR_STAGE_TENSORRT INT32_C(7)
#define MTT_ERROR_STAGE_CUDA INT32_C(8)
#define MTT_ERROR_STAGE_PLUGIN INT32_C(9)
#define MTT_ERROR_STAGE_ALIGNMENT INT32_C(10)
#define MTT_ERROR_STAGE_CODEC INT32_C(11)

typedef int32_t mtt_request_state_t;

#define MTT_REQUEST_STATE_RUNNING INT32_C(1)
#define MTT_REQUEST_STATE_COMPLETED INT32_C(2)
#define MTT_REQUEST_STATE_CANCELLED INT32_C(3)
#define MTT_REQUEST_STATE_FAILED INT32_C(4)

typedef int32_t mtt_pcm_format_t;

#define MTT_PCM_FORMAT_F32_MONO INT32_C(1)

#define MTT_AUDIO_FLAG_FIRST UINT32_C(1)
#define MTT_AUDIO_FLAG_FINAL UINT32_C(2)
#define MTT_AUDIO_FLAG_ALIGNMENT_VALID UINT32_C(4)

typedef struct mtt_error_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  mtt_status_t code;
  mtt_error_stage_t stage;
  char message[MTT_ERROR_MESSAGE_CAPACITY];
} mtt_error_v1_t;

typedef struct mtt_runtime_desc_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t cuda_device_index;
  uint32_t flags;
  uint64_t reserved[4];
} mtt_runtime_desc_v1_t;

typedef struct mtt_model_desc_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  /*
   * Copied before model_load returns. The path is a byte span and need not be
   * NUL-terminated.
   */
  const char* bundle_path;
  uint64_t bundle_path_length;
  /*
   * Required trust anchor from authenticated release metadata. The runtime
   * hashes the exact manifest snapshot it parses and requires byte equality.
   */
  uint8_t expected_manifest_sha256[MTT_SHA256_BYTES];
  uint32_t flags;
  uint32_t reserved_0;
  uint64_t reserved[4];
} mtt_model_desc_v1_t;

/*
 * Authenticated model properties needed by a text frontend and streaming
 * consumer. The caller initializes struct_size and abi_version; model_get_info
 * overwrites every other field on success.
 */
typedef struct mtt_model_info_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  /*
   * Normal tokenizer rows are [0, tokenizer_vocabulary_size). BOS and EOS
   * occupy separately authenticated special rows below text_embedding_rows.
   * A prepared request must end in exactly one eos_token_id; every preceding
   * token must be a normal tokenizer row. BOS is not a request token.
   */
  uint32_t tokenizer_vocabulary_size;
  uint32_t text_embedding_rows;
  uint32_t bos_token_id;
  uint32_t eos_token_id;
  uint32_t japanese_global_pad_token_id;
  uint32_t maximum_text_tokens;
  uint32_t maximum_audio_frames;
  uint32_t sample_rate_hz;
  uint32_t channels;
  mtt_pcm_format_t pcm_format;
  uint32_t codec_frame_samples;
  uint32_t initial_frames;
  uint32_t steady_frames;
  uint32_t tail_min_frames;
  uint32_t tail_max_frames;
  uint8_t tokenizer_identity_sha256[MTT_SHA256_BYTES];
  uint32_t reserved_0;
  uint64_t reserved[4];
} mtt_model_info_v1_t;

typedef struct mtt_session_desc_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t flags;
  uint32_t reserved_0;
  uint64_t reserved[4];
} mtt_session_desc_v1_t;

typedef struct mtt_request_desc_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  /*
   * Copied into request-owned storage before request_start returns. The
   * runtime never retains this caller-owned pointer. Every identifier must be
   * non-negative, fit INT32, and belong to the verified bundle vocabulary.
   */
  const int64_t* text_token_ids;
  uint64_t text_token_count;
  /* Accepted range is [0, 2^32); no truncation or modulo conversion. */
  uint64_t random_seed;
  uint32_t flags;
  uint32_t reserved_0;
  uint64_t reserved[4];
} mtt_request_desc_v1_t;

typedef struct mtt_request_snapshot_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t revision;
  mtt_request_state_t state;
  uint32_t available_audio_leases;
  uint64_t generated_codec_frames;
  uint64_t published_samples;
  uint64_t committed_text_tokens;
  /*
   * State/status/diagnostic relation:
   *   RUNNING or COMPLETED -> OK, NONE, empty message
   *   CANCELLED            -> CANCELLED, REQUEST, empty message
   *   FAILED               -> one of INVALID_ARGUMENT, ABI_MISMATCH,
   *     IO_ERROR, MANIFEST_ERROR, RUNTIME_MISMATCH, HASH_MISMATCH,
   *     ENGINE_ERROR, CUDA_ERROR, POISONED, UNAVAILABLE, or INTERNAL_ERROR;
   *     a declared non-NONE stage; and a non-empty message.
   * BUSY, WOULD_BLOCK, TIMEOUT, and CANCELLED are never FAILED statuses.
   * terminal_error_message is always NUL-terminated. A FAILED diagnostic is
   * retained unchanged in every later snapshot of the request.
   */
  mtt_status_t terminal_status;
  mtt_error_stage_t terminal_error_stage;
  char terminal_error_message[MTT_ERROR_MESSAGE_CAPACITY];
  uint64_t reserved[4];
} mtt_request_snapshot_v1_t;

/*
 * One monotonic text-alignment observation for generated audio. sample_index
 * is an utterance-global, end-exclusive played-sample boundary: text progress
 * becomes valid only after playback has drained every sample before this
 * index. It normally follows a two-frame Main Decoder step. When EOS
 * terminates the first frame of a step, the final event follows that one
 * emitted frame. committed_text_tokens is an end-exclusive position in the
 * request's prepared token sequence.
 */
typedef struct mtt_alignment_event_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t sample_index;
  uint64_t committed_text_tokens;
  uint64_t reserved[2];
} mtt_alignment_event_v1_t;

typedef struct mtt_audio_lease_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  /*
   * A nonzero identifier from one process-wide, strictly increasing sequence
   * shared by every runtime created through this loaded library. An identifier
   * is never reused, including after release or request destruction. Sequence
   * exhaustion fails closed with POISONED before returning another lease.
   * Callers must not assume adjacent values because sessions may run
   * concurrently.
  */
  uint64_t lease_id;
  /*
   * Non-null PCM storage when sample_count is nonzero. A zero sample_count
   * requires a null pointer and represents only a non-FIRST FINAL control
   * marker after prior PCM. That marker closes a stream when Local AR emits
   * EOS at frame zero; EOS itself is never passed to NanoCodec.
   */
  const float* samples;
  uint64_t sample_count;
  uint64_t first_sample_index;
  uint64_t sequence;
  uint32_t sample_rate_hz;
  uint32_t channels;
  mtt_pcm_format_t format;
  uint32_t flags;
  uint64_t committed_text_tokens;
  /*
   * Owned by the lease and valid through the successful audio_release call.
   * Events are ordered, lie after the lease start and at or before its
   * end-exclusive sample boundary, and include only strict advances in
   * committed_text_tokens. A zero count requires a null pointer. A zero-sample
   * FINAL control marker has no events and cannot advance text progress.
   */
  const mtt_alignment_event_v1_t* alignment_events;
  uint64_t alignment_event_count;
  uint64_t reserved[2];
} mtt_audio_lease_v1_t;

typedef struct mtt_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;

  mtt_status_t (*runtime_create)(
      const mtt_runtime_desc_v1_t* desc,
      mtt_runtime_t** runtime,
      mtt_error_v1_t* error);
  /*
   * On success, destroy operations that own CUDA resources leave the
   * runtime's CUDA device selected on the calling thread. A non-OK return
   * retains ownership and does not consume the handle.
   */
  mtt_status_t (*runtime_destroy)(
      mtt_runtime_t* runtime,
      mtt_error_v1_t* error);

  mtt_status_t (*model_load)(
      mtt_runtime_t* runtime,
      const mtt_model_desc_v1_t* desc,
      mtt_model_t** model,
      mtt_error_v1_t* error);
  /* Successful destruction leaves the owning runtime's CUDA device selected. */
  mtt_status_t (*model_destroy)(
      mtt_model_t* model,
      mtt_error_v1_t* error);
  mtt_status_t (*model_get_info)(
      mtt_model_t* model,
      mtt_model_info_v1_t* info,
      mtt_error_v1_t* error);

  mtt_status_t (*session_create)(
      mtt_model_t* model,
      const mtt_session_desc_v1_t* desc,
      mtt_session_t** session,
      mtt_error_v1_t* error);
  /* Successful destruction leaves the owning runtime's CUDA device selected. */
  mtt_status_t (*session_destroy)(
      mtt_session_t* session,
      mtt_error_v1_t* error);

  mtt_status_t (*request_start)(
      mtt_session_t* session,
      const mtt_request_desc_v1_t* desc,
      mtt_request_t** request,
      mtt_error_v1_t* error);
  mtt_status_t (*request_poll)(
      mtt_request_t* request,
      mtt_request_snapshot_v1_t* snapshot,
      mtt_error_v1_t* error);
  /*
   * Returns OK only with snapshot.revision > after_revision. A timeout,
   * including timeout_nanoseconds == 0 when no newer revision exists, returns
   * TIMEOUT and leaves snapshot unchanged.
   */
  mtt_status_t (*request_wait)(
      mtt_request_t* request,
      uint64_t after_revision,
      uint64_t timeout_nanoseconds,
      mtt_request_snapshot_v1_t* snapshot,
      mtt_error_v1_t* error);
  mtt_status_t (*request_cancel)(
      mtt_request_t* request,
      mtt_error_v1_t* error);
  mtt_status_t (*request_destroy)(
      mtt_request_t* request,
      mtt_error_v1_t* error);

  mtt_status_t (*audio_acquire)(
      mtt_request_t* request,
      mtt_audio_lease_v1_t* lease,
      mtt_error_v1_t* error);
  mtt_status_t (*audio_release)(
      mtt_request_t* request,
      /*
       * If lease_id identifies a live lease owned by request at call entry,
       * only OK consumes it and ends the samples lifetime. A non-OK return
       * leaves that live lease and PCM unchanged so the identical release may
       * be retried. Because identifiers are process-wide and never reused, a
       * stale, unknown, or cross-request identifier cannot alias a different
       * live lease and does not restore an already consumed buffer.
       */
      uint64_t lease_id,
      mtt_error_v1_t* error);
} mtt_api_v1_t;

MTT_API mtt_status_t mtt_get_api(
    uint32_t requested_abi_version,
    mtt_api_v1_t* api);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MAGPIE_TTS_RT_MAGPIE_TTS_RT_H_

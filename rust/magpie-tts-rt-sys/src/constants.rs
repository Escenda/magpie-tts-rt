// Numeric macros are checked against the public C header by
// tools/check_abi_constants.py. Keep each value as a literal so that check can
// reject missing, extra, or changed declarations.

pub const MTT_ABI_VERSION_1: u32 = 1;
pub const MTT_ERROR_MESSAGE_CAPACITY: u32 = 512;
pub const MTT_SHA256_BYTES: u32 = 32;
pub const MTT_MAX_BUNDLE_PATH_BYTES: u64 = 4096;
pub const MTT_MAX_TEXT_TOKENS: u64 = 512;

pub const MTT_STATUS_OK: mtt_status_t = 0;
pub const MTT_STATUS_INVALID_ARGUMENT: mtt_status_t = 1;
pub const MTT_STATUS_ABI_MISMATCH: mtt_status_t = 2;
pub const MTT_STATUS_BUSY: mtt_status_t = 3;
pub const MTT_STATUS_IO_ERROR: mtt_status_t = 4;
pub const MTT_STATUS_MANIFEST_ERROR: mtt_status_t = 5;
pub const MTT_STATUS_RUNTIME_MISMATCH: mtt_status_t = 6;
pub const MTT_STATUS_HASH_MISMATCH: mtt_status_t = 7;
pub const MTT_STATUS_ENGINE_ERROR: mtt_status_t = 8;
pub const MTT_STATUS_CUDA_ERROR: mtt_status_t = 9;
pub const MTT_STATUS_CANCELLED: mtt_status_t = 10;
pub const MTT_STATUS_WOULD_BLOCK: mtt_status_t = 11;
pub const MTT_STATUS_TIMEOUT: mtt_status_t = 12;
pub const MTT_STATUS_POISONED: mtt_status_t = 13;
pub const MTT_STATUS_UNAVAILABLE: mtt_status_t = 14;
pub const MTT_STATUS_INTERNAL_ERROR: mtt_status_t = 15;

pub const MTT_ERROR_STAGE_NONE: mtt_error_stage_t = 0;
pub const MTT_ERROR_STAGE_ABI: mtt_error_stage_t = 1;
pub const MTT_ERROR_STAGE_RUNTIME: mtt_error_stage_t = 2;
pub const MTT_ERROR_STAGE_MANIFEST: mtt_error_stage_t = 3;
pub const MTT_ERROR_STAGE_MODEL: mtt_error_stage_t = 4;
pub const MTT_ERROR_STAGE_SESSION: mtt_error_stage_t = 5;
pub const MTT_ERROR_STAGE_REQUEST: mtt_error_stage_t = 6;
pub const MTT_ERROR_STAGE_TENSORRT: mtt_error_stage_t = 7;
pub const MTT_ERROR_STAGE_CUDA: mtt_error_stage_t = 8;
pub const MTT_ERROR_STAGE_PLUGIN: mtt_error_stage_t = 9;
pub const MTT_ERROR_STAGE_ALIGNMENT: mtt_error_stage_t = 10;
pub const MTT_ERROR_STAGE_CODEC: mtt_error_stage_t = 11;

pub const MTT_REQUEST_STATE_RUNNING: mtt_request_state_t = 1;
pub const MTT_REQUEST_STATE_COMPLETED: mtt_request_state_t = 2;
pub const MTT_REQUEST_STATE_CANCELLED: mtt_request_state_t = 3;
pub const MTT_REQUEST_STATE_FAILED: mtt_request_state_t = 4;

pub const MTT_PCM_FORMAT_F32_MONO: mtt_pcm_format_t = 1;

pub const MTT_AUDIO_FLAG_FIRST: u32 = 1;
pub const MTT_AUDIO_FLAG_FINAL: u32 = 2;
pub const MTT_AUDIO_FLAG_ALIGNMENT_VALID: u32 = 4;

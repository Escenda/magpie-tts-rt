use std::mem::{align_of, size_of};
use std::ptr::NonNull;
use std::rc::Rc;
use std::time::Duration;

use crate::api::Api;
use crate::error::{Error, ErrorStage, NativeError, Result, Status};
use crate::sys;

const SAMPLE_RATE_HZ: u32 = 22_050;
const CHANNELS: u32 = 1;
const CODEC_FRAME_SAMPLES: u64 = 1_024;
const DECODER_STEP_SAMPLES: u64 = CODEC_FRAME_SAMPLES * 2;
const KNOWN_AUDIO_FLAGS: u32 =
    sys::MTT_AUDIO_FLAG_FIRST | sys::MTT_AUDIO_FLAG_FINAL | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    cuda_device_index: i32,
}

impl RuntimeConfig {
    pub fn new(cuda_device_index: i32) -> Result<Self> {
        if cuda_device_index < 0 {
            return Err(Error::InvalidInput {
                field: "cuda_device_index",
                reason: "must identify an explicit non-negative CUDA device",
            });
        }
        Ok(Self { cuda_device_index })
    }
}

struct RuntimeInner {
    api: Api,
    handle: Option<NonNull<sys::mtt_runtime_t>>,
    close_error: Option<Error>,
}

impl RuntimeInner {
    fn handle_ptr(&self) -> Result<*mut sys::mtt_runtime_t> {
        self.handle.map(NonNull::as_ptr).ok_or(Error::Closed {
            resource: "Runtime",
        })
    }

    fn try_close(&mut self) -> Result<()> {
        let Some(handle) = self.handle else {
            return Ok(());
        };
        self.close_error = None;
        let mut error = Api::error_buffer();
        let destroy = self
            .api
            .raw()
            .runtime_destroy
            .expect("validated runtime_destroy function");
        // SAFETY: handle is the live value returned by runtime_create.
        let status = unsafe { destroy(handle.as_ptr(), &mut error) };
        match Api::result_from_call("runtime_destroy", status, &error) {
            Ok(()) => {
                self.handle = None;
                Ok(())
            }
            Err(close_error) => {
                self.close_error = Some(close_error.clone());
                Err(close_error)
            }
        }
    }
}

impl Drop for RuntimeInner {
    fn drop(&mut self) {
        if let Some(close_error) = &self.close_error {
            abort_drop_failure("Runtime", close_error);
        }
        if let Err(error) = self.try_close() {
            abort_drop_failure("Runtime", &error);
        }
    }
}

pub struct Runtime {
    inner: Rc<RuntimeInner>,
}

impl Runtime {
    pub fn create(api: &Api, config: RuntimeConfig) -> Result<Self> {
        let desc = sys::mtt_runtime_desc_v1_t {
            struct_size: size_of::<sys::mtt_runtime_desc_v1_t>() as u32,
            abi_version: sys::MTT_ABI_VERSION_1,
            cuda_device_index: config.cuda_device_index,
            flags: 0,
            reserved: [0; 4],
        };
        let mut handle = std::ptr::null_mut();
        let mut error = Api::error_buffer();
        let create = api
            .raw()
            .runtime_create
            .expect("validated runtime_create function");
        // SAFETY: every pointer references a correctly initialized ABI v1
        // structure for the duration of the call.
        let status = unsafe { create(&desc, &mut handle, &mut error) };
        Api::result_from_call("runtime_create", status, &error)?;
        let handle = NonNull::new(handle).ok_or(Error::NullHandle {
            operation: "runtime_create",
        })?;

        Ok(Self {
            inner: Rc::new(RuntimeInner {
                api: api.clone(),
                handle: Some(handle),
                close_error: None,
            }),
        })
    }

    pub fn close(&mut self) -> Result<()> {
        let Some(inner) = Rc::get_mut(&mut self.inner) else {
            return Err(Error::ResourceInUse {
                resource: "Runtime",
            });
        };
        inner.try_close()
    }

    pub fn load_model(
        &self,
        bundle_path: &str,
        expected_manifest_sha256: [u8; sys::MTT_SHA256_BYTES as usize],
    ) -> Result<Model> {
        if bundle_path.is_empty() {
            return Err(Error::InvalidInput {
                field: "bundle_path",
                reason: "must not be empty",
            });
        }
        if bundle_path.as_bytes().contains(&0) {
            return Err(Error::InvalidInput {
                field: "bundle_path",
                reason: "must not contain an embedded NUL byte",
            });
        }
        if bundle_path.len() > sys::MTT_MAX_BUNDLE_PATH_BYTES as usize {
            return Err(Error::InvalidInput {
                field: "bundle_path",
                reason: "exceeds the ABI v1 byte limit",
            });
        }
        if expected_manifest_sha256.iter().all(|byte| *byte == 0) {
            return Err(Error::InvalidInput {
                field: "expected_manifest_sha256",
                reason: "must contain the authenticated manifest digest",
            });
        }
        let path_length = u64::try_from(bundle_path.len()).map_err(|_| Error::InvalidInput {
            field: "bundle_path",
            reason: "length does not fit the C ABI",
        })?;
        let mut handle = std::ptr::null_mut();
        let mut error = Api::error_buffer();
        let desc = sys::mtt_model_desc_v1_t {
            struct_size: size_of::<sys::mtt_model_desc_v1_t>() as u32,
            abi_version: sys::MTT_ABI_VERSION_1,
            bundle_path: bundle_path.as_ptr().cast(),
            bundle_path_length: path_length,
            expected_manifest_sha256,
            flags: 0,
            reserved_0: 0,
            reserved: [0; 4],
        };
        let load = self
            .inner
            .api
            .raw()
            .model_load
            .expect("validated model_load function");
        // SAFETY: the runtime is live, the UTF-8 byte pointer remains valid for
        // path_length bytes during this call, and output pointers are writable.
        let status = unsafe { load(self.inner.handle_ptr()?, &desc, &mut handle, &mut error) };
        Api::result_from_call("model_load", status, &error)?;
        let handle = NonNull::new(handle).ok_or(Error::NullHandle {
            operation: "model_load",
        })?;

        Ok(Model {
            inner: Rc::new(ModelInner {
                runtime: Rc::clone(&self.inner),
                handle: Some(handle),
                close_error: None,
            }),
        })
    }
}

struct ModelInner {
    runtime: Rc<RuntimeInner>,
    handle: Option<NonNull<sys::mtt_model_t>>,
    close_error: Option<Error>,
}

impl ModelInner {
    fn handle_ptr(&self) -> Result<*mut sys::mtt_model_t> {
        self.handle
            .map(NonNull::as_ptr)
            .ok_or(Error::Closed { resource: "Model" })
    }

    fn try_close(&mut self) -> Result<()> {
        let Some(handle) = self.handle else {
            return Ok(());
        };
        self.close_error = None;
        let mut error = Api::error_buffer();
        let destroy = self
            .runtime
            .api
            .raw()
            .model_destroy
            .expect("validated model_destroy function");
        // SAFETY: handle is the live value returned by model_load.
        let status = unsafe { destroy(handle.as_ptr(), &mut error) };
        match Api::result_from_call("model_destroy", status, &error) {
            Ok(()) => {
                self.handle = None;
                Ok(())
            }
            Err(close_error) => {
                self.close_error = Some(close_error.clone());
                Err(close_error)
            }
        }
    }
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        if let Some(close_error) = &self.close_error {
            abort_drop_failure("Model", close_error);
        }
        if let Err(error) = self.try_close() {
            abort_drop_failure("Model", &error);
        }
    }
}

pub struct Model {
    inner: Rc<ModelInner>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInfo {
    pub tokenizer_vocabulary_size: u32,
    pub text_embedding_rows: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub japanese_global_pad_token_id: u32,
    pub maximum_text_tokens: u32,
    pub maximum_audio_frames: u32,
    pub sample_rate_hz: u32,
    pub channels: u32,
    pub codec_frame_samples: u32,
    pub initial_frames: u32,
    pub steady_frames: u32,
    pub tail_min_frames: u32,
    pub tail_max_frames: u32,
    pub tokenizer_identity_sha256: [u8; sys::MTT_SHA256_BYTES as usize],
}

impl Model {
    pub fn info(&self) -> Result<ModelInfo> {
        let mut raw = sys::mtt_model_info_v1_t {
            struct_size: size_of::<sys::mtt_model_info_v1_t>() as u32,
            abi_version: sys::MTT_ABI_VERSION_1,
            tokenizer_vocabulary_size: 0,
            text_embedding_rows: 0,
            bos_token_id: 0,
            eos_token_id: 0,
            japanese_global_pad_token_id: 0,
            maximum_text_tokens: 0,
            maximum_audio_frames: 0,
            sample_rate_hz: 0,
            channels: 0,
            pcm_format: 0,
            codec_frame_samples: 0,
            initial_frames: 0,
            steady_frames: 0,
            tail_min_frames: 0,
            tail_max_frames: 0,
            tokenizer_identity_sha256: [0; sys::MTT_SHA256_BYTES as usize],
            reserved_0: 0,
            reserved: [0; 4],
        };
        let mut error = Api::error_buffer();
        let get_info = self
            .inner
            .runtime
            .api
            .raw()
            .model_get_info
            .expect("validated model_get_info function");
        // SAFETY: the model is live and raw/error are writable ABI v1 values.
        let status = unsafe { get_info(self.inner.handle_ptr()?, &mut raw, &mut error) };
        Api::result_from_call("model_get_info", status, &error)?;
        if raw.struct_size != size_of::<sys::mtt_model_info_v1_t>() as u32
            || raw.abi_version != sys::MTT_ABI_VERSION_1
            || raw.tokenizer_vocabulary_size == 0
            || raw.tokenizer_vocabulary_size > i32::MAX as u32
            || raw.text_embedding_rows <= raw.tokenizer_vocabulary_size
            || raw.text_embedding_rows > i32::MAX as u32
            || raw.bos_token_id < raw.tokenizer_vocabulary_size
            || raw.bos_token_id >= raw.text_embedding_rows
            || raw.eos_token_id < raw.tokenizer_vocabulary_size
            || raw.eos_token_id >= raw.text_embedding_rows
            || raw.bos_token_id == raw.eos_token_id
            || raw.japanese_global_pad_token_id >= raw.tokenizer_vocabulary_size
            || raw.maximum_text_tokens == 0
            || u64::from(raw.maximum_text_tokens) > sys::MTT_MAX_TEXT_TOKENS
            || raw.maximum_audio_frames == 0
            || raw.sample_rate_hz != SAMPLE_RATE_HZ
            || raw.channels != CHANNELS
            || raw.pcm_format != sys::MTT_PCM_FORMAT_F32_MONO
            || raw.codec_frame_samples as u64 != CODEC_FRAME_SAMPLES
            || raw.initial_frames != 4
            || raw.steady_frames != 8
            || raw.tail_min_frames != 1
            || raw.tail_max_frames != 8
            || raw.reserved_0 != 0
            || raw.reserved != [0; 4]
            || raw.tokenizer_identity_sha256.iter().all(|byte| *byte == 0)
        {
            return Err(Error::InvalidNativeData {
                operation: "model_get_info",
                field: "model_info",
                reason: "authenticated model properties violate the ABI v1 contract",
            });
        }
        Ok(ModelInfo {
            tokenizer_vocabulary_size: raw.tokenizer_vocabulary_size,
            text_embedding_rows: raw.text_embedding_rows,
            bos_token_id: raw.bos_token_id,
            eos_token_id: raw.eos_token_id,
            japanese_global_pad_token_id: raw.japanese_global_pad_token_id,
            maximum_text_tokens: raw.maximum_text_tokens,
            maximum_audio_frames: raw.maximum_audio_frames,
            sample_rate_hz: raw.sample_rate_hz,
            channels: raw.channels,
            codec_frame_samples: raw.codec_frame_samples,
            initial_frames: raw.initial_frames,
            steady_frames: raw.steady_frames,
            tail_min_frames: raw.tail_min_frames,
            tail_max_frames: raw.tail_max_frames,
            tokenizer_identity_sha256: raw.tokenizer_identity_sha256,
        })
    }

    pub fn close(&mut self) -> Result<()> {
        let Some(inner) = Rc::get_mut(&mut self.inner) else {
            return Err(Error::ResourceInUse { resource: "Model" });
        };
        inner.try_close()
    }

    pub fn create_session(&self) -> Result<Session> {
        let desc = sys::mtt_session_desc_v1_t {
            struct_size: size_of::<sys::mtt_session_desc_v1_t>() as u32,
            abi_version: sys::MTT_ABI_VERSION_1,
            flags: 0,
            reserved_0: 0,
            reserved: [0; 4],
        };
        let mut handle = std::ptr::null_mut();
        let mut error = Api::error_buffer();
        let create = self
            .inner
            .runtime
            .api
            .raw()
            .session_create
            .expect("validated session_create function");
        // SAFETY: the model is live and the descriptor/output pointers are valid.
        let status = unsafe { create(self.inner.handle_ptr()?, &desc, &mut handle, &mut error) };
        Api::result_from_call("session_create", status, &error)?;
        let handle = NonNull::new(handle).ok_or(Error::NullHandle {
            operation: "session_create",
        })?;

        Ok(Session {
            inner: Rc::new(SessionInner {
                model: Rc::clone(&self.inner),
                handle: Some(handle),
                close_error: None,
            }),
        })
    }
}

struct SessionInner {
    model: Rc<ModelInner>,
    handle: Option<NonNull<sys::mtt_session_t>>,
    close_error: Option<Error>,
}

impl SessionInner {
    fn handle_ptr(&self) -> Result<*mut sys::mtt_session_t> {
        self.handle.map(NonNull::as_ptr).ok_or(Error::Closed {
            resource: "Session",
        })
    }

    fn try_close(&mut self) -> Result<()> {
        let Some(handle) = self.handle else {
            return Ok(());
        };
        self.close_error = None;
        let mut error = Api::error_buffer();
        let destroy = self
            .model
            .runtime
            .api
            .raw()
            .session_destroy
            .expect("validated session_destroy function");
        // SAFETY: handle is the live value returned by session_create.
        let status = unsafe { destroy(handle.as_ptr(), &mut error) };
        match Api::result_from_call("session_destroy", status, &error) {
            Ok(()) => {
                self.handle = None;
                Ok(())
            }
            Err(close_error) => {
                self.close_error = Some(close_error.clone());
                Err(close_error)
            }
        }
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        if let Some(close_error) = &self.close_error {
            abort_drop_failure("Session", close_error);
        }
        if let Err(error) = self.try_close() {
            abort_drop_failure("Session", &error);
        }
    }
}

pub struct Session {
    inner: Rc<SessionInner>,
}

impl Session {
    pub fn close(&mut self) -> Result<()> {
        let Some(inner) = Rc::get_mut(&mut self.inner) else {
            return Err(Error::ResourceInUse {
                resource: "Session",
            });
        };
        inner.try_close()
    }

    pub fn start_request(&self, text_token_ids: &[i64], random_seed: u64) -> Result<Request> {
        if text_token_ids.is_empty() {
            return Err(Error::InvalidInput {
                field: "text_token_ids",
                reason: "must contain at least one prepared token",
            });
        }
        if text_token_ids.len() > sys::MTT_MAX_TEXT_TOKENS as usize {
            return Err(Error::InvalidInput {
                field: "text_token_ids",
                reason: "exceeds the ABI v1 token limit",
            });
        }
        if text_token_ids
            .iter()
            .any(|token| *token < 0 || *token > i64::from(i32::MAX))
        {
            return Err(Error::InvalidInput {
                field: "text_token_ids",
                reason: "every identifier must fit a non-negative INT32 value",
            });
        }
        if random_seed > u64::from(u32::MAX) {
            return Err(Error::InvalidInput {
                field: "random_seed",
                reason: "must be in [0, 2^32)",
            });
        }
        let token_count = u64::try_from(text_token_ids.len()).map_err(|_| Error::InvalidInput {
            field: "text_token_ids",
            reason: "length does not fit the C ABI",
        })?;
        let desc = sys::mtt_request_desc_v1_t {
            struct_size: size_of::<sys::mtt_request_desc_v1_t>() as u32,
            abi_version: sys::MTT_ABI_VERSION_1,
            text_token_ids: text_token_ids.as_ptr(),
            text_token_count: token_count,
            random_seed,
            flags: 0,
            reserved_0: 0,
            reserved: [0; 4],
        };
        let mut handle = std::ptr::null_mut();
        let mut error = Api::error_buffer();
        let start = self
            .inner
            .model
            .runtime
            .api
            .raw()
            .request_start
            .expect("validated request_start function");
        // SAFETY: the session is live, the token slice remains valid for this
        // call, and the descriptor/output pointers are initialized.
        let status = unsafe { start(self.inner.handle_ptr()?, &desc, &mut handle, &mut error) };
        Api::result_from_call("request_start", status, &error)?;
        let handle = NonNull::new(handle).ok_or(Error::NullHandle {
            operation: "request_start",
        })?;

        Ok(Request {
            session: Rc::clone(&self.inner),
            handle: Some(handle),
            token_count,
            last_snapshot: None,
            next_audio_position: None,
            last_audio_committed_tokens: None,
            last_audio_lease_id: None,
            final_audio_acquired: false,
            poisoned: false,
            close_error: None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestState {
    Running,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSnapshot {
    pub revision: u64,
    pub state: RequestState,
    pub available_audio_leases: u32,
    pub generated_codec_frames: u64,
    pub published_samples: u64,
    pub committed_text_tokens: u64,
    pub terminal_status: Status,
    pub terminal_error: Option<NativeError>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlignmentEvent {
    pub sample_index: u64,
    pub committed_text_tokens: u64,
}

#[must_use = "a running request must be explicitly cancelled/waited or completed before destruction"]
pub struct Request {
    session: Rc<SessionInner>,
    handle: Option<NonNull<sys::mtt_request_t>>,
    token_count: u64,
    last_snapshot: Option<RequestSnapshot>,
    next_audio_position: Option<(u64, u64)>,
    last_audio_committed_tokens: Option<u64>,
    last_audio_lease_id: Option<u64>,
    final_audio_acquired: bool,
    poisoned: bool,
    close_error: Option<Error>,
}

impl Request {
    pub fn poll(&mut self) -> Result<RequestSnapshot> {
        let mut raw = empty_snapshot();
        let mut error = Api::error_buffer();
        let poll = self
            .api()
            .raw()
            .request_poll
            .expect("validated request_poll function");
        // SAFETY: Request owns a live handle and both output structures are valid.
        let status = unsafe { poll(self.handle_ptr()?, &mut raw, &mut error) };
        Api::result_from_call("request_poll", status, &error)?;
        self.validate_snapshot("request_poll", raw, None)
    }

    pub fn wait_after(
        &mut self,
        after_revision: u64,
        timeout: Duration,
    ) -> Result<Option<RequestSnapshot>> {
        let timeout_nanoseconds =
            u64::try_from(timeout.as_nanos()).map_err(|_| Error::InvalidInput {
                field: "timeout",
                reason: "nanoseconds do not fit the C ABI",
            })?;
        let mut raw = empty_snapshot();
        let mut error = Api::error_buffer();
        let wait = self
            .api()
            .raw()
            .request_wait
            .expect("validated request_wait function");
        // SAFETY: Request owns a live handle and both output structures are valid.
        let raw_status = unsafe {
            wait(
                self.handle_ptr()?,
                after_revision,
                timeout_nanoseconds,
                &mut raw,
                &mut error,
            )
        };
        let status = Api::validate_call_status("request_wait", raw_status, &error)?;
        if status == Status::TIMEOUT {
            return Ok(None);
        }
        if !status.is_ok() {
            return Err(Api::error_from_validated_call(
                "request_wait",
                status,
                &error,
            ));
        }
        self.validate_snapshot("request_wait", raw, Some(after_revision))
            .map(Some)
    }

    pub fn cancel(&mut self) -> Result<()> {
        let mut error = Api::error_buffer();
        let cancel = self
            .api()
            .raw()
            .request_cancel
            .expect("validated request_cancel function");
        // SAFETY: Request owns a live handle. This is the only method that asks
        // the native runtime to cancel generation.
        let status = unsafe { cancel(self.handle_ptr()?, &mut error) };
        Api::result_from_call("request_cancel", status, &error)
    }

    pub fn acquire_audio(&mut self) -> Result<Option<AudioLease<'_>>> {
        let _ = self.handle_ptr()?;
        if self.final_audio_acquired {
            return Ok(None);
        }
        let mut raw = empty_audio_lease();
        let mut error = Api::error_buffer();
        let acquire = self
            .api()
            .raw()
            .audio_acquire
            .expect("validated audio_acquire function");
        // SAFETY: Request owns a live handle and both output structures are valid.
        let raw_status = unsafe { acquire(self.handle_ptr()?, &mut raw, &mut error) };
        let status = Api::validate_call_status("audio_acquire", raw_status, &error)?;
        if status == Status::WOULD_BLOCK {
            return Ok(None);
        }
        if !status.is_ok() {
            return Err(Api::error_from_validated_call(
                "audio_acquire",
                status,
                &error,
            ));
        }

        let validated_lease = self
            .validate_audio_lease(&raw)
            .and_then(|alignment_events| {
                let next_sequence =
                    raw.sequence
                        .checked_add(1)
                        .ok_or(Error::InvalidNativeData {
                            operation: "audio_acquire",
                            field: "sequence",
                            reason: "overflowed",
                        })?;
                let next_sample_index = raw
                    .first_sample_index
                    .checked_add(raw.sample_count)
                    .ok_or(Error::InvalidNativeData {
                        operation: "audio_acquire",
                        field: "sample range",
                        reason: "overflowed",
                    })?;
                Ok(((next_sequence, next_sample_index), alignment_events))
            });
        let (next_audio_position, alignment_events) = match validated_lease {
            Ok(validated) => validated,
            Err(lease_error) => {
                let release_result = self.release_audio_id(raw.lease_id);
                return match release_result {
                    Ok(()) => Err(lease_error),
                    Err(release_error) => {
                        self.poisoned = true;
                        Err(Error::InvalidAudioLeaseAndReleaseFailed {
                            lease_error: Box::new(lease_error),
                            release_error: Box::new(release_error),
                        })
                    }
                };
            }
        };

        self.next_audio_position = Some(next_audio_position);
        if raw.flags & sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID != 0 {
            self.last_audio_committed_tokens = Some(raw.committed_text_tokens);
        }
        self.final_audio_acquired = raw.flags & sys::MTT_AUDIO_FLAG_FINAL != 0;

        Ok(Some(AudioLease {
            request: self,
            raw,
            alignment_events,
            released: false,
            release_error: None,
        }))
    }

    /// Destroys a request explicitly.
    ///
    /// This operation never calls `request_cancel` or `request_wait`. Native
    /// `BUSY` is returned while ownership is retained, so the caller may
    /// explicitly cancel/wait and then call `close` again.
    pub fn close(&mut self) -> Result<()> {
        self.destroy_once()
    }

    fn api(&self) -> &Api {
        &self.session.model.runtime.api
    }

    fn handle_ptr(&self) -> Result<*mut sys::mtt_request_t> {
        if self.poisoned {
            return Err(Error::Poisoned {
                resource: "Request",
                reason: "an invalid native audio lease could not be released",
            });
        }
        self.handle.map(NonNull::as_ptr).ok_or(Error::Closed {
            resource: "Request",
        })
    }

    fn destroy_once(&mut self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Poisoned {
                resource: "Request",
                reason: "an invalid native audio lease could not be released",
            });
        }
        let Some(handle) = self.handle else {
            return Ok(());
        };
        self.close_error = None;
        let mut error = Api::error_buffer();
        let destroy = self
            .api()
            .raw()
            .request_destroy
            .expect("validated request_destroy function");
        // SAFETY: the handle came from request_start and remains owned by this
        // wrapper unless native destruction succeeds.
        let status = unsafe { destroy(handle.as_ptr(), &mut error) };
        match Api::result_from_call("request_destroy", status, &error) {
            Ok(()) => {
                self.handle = None;
                Ok(())
            }
            Err(close_error) => {
                self.close_error = Some(close_error.clone());
                Err(close_error)
            }
        }
    }

    fn release_audio_id(&mut self, lease_id: u64) -> Result<()> {
        let mut error = Api::error_buffer();
        let release = self
            .api()
            .raw()
            .audio_release
            .expect("validated audio_release function");
        // SAFETY: lease_id was returned for this request and is released once.
        let status = unsafe { release(self.handle_ptr()?, lease_id, &mut error) };
        Api::result_from_call("audio_release", status, &error)
    }

    fn validate_snapshot(
        &mut self,
        operation: &'static str,
        raw: sys::mtt_request_snapshot_v1_t,
        minimum_revision_exclusive: Option<u64>,
    ) -> Result<RequestSnapshot> {
        validate_header(
            operation,
            raw.struct_size,
            raw.abi_version,
            size_of::<sys::mtt_request_snapshot_v1_t>() as u32,
        )?;
        if raw.reserved != [0; 4] {
            return Err(invalid_native(
                operation,
                "reserved",
                "must be zero for ABI v1",
            ));
        }
        if raw.committed_text_tokens > self.token_count {
            return Err(invalid_native(
                operation,
                "committed_text_tokens",
                "exceeds request text_token_count",
            ));
        }
        if minimum_revision_exclusive.is_some_and(|minimum| raw.revision <= minimum) {
            return Err(invalid_native(
                operation,
                "revision",
                "must be greater than after_revision when wait succeeds",
            ));
        }

        let state = match raw.state {
            sys::MTT_REQUEST_STATE_RUNNING => RequestState::Running,
            sys::MTT_REQUEST_STATE_COMPLETED => RequestState::Completed,
            sys::MTT_REQUEST_STATE_CANCELLED => RequestState::Cancelled,
            sys::MTT_REQUEST_STATE_FAILED => RequestState::Failed,
            _ => {
                return Err(invalid_native(
                    operation,
                    "state",
                    "is not a declared ABI v1 request state",
                ));
            }
        };
        let terminal_status = Status::from_raw(raw.terminal_status);
        let terminal_error_stage = ErrorStage::from_raw(raw.terminal_error_stage);
        let (terminal_error_message, terminal_message_is_terminated) =
            fixed_message_bytes(&raw.terminal_error_message);
        if !terminal_error_stage.is_declared() {
            return Err(invalid_native(
                operation,
                "terminal_error_stage",
                "is not a declared ABI v1 error stage",
            ));
        }
        if !terminal_message_is_terminated {
            return Err(invalid_native(
                operation,
                "terminal_error_message",
                "must be NUL-terminated",
            ));
        }
        match state {
            RequestState::Running | RequestState::Completed => {
                if !terminal_status.is_ok() {
                    return Err(invalid_native(
                        operation,
                        "terminal_status",
                        "must be OK for RUNNING or COMPLETED",
                    ));
                }
                if terminal_error_stage != ErrorStage::NONE || !terminal_error_message.is_empty() {
                    return Err(invalid_native(
                        operation,
                        "terminal diagnostic",
                        "must be NONE with an empty message for RUNNING or COMPLETED",
                    ));
                }
            }
            RequestState::Cancelled => {
                if terminal_status != Status::CANCELLED {
                    return Err(invalid_native(
                        operation,
                        "terminal_status",
                        "must be CANCELLED for a cancelled request",
                    ));
                }
                if terminal_error_stage != ErrorStage::REQUEST || !terminal_error_message.is_empty()
                {
                    return Err(invalid_native(
                        operation,
                        "terminal diagnostic",
                        "must be REQUEST with an empty message for CANCELLED",
                    ));
                }
            }
            RequestState::Failed => {
                if !terminal_status.is_terminal_failure() {
                    return Err(invalid_native(
                        operation,
                        "terminal_status",
                        "must be a declared non-control failure status",
                    ));
                }
                if terminal_error_stage == ErrorStage::NONE {
                    return Err(invalid_native(
                        operation,
                        "terminal_error_stage",
                        "must be a declared non-NONE stage for FAILED",
                    ));
                }
                if terminal_error_message.is_empty() {
                    return Err(invalid_native(
                        operation,
                        "terminal_error_message",
                        "must be non-empty for FAILED",
                    ));
                }
            }
        }
        let terminal_error = (state == RequestState::Failed).then(|| {
            NativeError::new(
                "request_terminal",
                terminal_status,
                terminal_error_stage,
                terminal_error_message,
            )
        });

        let snapshot = RequestSnapshot {
            revision: raw.revision,
            state,
            available_audio_leases: raw.available_audio_leases,
            generated_codec_frames: raw.generated_codec_frames,
            published_samples: raw.published_samples,
            committed_text_tokens: raw.committed_text_tokens,
            terminal_status,
            terminal_error,
        };
        if let Some(previous) = &self.last_snapshot {
            if snapshot.revision < previous.revision
                || snapshot.generated_codec_frames < previous.generated_codec_frames
                || snapshot.published_samples < previous.published_samples
                || snapshot.committed_text_tokens < previous.committed_text_tokens
            {
                return Err(invalid_native(
                    operation,
                    "snapshot counters",
                    "must be monotonic within a request",
                ));
            }
            if snapshot.revision == previous.revision && &snapshot != previous {
                return Err(invalid_native(
                    operation,
                    "revision",
                    "may repeat only when the complete snapshot is unchanged",
                ));
            }
            if previous.state != RequestState::Running && snapshot.state != previous.state {
                return Err(invalid_native(
                    operation,
                    "state",
                    "must not transition away from a terminal state",
                ));
            }
            if previous.state == RequestState::Failed
                && (snapshot.terminal_status != previous.terminal_status
                    || snapshot.terminal_error != previous.terminal_error)
            {
                return Err(invalid_native(
                    operation,
                    "terminal diagnostic",
                    "must remain unchanged after FAILED",
                ));
            }
        }
        self.last_snapshot = Some(snapshot.clone());
        Ok(snapshot)
    }

    fn validate_audio_lease(
        &mut self,
        raw: &sys::mtt_audio_lease_v1_t,
    ) -> Result<Vec<AlignmentEvent>> {
        let operation = "audio_acquire";
        validate_header(
            operation,
            raw.struct_size,
            raw.abi_version,
            size_of::<sys::mtt_audio_lease_v1_t>() as u32,
        )?;
        if raw.reserved != [0; 2] {
            return Err(invalid_native(
                operation,
                "reserved",
                "must be zero for ABI v1",
            ));
        }
        if raw.lease_id == 0 {
            return Err(invalid_native(operation, "lease_id", "must be non-zero"));
        }
        if self
            .last_audio_lease_id
            .is_some_and(|previous| raw.lease_id <= previous)
        {
            return Err(invalid_native(
                operation,
                "lease_id",
                "must strictly increase and must never be reused",
            ));
        }
        self.last_audio_lease_id = Some(raw.lease_id);
        if raw.flags & !KNOWN_AUDIO_FLAGS != 0 {
            return Err(invalid_native(
                operation,
                "flags",
                "contains flags unknown to ABI v1",
            ));
        }
        let terminal_control_marker = raw.sample_count == 0;
        if terminal_control_marker {
            if !raw.samples.is_null() {
                return Err(invalid_native(
                    operation,
                    "samples",
                    "must be null for a zero-sample FINAL control marker",
                ));
            }
            if raw.flags & sys::MTT_AUDIO_FLAG_FINAL == 0
                || raw.flags & sys::MTT_AUDIO_FLAG_FIRST != 0
            {
                return Err(invalid_native(
                    operation,
                    "flags",
                    "zero samples require FINAL without FIRST",
                ));
            }
        } else {
            if raw.samples.is_null() {
                return Err(invalid_native(operation, "samples", "must not be null"));
            }
            if !raw.samples.addr().is_multiple_of(align_of::<f32>()) {
                return Err(invalid_native(
                    operation,
                    "samples",
                    "must be aligned for f32",
                ));
            }
            if !raw.sample_count.is_multiple_of(CODEC_FRAME_SAMPLES) {
                return Err(invalid_native(
                    operation,
                    "sample_count",
                    "must contain complete 1,024-sample codec frames",
                ));
            }
        }
        let sample_count = usize::try_from(raw.sample_count).map_err(|_| {
            invalid_native(operation, "sample_count", "does not fit this address space")
        })?;
        if sample_count > isize::MAX as usize / size_of::<f32>() {
            return Err(invalid_native(
                operation,
                "sample_count",
                "exceeds Rust slice size limits",
            ));
        }
        if raw.sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(invalid_native(
                operation,
                "sample_rate_hz",
                "must equal the ABI v1 MagpieTTS rate of 22,050 Hz",
            ));
        }
        if raw.channels != CHANNELS || raw.format != sys::MTT_PCM_FORMAT_F32_MONO {
            return Err(invalid_native(operation, "PCM layout", "must be mono f32"));
        }
        if raw.committed_text_tokens > self.token_count {
            return Err(invalid_native(
                operation,
                "committed_text_tokens",
                "exceeds request text_token_count",
            ));
        }
        let alignment_events = self.validate_alignment_events(raw)?;
        if let Some((expected_sequence, expected_sample_index)) = self.next_audio_position {
            if raw.sequence != expected_sequence || raw.first_sample_index != expected_sample_index
            {
                return Err(invalid_native(
                    operation,
                    "audio position",
                    "must be contiguous with the previous lease",
                ));
            }
            if raw.flags & sys::MTT_AUDIO_FLAG_FIRST != 0 {
                return Err(invalid_native(
                    operation,
                    "flags",
                    "FIRST may only appear on the first lease",
                ));
            }
        } else if raw.flags & sys::MTT_AUDIO_FLAG_FIRST == 0 || raw.first_sample_index != 0 {
            return Err(invalid_native(
                operation,
                "first lease",
                "must carry FIRST and begin at sample zero",
            ));
        }
        Ok(alignment_events)
    }

    fn validate_alignment_events(
        &self,
        raw: &sys::mtt_audio_lease_v1_t,
    ) -> Result<Vec<AlignmentEvent>> {
        let operation = "audio_acquire";
        const ALIGNMENT_VALID: u32 = sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
        if raw.flags & ALIGNMENT_VALID == 0 {
            if raw.committed_text_tokens != 0 {
                return Err(invalid_native(
                    operation,
                    "committed_text_tokens",
                    "must be zero when ALIGNMENT_VALID is absent",
                ));
            }
            if raw.alignment_event_count != 0 || !raw.alignment_events.is_null() {
                return Err(invalid_native(
                    operation,
                    "alignment_events",
                    "must be null and empty when ALIGNMENT_VALID is absent",
                ));
            }
            return Ok(Vec::new());
        }

        if self
            .last_audio_committed_tokens
            .is_some_and(|previous| raw.committed_text_tokens < previous)
        {
            return Err(invalid_native(
                operation,
                "committed_text_tokens",
                "must be monotonic across audio leases",
            ));
        }
        if raw.alignment_event_count == 0 {
            if !raw.alignment_events.is_null() {
                return Err(invalid_native(
                    operation,
                    "alignment_events",
                    "must be null when alignment_event_count is zero",
                ));
            }
            if raw.committed_text_tokens != self.last_audio_committed_tokens.unwrap_or(0) {
                return Err(invalid_native(
                    operation,
                    "committed_text_tokens",
                    "cannot advance without an alignment event",
                ));
            }
            return Ok(Vec::new());
        }
        if raw.alignment_events.is_null() {
            return Err(invalid_native(
                operation,
                "alignment_events",
                "must not be null when alignment_event_count is nonzero",
            ));
        }
        if !raw
            .alignment_events
            .addr()
            .is_multiple_of(align_of::<sys::mtt_alignment_event_v1_t>())
        {
            return Err(invalid_native(
                operation,
                "alignment_events",
                "must be aligned for mtt_alignment_event_v1_t",
            ));
        }
        let event_count = usize::try_from(raw.alignment_event_count).map_err(|_| {
            invalid_native(
                operation,
                "alignment_event_count",
                "does not fit this address space",
            )
        })?;
        if event_count > isize::MAX as usize / size_of::<sys::mtt_alignment_event_v1_t>() {
            return Err(invalid_native(
                operation,
                "alignment_event_count",
                "exceeds Rust slice size limits",
            ));
        }
        let maximum_event_count = raw.sample_count.div_ceil(DECODER_STEP_SAMPLES);
        if raw.alignment_event_count > maximum_event_count {
            return Err(invalid_native(
                operation,
                "alignment_event_count",
                "exceeds one event per Main Decoder step",
            ));
        }
        let sample_end = raw
            .first_sample_index
            .checked_add(raw.sample_count)
            .ok_or_else(|| invalid_native(operation, "sample range", "overflowed"))?;
        // SAFETY: the native ABI guarantees that a non-null, aligned event
        // region with this validated length remains alive until release.
        let native_events =
            unsafe { std::slice::from_raw_parts(raw.alignment_events, event_count) };
        let mut events = Vec::with_capacity(event_count);
        let mut previous_sample_index = None;
        let mut previous_committed = self.last_audio_committed_tokens.unwrap_or(0);
        for event in native_events {
            validate_header(
                operation,
                event.struct_size,
                event.abi_version,
                size_of::<sys::mtt_alignment_event_v1_t>() as u32,
            )?;
            if event.reserved != [0; 2] {
                return Err(invalid_native(
                    operation,
                    "alignment event reserved",
                    "must be zero for ABI v1",
                ));
            }
            let is_full_decoder_step_boundary =
                event.sample_index.is_multiple_of(DECODER_STEP_SAMPLES);
            let is_single_frame_terminal_boundary = raw.flags & sys::MTT_AUDIO_FLAG_FINAL != 0
                && event.sample_index == sample_end
                && event.sample_index.is_multiple_of(CODEC_FRAME_SAMPLES);
            if event.sample_index <= raw.first_sample_index
                || event.sample_index > sample_end
                || (!is_full_decoder_step_boundary && !is_single_frame_terminal_boundary)
            {
                return Err(invalid_native(
                    operation,
                    "alignment event sample_index",
                    "must follow a decoder step inside the lease or its single-frame EOS tail",
                ));
            }
            if previous_sample_index.is_some_and(|previous| event.sample_index <= previous) {
                return Err(invalid_native(
                    operation,
                    "alignment event sample_index",
                    "must strictly increase within a lease",
                ));
            }
            if event.committed_text_tokens <= previous_committed
                || event.committed_text_tokens > self.token_count
            {
                return Err(invalid_native(
                    operation,
                    "alignment event committed_text_tokens",
                    "must strictly advance without exceeding text_token_count",
                ));
            }
            events.push(AlignmentEvent {
                sample_index: event.sample_index,
                committed_text_tokens: event.committed_text_tokens,
            });
            previous_sample_index = Some(event.sample_index);
            previous_committed = event.committed_text_tokens;
        }
        if previous_committed != raw.committed_text_tokens {
            return Err(invalid_native(
                operation,
                "committed_text_tokens",
                "must equal the final in-lease alignment event",
            ));
        }
        Ok(events)
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        // request_destroy is deliberately the only native call here. Drop never
        // cancels, waits, or retries a prior explicit close failure.
        if let Some(close_error) = &self.close_error {
            abort_drop_failure("Request", close_error);
        }
        if let Err(error) = self.destroy_once() {
            abort_drop_failure("Request", &error);
        }
    }
}

pub struct AudioLease<'request> {
    request: &'request mut Request,
    raw: sys::mtt_audio_lease_v1_t,
    alignment_events: Vec<AlignmentEvent>,
    released: bool,
    release_error: Option<Error>,
}

impl AudioLease<'_> {
    pub fn samples(&self) -> Result<&[f32]> {
        if self.released {
            return Err(Error::Closed {
                resource: "AudioLease",
            });
        }
        let len = usize::try_from(self.raw.sample_count)
            .expect("sample_count was validated before AudioLease construction");
        if len == 0 {
            return Ok(&[]);
        }
        // SAFETY: the C ABI keeps this non-null region alive until audio_release.
        Ok(unsafe { std::slice::from_raw_parts(self.raw.samples, len) })
    }

    #[must_use]
    pub const fn lease_id(&self) -> u64 {
        self.raw.lease_id
    }

    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.raw.sequence
    }

    #[must_use]
    pub const fn first_sample_index(&self) -> u64 {
        self.raw.first_sample_index
    }

    #[must_use]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.raw.sample_rate_hz
    }

    #[must_use]
    pub const fn is_first(&self) -> bool {
        self.raw.flags & sys::MTT_AUDIO_FLAG_FIRST != 0
    }

    #[must_use]
    pub const fn is_final(&self) -> bool {
        self.raw.flags & sys::MTT_AUDIO_FLAG_FINAL != 0
    }

    #[must_use]
    pub const fn committed_text_tokens(&self) -> Option<u64> {
        if self.raw.flags & sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID != 0 {
            Some(self.raw.committed_text_tokens)
        } else {
            None
        }
    }

    #[must_use]
    pub fn alignment_events(&self) -> &[AlignmentEvent] {
        &self.alignment_events
    }

    pub fn release(&mut self) -> Result<()> {
        self.release_once()
    }

    fn release_once(&mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }
        self.release_error = None;
        match self.request.release_audio_id(self.raw.lease_id) {
            Ok(()) => {
                self.released = true;
                Ok(())
            }
            Err(release_error) => {
                self.release_error = Some(release_error.clone());
                Err(release_error)
            }
        }
    }
}

impl Drop for AudioLease<'_> {
    fn drop(&mut self) {
        if let Some(release_error) = &self.release_error {
            abort_drop_failure("AudioLease", release_error);
        }
        if let Err(error) = self.release_once() {
            abort_drop_failure("AudioLease", &error);
        }
    }
}

fn empty_snapshot() -> sys::mtt_request_snapshot_v1_t {
    sys::mtt_request_snapshot_v1_t {
        struct_size: size_of::<sys::mtt_request_snapshot_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        revision: 0,
        state: 0,
        available_audio_leases: 0,
        generated_codec_frames: 0,
        published_samples: 0,
        committed_text_tokens: 0,
        terminal_status: sys::MTT_STATUS_OK,
        terminal_error_stage: sys::MTT_ERROR_STAGE_NONE,
        terminal_error_message: [0; sys::MTT_ERROR_MESSAGE_CAPACITY as usize],
        reserved: [0; 4],
    }
}

fn empty_audio_lease() -> sys::mtt_audio_lease_v1_t {
    sys::mtt_audio_lease_v1_t {
        struct_size: size_of::<sys::mtt_audio_lease_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        lease_id: 0,
        samples: std::ptr::null(),
        sample_count: 0,
        first_sample_index: 0,
        sequence: 0,
        sample_rate_hz: 0,
        channels: 0,
        format: 0,
        flags: 0,
        committed_text_tokens: 0,
        alignment_events: std::ptr::null(),
        alignment_event_count: 0,
        reserved: [0; 2],
    }
}

fn fixed_message_bytes<const N: usize>(raw: &[core::ffi::c_char; N]) -> (Vec<u8>, bool) {
    let bytes: Vec<u8> = raw.iter().map(|byte| byte.to_ne_bytes()[0]).collect();
    match bytes.iter().position(|byte| *byte == 0) {
        Some(terminator) => (bytes[..terminator].to_vec(), true),
        None => (bytes, false),
    }
}

fn validate_header(
    operation: &'static str,
    struct_size: u32,
    abi_version: u32,
    expected_size: u32,
) -> Result<()> {
    if struct_size != expected_size {
        return Err(invalid_native(
            operation,
            "struct_size",
            "does not match the complete ABI v1 structure",
        ));
    }
    if abi_version != sys::MTT_ABI_VERSION_1 {
        return Err(invalid_native(
            operation,
            "abi_version",
            "does not equal ABI v1",
        ));
    }
    Ok(())
}

fn invalid_native(operation: &'static str, field: &'static str, reason: &'static str) -> Error {
    Error::InvalidNativeData {
        operation,
        field,
        reason,
    }
}

/// `Drop` cannot return a native destruction/release error. Continuing would
/// hide a live GPU/lease resource and violate the fail-closed contract, while
/// retrying, cancelling, or waiting would introduce implicit behavior.
fn abort_drop_failure(resource: &'static str, error: &Error) -> ! {
    eprintln!("MagpieTTS-RT fatal {resource} Drop failure: {error}");
    std::process::abort()
}

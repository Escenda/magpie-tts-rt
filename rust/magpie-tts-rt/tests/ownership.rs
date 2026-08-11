use std::mem::size_of;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use magpie_tts_rt::{
    AlignmentEvent, Api, Error, ErrorStage, InferenceWorker, RequestState, Runtime, RuntimeConfig,
    Status, SynthesisEvent, WorkerConfig, sys,
};

static TEST_LOCK: Mutex<()> = Mutex::new(());
static EVENTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
static REQUEST_DESTROY_FAILURES: AtomicUsize = AtomicUsize::new(0);
static AUDIO_RELEASE_FAILURES: AtomicUsize = AtomicUsize::new(0);
static AUDIO_ACQUIRE_MODE: AtomicUsize = AtomicUsize::new(0);
static AUDIO_ACQUIRE_CALLS: AtomicUsize = AtomicUsize::new(0);
static SNAPSHOT_MODE: AtomicUsize = AtomicUsize::new(0);
static SNAPSHOT_CALLS: AtomicUsize = AtomicUsize::new(0);
static LAST_WAIT_TIMEOUT_NANOSECONDS: AtomicU64 = AtomicU64::new(0);
static REQUEST_WAIT_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
static REQUEST_OPERATION_OVERLAP: AtomicBool = AtomicBool::new(false);
static CANCEL_MODE: AtomicUsize = AtomicUsize::new(0);
static REQUEST_START_MODE: AtomicUsize = AtomicUsize::new(0);
static AUDIO: [f32; 4_096] = [0.25; 4_096];
static ALIGNMENT_COMMITTED_ONE: [sys::mtt_alignment_event_v1_t; 1] =
    [sys::mtt_alignment_event_v1_t {
        struct_size: size_of::<sys::mtt_alignment_event_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        sample_index: 2_048,
        committed_text_tokens: 1,
        reserved: [0; 2],
    }];
static ALIGNMENT_COMMITTED_TWO: [sys::mtt_alignment_event_v1_t; 1] =
    [sys::mtt_alignment_event_v1_t {
        struct_size: size_of::<sys::mtt_alignment_event_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        sample_index: 2_048,
        committed_text_tokens: 2,
        reserved: [0; 2],
    }];
static ALIGNMENT_REGRESSED_ONE: [sys::mtt_alignment_event_v1_t; 1] =
    [sys::mtt_alignment_event_v1_t {
        struct_size: size_of::<sys::mtt_alignment_event_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        sample_index: AUDIO.len() as u64 + 2_048,
        committed_text_tokens: 1,
        reserved: [0; 2],
    }];
static ALIGNMENT_TERMINAL_ONE_FRAME: [sys::mtt_alignment_event_v1_t; 1] =
    [sys::mtt_alignment_event_v1_t {
        struct_size: size_of::<sys::mtt_alignment_event_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        sample_index: 1_024,
        committed_text_tokens: 1,
        reserved: [0; 2],
    }];
static ALIGNMENT_AT_LEASE_START: [sys::mtt_alignment_event_v1_t; 1] =
    [sys::mtt_alignment_event_v1_t {
        struct_size: size_of::<sys::mtt_alignment_event_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        sample_index: 0,
        committed_text_tokens: 1,
        reserved: [0; 2],
    }];

fn record(event: &'static str) {
    EVENTS.lock().expect("event lock poisoned").push(event);
}

unsafe fn allocate_handle<T>(output: *mut *mut T) {
    let allocation = Box::into_raw(Box::new(0_u8)).cast::<T>();
    // SAFETY: every mock create function receives a valid out pointer from the
    // safe wrapper, and the allocation remains live until its matching destroy.
    unsafe { output.write(allocation) };
}

unsafe fn destroy_handle<T>(handle: *mut T) {
    // SAFETY: each mock handle was allocated by allocate_handle and is destroyed
    // exactly once by the matching safe wrapper owner.
    unsafe { drop(Box::from_raw(handle.cast::<u8>())) };
}

unsafe fn write_error(
    error: *mut sys::mtt_error_v1_t,
    status: sys::mtt_status_t,
    stage: sys::mtt_error_stage_t,
) {
    // SAFETY: mock functions receive a writable initialized error structure.
    let error = unsafe { &mut *error };
    error.code = status;
    error.stage = stage;
    error.message = [0; sys::MTT_ERROR_MESSAGE_CAPACITY as usize];
    error.message[0] = b'm' as core::ffi::c_char;
}

unsafe fn write_control_error(error: *mut sys::mtt_error_v1_t, status: sys::mtt_status_t) {
    // SAFETY: mock functions receive a writable initialized error structure.
    let error = unsafe { &mut *error };
    error.code = status;
    error.stage = sys::MTT_ERROR_STAGE_REQUEST;
    error.message = [0; sys::MTT_ERROR_MESSAGE_CAPACITY as usize];
}

unsafe extern "C" fn runtime_create(
    _desc: *const sys::mtt_runtime_desc_v1_t,
    runtime: *mut *mut sys::mtt_runtime_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("runtime_create");
    // SAFETY: runtime is the valid output pointer supplied by Runtime::create.
    unsafe { allocate_handle(runtime) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn runtime_destroy(
    runtime: *mut sys::mtt_runtime_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("runtime_destroy");
    // SAFETY: runtime was allocated by runtime_create and is destroyed once.
    unsafe { destroy_handle(runtime) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn model_load(
    _runtime: *mut sys::mtt_runtime_t,
    _desc: *const sys::mtt_model_desc_v1_t,
    model: *mut *mut sys::mtt_model_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("model_load");
    // SAFETY: model is the valid output pointer supplied by Runtime::load_model.
    unsafe { allocate_handle(model) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn model_destroy(
    model: *mut sys::mtt_model_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("model_destroy");
    // SAFETY: model was allocated by model_load and is destroyed once.
    unsafe { destroy_handle(model) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn model_get_info(
    _model: *mut sys::mtt_model_t,
    info: *mut sys::mtt_model_info_v1_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("model_get_info");
    // SAFETY: info is the writable ABI v1 value supplied by Model::info.
    let info = unsafe { &mut *info };
    *info = sys::mtt_model_info_v1_t {
        struct_size: size_of::<sys::mtt_model_info_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        tokenizer_vocabulary_size: 3_357,
        text_embedding_rows: 3_359,
        bos_token_id: 3_357,
        eos_token_id: 3_358,
        japanese_global_pad_token_id: 1_015,
        maximum_text_tokens: 512,
        maximum_audio_frames: 1_024,
        sample_rate_hz: 22_050,
        channels: 1,
        pcm_format: sys::MTT_PCM_FORMAT_F32_MONO,
        codec_frame_samples: 1_024,
        initial_frames: 4,
        steady_frames: 8,
        tail_min_frames: 1,
        tail_max_frames: 8,
        tokenizer_identity_sha256: [1; sys::MTT_SHA256_BYTES as usize],
        reserved_0: 0,
        reserved: [0; 4],
    };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn session_create(
    _model: *mut sys::mtt_model_t,
    _desc: *const sys::mtt_session_desc_v1_t,
    session: *mut *mut sys::mtt_session_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("session_create");
    // SAFETY: session is the valid output pointer supplied by create_session.
    unsafe { allocate_handle(session) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn session_destroy(
    session: *mut sys::mtt_session_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("session_destroy");
    // SAFETY: session was allocated by session_create and is destroyed once.
    unsafe { destroy_handle(session) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn request_start(
    _session: *mut sys::mtt_session_t,
    _desc: *const sys::mtt_request_desc_v1_t,
    request: *mut *mut sys::mtt_request_t,
    error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_start");
    if REQUEST_START_MODE.load(Ordering::SeqCst) == 1 {
        // SAFETY: error is the valid buffer supplied by the wrapper.
        unsafe { write_error(error, sys::MTT_STATUS_CUDA_ERROR, sys::MTT_ERROR_STAGE_CUDA) };
        return sys::MTT_STATUS_CUDA_ERROR;
    }
    // SAFETY: request is the valid output pointer supplied by start_request.
    unsafe { allocate_handle(request) };
    sys::MTT_STATUS_OK
}

unsafe fn write_running_snapshot(snapshot: *mut sys::mtt_request_snapshot_v1_t, revision: u64) {
    // SAFETY: the mock poll/wait callers provide a writable initialized snapshot.
    let snapshot = unsafe { &mut *snapshot };
    snapshot.struct_size = size_of::<sys::mtt_request_snapshot_v1_t>() as u32;
    snapshot.abi_version = sys::MTT_ABI_VERSION_1;
    snapshot.revision = revision;
    snapshot.state = sys::MTT_REQUEST_STATE_RUNNING;
    snapshot.available_audio_leases = 1;
    snapshot.generated_codec_frames = 4;
    snapshot.published_samples = 4_096;
    snapshot.committed_text_tokens = 1;
    snapshot.terminal_status = sys::MTT_STATUS_OK;
    snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_NONE;
    snapshot.terminal_error_message = [0; sys::MTT_ERROR_MESSAGE_CAPACITY as usize];
    snapshot.reserved = [0; 4];
}

fn write_terminal_message(snapshot: &mut sys::mtt_request_snapshot_v1_t, message: &[u8]) {
    snapshot.terminal_error_message = [0; sys::MTT_ERROR_MESSAGE_CAPACITY as usize];
    for (target, source) in snapshot
        .terminal_error_message
        .iter_mut()
        .zip(message.iter().copied())
    {
        *target = source as core::ffi::c_char;
    }
}

unsafe extern "C" fn request_poll(
    _request: *mut sys::mtt_request_t,
    snapshot: *mut sys::mtt_request_snapshot_v1_t,
    error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_poll");
    let call_index = SNAPSHOT_CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: snapshot is the writable pointer supplied by Request::poll.
    unsafe { write_running_snapshot(snapshot, (call_index + 1) as u64) };
    // SAFETY: snapshot remains the writable output supplied by Request::poll.
    let snapshot = unsafe { &mut *snapshot };
    match SNAPSHOT_MODE.load(Ordering::SeqCst) {
        0 => {}
        1 if call_index > 0 => {
            snapshot.revision = 1;
            snapshot.generated_codec_frames += 1;
        }
        2 if call_index == 0 => snapshot.state = sys::MTT_REQUEST_STATE_COMPLETED,
        2 => {}
        3 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = sys::MTT_STATUS_CANCELLED;
        }
        4 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = sys::MTT_STATUS_TIMEOUT;
        }
        5 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = 10_000;
        }
        6 if call_index > 0 => snapshot.published_samples = 1,
        6 => {}
        8 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = sys::MTT_STATUS_CUDA_ERROR;
            snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_CUDA;
            write_terminal_message(snapshot, b"kernel launch failed");
        }
        9 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = sys::MTT_STATUS_CUDA_ERROR;
            write_terminal_message(snapshot, b"stage is missing");
        }
        10 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = sys::MTT_STATUS_CUDA_ERROR;
            snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_CUDA;
        }
        11 => {
            snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_CUDA;
            write_terminal_message(snapshot, b"running diagnostic");
        }
        12 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = sys::MTT_STATUS_CUDA_ERROR;
            snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_CUDA;
            if call_index == 0 {
                write_terminal_message(snapshot, b"first failure");
            } else {
                write_terminal_message(snapshot, b"changed failure");
            }
        }
        13 => {
            snapshot.state = sys::MTT_REQUEST_STATE_FAILED;
            snapshot.terminal_status = sys::MTT_STATUS_CUDA_ERROR;
            snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_CUDA;
            snapshot.terminal_error_message =
                [b'x' as core::ffi::c_char; sys::MTT_ERROR_MESSAGE_CAPACITY as usize];
        }
        14 => snapshot.state = sys::MTT_REQUEST_STATE_COMPLETED,
        15 => {
            // SAFETY: error is the valid buffer supplied by the wrapper.
            unsafe { write_error(error, sys::MTT_STATUS_CUDA_ERROR, sys::MTT_ERROR_STAGE_CUDA) };
            return sys::MTT_STATUS_CUDA_ERROR;
        }
        16 => {
            snapshot.state = sys::MTT_REQUEST_STATE_CANCELLED;
            snapshot.available_audio_leases = 0;
            snapshot.terminal_status = sys::MTT_STATUS_CANCELLED;
            snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_REQUEST;
        }
        _ => {}
    }
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn request_wait(
    _request: *mut sys::mtt_request_t,
    after_revision: u64,
    timeout_nanoseconds: u64,
    snapshot: *mut sys::mtt_request_snapshot_v1_t,
    error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_wait");
    LAST_WAIT_TIMEOUT_NANOSECONDS.store(timeout_nanoseconds, Ordering::SeqCst);
    if REQUEST_WAIT_IN_FLIGHT.swap(true, Ordering::SeqCst) {
        REQUEST_OPERATION_OVERLAP.store(true, Ordering::SeqCst);
    }
    let call_index = SNAPSHOT_CALLS.fetch_add(1, Ordering::SeqCst);
    let mode = SNAPSHOT_MODE.load(Ordering::SeqCst);
    if mode == 17 && call_index > 0 {
        thread::sleep(Duration::from_nanos(timeout_nanoseconds));
        // SAFETY: error is the writable initialized buffer supplied by the wrapper.
        unsafe { write_control_error(error, sys::MTT_STATUS_TIMEOUT) };
        REQUEST_WAIT_IN_FLIGHT.store(false, Ordering::SeqCst);
        return sys::MTT_STATUS_TIMEOUT;
    }

    const STALE_WAIT_MODE: usize = 7;
    let revision = if mode == STALE_WAIT_MODE {
        after_revision
    } else {
        after_revision.saturating_add(1)
    };
    // SAFETY: snapshot is the writable pointer supplied by Request::wait_after.
    unsafe { write_running_snapshot(snapshot, revision) };
    // SAFETY: snapshot remains the writable output supplied by Request::wait_after.
    let snapshot = unsafe { &mut *snapshot };
    if matches!(AUDIO_ACQUIRE_MODE.load(Ordering::SeqCst), 3 | 4 | 7) {
        snapshot.available_audio_leases = 2;
    }
    let status = match mode {
        14 => {
            snapshot.state = sys::MTT_REQUEST_STATE_COMPLETED;
            sys::MTT_STATUS_OK
        }
        15 if call_index > 0 => {
            // SAFETY: error is the valid buffer supplied by the wrapper.
            unsafe { write_error(error, sys::MTT_STATUS_CUDA_ERROR, sys::MTT_ERROR_STAGE_CUDA) };
            sys::MTT_STATUS_CUDA_ERROR
        }
        16 => {
            snapshot.state = sys::MTT_REQUEST_STATE_CANCELLED;
            snapshot.available_audio_leases = 0;
            snapshot.terminal_status = sys::MTT_STATUS_CANCELLED;
            snapshot.terminal_error_stage = sys::MTT_ERROR_STAGE_REQUEST;
            sys::MTT_STATUS_OK
        }
        17 => {
            snapshot.available_audio_leases = 0;
            snapshot.generated_codec_frames = 0;
            snapshot.published_samples = 0;
            snapshot.committed_text_tokens = 0;
            sys::MTT_STATUS_OK
        }
        _ => sys::MTT_STATUS_OK,
    };
    REQUEST_WAIT_IN_FLIGHT.store(false, Ordering::SeqCst);
    status
}

unsafe extern "C" fn request_cancel(
    _request: *mut sys::mtt_request_t,
    error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_cancel");
    if REQUEST_WAIT_IN_FLIGHT.load(Ordering::SeqCst) {
        REQUEST_OPERATION_OVERLAP.store(true, Ordering::SeqCst);
    }
    match CANCEL_MODE.load(Ordering::SeqCst) {
        0 => sys::MTT_STATUS_OK,
        1 => {
            // SAFETY: error is the valid buffer supplied by the wrapper.
            unsafe {
                write_error(
                    error,
                    sys::MTT_STATUS_INVALID_ARGUMENT,
                    sys::MTT_ERROR_STAGE_REQUEST,
                )
            };
            sys::MTT_STATUS_INVALID_ARGUMENT
        }
        2 => {
            // SAFETY: error is the valid buffer supplied by the wrapper.
            unsafe { write_error(error, sys::MTT_STATUS_CUDA_ERROR, sys::MTT_ERROR_STAGE_CUDA) };
            sys::MTT_STATUS_CUDA_ERROR
        }
        3 => {
            SNAPSHOT_MODE.store(16, Ordering::SeqCst);
            sys::MTT_STATUS_OK
        }
        4 => {
            SNAPSHOT_MODE.store(14, Ordering::SeqCst);
            // SAFETY: error is the valid buffer supplied by the wrapper.
            unsafe {
                write_error(
                    error,
                    sys::MTT_STATUS_INVALID_ARGUMENT,
                    sys::MTT_ERROR_STAGE_REQUEST,
                )
            };
            sys::MTT_STATUS_INVALID_ARGUMENT
        }
        unexpected => panic!("unknown request cancel mode {unexpected}"),
    }
}

unsafe extern "C" fn request_destroy(
    request: *mut sys::mtt_request_t,
    error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_destroy");
    if REQUEST_DESTROY_FAILURES
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
            remaining.checked_sub(1)
        })
        .is_ok()
    {
        // SAFETY: error is the valid buffer supplied by the wrapper.
        unsafe { write_error(error, sys::MTT_STATUS_BUSY, sys::MTT_ERROR_STAGE_REQUEST) };
        return sys::MTT_STATUS_BUSY;
    }
    // SAFETY: request was allocated by request_start and is destroyed once.
    unsafe { destroy_handle(request) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn audio_acquire(
    _request: *mut sys::mtt_request_t,
    lease: *mut sys::mtt_audio_lease_v1_t,
    error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("audio_acquire");
    if AUDIO_ACQUIRE_MODE.load(Ordering::SeqCst) == 8 {
        // SAFETY: error is the writable initialized buffer supplied by the
        // wrapper. WOULD_BLOCK carries a request-stage empty diagnostic.
        let error = unsafe { &mut *error };
        error.code = sys::MTT_STATUS_WOULD_BLOCK;
        error.stage = sys::MTT_ERROR_STAGE_REQUEST;
        error.message = [0; sys::MTT_ERROR_MESSAGE_CAPACITY as usize];
        return sys::MTT_STATUS_WOULD_BLOCK;
    }
    // SAFETY: lease is the writable pointer supplied by Request::acquire_audio.
    let lease = unsafe { &mut *lease };
    let call_index = AUDIO_ACQUIRE_CALLS.fetch_add(1, Ordering::SeqCst);
    lease.struct_size = size_of::<sys::mtt_audio_lease_v1_t>() as u32;
    lease.abi_version = sys::MTT_ABI_VERSION_1;
    lease.lease_id = 17_u64
        .checked_add(call_index as u64)
        .expect("test lease identifier overflow");
    lease.samples = AUDIO.as_ptr();
    lease.sample_count = AUDIO.len() as u64;
    lease.first_sample_index = 0;
    lease.sequence = 0;
    lease.sample_rate_hz = 22_050;
    lease.channels = 1;
    lease.format = sys::MTT_PCM_FORMAT_F32_MONO;
    lease.flags =
        sys::MTT_AUDIO_FLAG_FIRST | sys::MTT_AUDIO_FLAG_FINAL | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
    lease.committed_text_tokens = 1;
    lease.alignment_events = ALIGNMENT_COMMITTED_ONE.as_ptr();
    lease.alignment_event_count = ALIGNMENT_COMMITTED_ONE.len() as u64;
    lease.reserved = [0; 2];
    match AUDIO_ACQUIRE_MODE.load(Ordering::SeqCst) {
        0 => {}
        1 => lease.lease_id = 0,
        2 => lease.sequence = u64::MAX,
        3 => {
            if call_index == 0 {
                lease.flags = sys::MTT_AUDIO_FLAG_FIRST | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
                lease.committed_text_tokens = 2;
                lease.alignment_events = ALIGNMENT_COMMITTED_TWO.as_ptr();
                lease.alignment_event_count = ALIGNMENT_COMMITTED_TWO.len() as u64;
            } else {
                lease.sequence = 1;
                lease.first_sample_index = AUDIO.len() as u64;
                lease.flags = sys::MTT_AUDIO_FLAG_FINAL | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
                lease.committed_text_tokens = 1;
                lease.alignment_events = ALIGNMENT_REGRESSED_ONE.as_ptr();
                lease.alignment_event_count = ALIGNMENT_REGRESSED_ONE.len() as u64;
            }
        }
        4 => {
            if call_index == 0 {
                lease.flags = sys::MTT_AUDIO_FLAG_FIRST;
                lease.committed_text_tokens = 0;
                lease.alignment_events = std::ptr::null();
                lease.alignment_event_count = 0;
            } else {
                lease.lease_id = 17;
                lease.sequence = 1;
                lease.first_sample_index = AUDIO.len() as u64;
                lease.flags = sys::MTT_AUDIO_FLAG_FINAL;
                lease.committed_text_tokens = 0;
                lease.alignment_events = std::ptr::null();
                lease.alignment_event_count = 0;
            }
        }
        5 => {
            lease.sample_count = 1_024;
            lease.alignment_events = ALIGNMENT_TERMINAL_ONE_FRAME.as_ptr();
            lease.alignment_event_count = ALIGNMENT_TERMINAL_ONE_FRAME.len() as u64;
        }
        6 => {
            lease.alignment_events = ALIGNMENT_AT_LEASE_START.as_ptr();
            lease.alignment_event_count = ALIGNMENT_AT_LEASE_START.len() as u64;
        }
        7 => {
            if call_index == 0 {
                lease.flags = sys::MTT_AUDIO_FLAG_FIRST | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
            } else {
                lease.samples = std::ptr::null();
                lease.sample_count = 0;
                lease.first_sample_index = AUDIO.len() as u64;
                lease.sequence = 1;
                lease.flags = sys::MTT_AUDIO_FLAG_FINAL | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
                lease.alignment_events = std::ptr::null();
                lease.alignment_event_count = 0;
            }
        }
        unexpected => panic!("unknown audio acquire mode {unexpected}"),
    }
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn audio_release(
    _request: *mut sys::mtt_request_t,
    lease_id: u64,
    error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("audio_release");
    if lease_id == 0 {
        // SAFETY: error is the valid buffer supplied by the wrapper.
        unsafe {
            write_error(
                error,
                sys::MTT_STATUS_INVALID_ARGUMENT,
                sys::MTT_ERROR_STAGE_REQUEST,
            )
        };
        return sys::MTT_STATUS_INVALID_ARGUMENT;
    }
    if AUDIO_RELEASE_FAILURES
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
            remaining.checked_sub(1)
        })
        .is_ok()
    {
        // SAFETY: error is the valid buffer supplied by the wrapper.
        unsafe {
            write_error(
                error,
                sys::MTT_STATUS_INTERNAL_ERROR,
                sys::MTT_ERROR_STAGE_CODEC,
            )
        };
        return sys::MTT_STATUS_INTERNAL_ERROR;
    }
    sys::MTT_STATUS_OK
}

fn mock_table() -> sys::mtt_api_v1_t {
    sys::mtt_api_v1_t {
        struct_size: size_of::<sys::mtt_api_v1_t>() as u32,
        abi_version: sys::MTT_ABI_VERSION_1,
        runtime_create: Some(runtime_create),
        runtime_destroy: Some(runtime_destroy),
        model_load: Some(model_load),
        model_destroy: Some(model_destroy),
        model_get_info: Some(model_get_info),
        session_create: Some(session_create),
        session_destroy: Some(session_destroy),
        request_start: Some(request_start),
        request_poll: Some(request_poll),
        request_wait: Some(request_wait),
        request_cancel: Some(request_cancel),
        request_destroy: Some(request_destroy),
        audio_acquire: Some(audio_acquire),
        audio_release: Some(audio_release),
    }
}

unsafe extern "C" fn mock_get_api(
    requested_abi_version: u32,
    table: *mut sys::mtt_api_v1_t,
) -> sys::mtt_status_t {
    if requested_abi_version != sys::MTT_ABI_VERSION_1 || table.is_null() {
        return sys::MTT_STATUS_ABI_MISMATCH;
    }
    // SAFETY: Api::negotiate supplies writable storage for one complete v1
    // table and keeps every mock function alive for the test process.
    unsafe { table.write(mock_table()) };
    sys::MTT_STATUS_OK
}

fn api() -> Api {
    // SAFETY: every entry in mock_table has the exact generated C signature and
    // all functions remain present for the entire test process.
    unsafe { Api::from_table(mock_table()) }.expect("valid mock API")
}

fn clear_events() {
    EVENTS.lock().expect("event lock poisoned").clear();
    REQUEST_DESTROY_FAILURES.store(0, Ordering::SeqCst);
    AUDIO_RELEASE_FAILURES.store(0, Ordering::SeqCst);
    AUDIO_ACQUIRE_MODE.store(0, Ordering::SeqCst);
    AUDIO_ACQUIRE_CALLS.store(0, Ordering::SeqCst);
    SNAPSHOT_MODE.store(0, Ordering::SeqCst);
    SNAPSHOT_CALLS.store(0, Ordering::SeqCst);
    LAST_WAIT_TIMEOUT_NANOSECONDS.store(0, Ordering::SeqCst);
    REQUEST_WAIT_IN_FLIGHT.store(false, Ordering::SeqCst);
    REQUEST_OPERATION_OVERLAP.store(false, Ordering::SeqCst);
    CANCEL_MODE.store(0, Ordering::SeqCst);
    REQUEST_START_MODE.store(0, Ordering::SeqCst);
}

fn recorded_events() -> Vec<&'static str> {
    EVENTS.lock().expect("event lock poisoned").clone()
}

fn event_count(event: &str) -> usize {
    EVENTS
        .lock()
        .expect("event lock poisoned")
        .iter()
        .filter(|recorded| **recorded == event)
        .count()
}

fn wait_for_event_count(event: &str, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while event_count(event) < expected {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} recorded {event} events"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn worker_config() -> WorkerConfig {
    WorkerConfig::new(
        RuntimeConfig::new(0).expect("valid device"),
        "/verified/bundle",
        [1; 32],
    )
    .expect("valid worker configuration")
}

fn receive_completed_stream(stream: &magpie_tts_rt::SynthesisStream) {
    assert!(matches!(
        stream.recv().expect("audio event"),
        SynthesisEvent::Audio(_)
    ));
    assert_eq!(
        stream.recv().expect("completion event"),
        SynthesisEvent::Completed
    );
}

fn with_request(mode: usize, operation: impl FnOnce(&mut magpie_tts_rt::Request)) {
    clear_events();
    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");
    SNAPSHOT_MODE.store(mode, Ordering::SeqCst);
    operation(&mut request);
    request.close().expect("request destroy");
}

#[test]
fn inference_worker_owns_and_joins_the_complete_native_hierarchy() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(14, Ordering::SeqCst);
    let config = WorkerConfig::new(
        RuntimeConfig::new(0).expect("valid device"),
        "/verified/bundle",
        [1; 32],
    )
    .expect("valid worker configuration");
    // SAFETY: mock_get_api returns the complete static mock ABI table above,
    // whose functions and backing audio remain live for the test process.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, config) }
        .expect("inference worker");
    assert_eq!(worker.model_info().tokenizer_vocabulary_size, 3_357);
    assert_eq!(worker.model_info().text_embedding_rows, 3_359);
    assert_eq!(worker.model_info().eos_token_id, 3_358);
    assert_eq!(worker.model_info().maximum_text_tokens, 512);
    assert_eq!(worker.model_info().tokenizer_identity_sha256, [1; 32]);
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");
    match stream.recv().expect("audio event") {
        SynthesisEvent::Audio(chunk) => {
            assert_eq!(chunk.sequence, 0);
            assert_eq!(chunk.first_sample_index, 0);
            assert!(chunk.first);
            assert!(chunk.final_chunk);
            assert_eq!(chunk.samples, AUDIO);
        }
        unexpected => panic!("unexpected first event: {unexpected:?}"),
    }
    assert_eq!(
        stream.recv().expect("completion event"),
        SynthesisEvent::Completed
    );
    worker.shutdown().expect("joined worker shutdown");
    assert_eq!(
        recorded_events(),
        vec![
            "runtime_create",
            "model_load",
            "model_get_info",
            "session_create",
            "request_start",
            "request_wait",
            "audio_acquire",
            "audio_release",
            "request_destroy",
            "session_destroy",
            "model_destroy",
            "runtime_destroy",
        ]
    );
}

#[test]
fn inference_worker_waits_for_advertised_audio_instead_of_speculative_acquire() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(17, Ordering::SeqCst);
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");

    wait_for_event_count("request_wait", 3);
    assert_eq!(event_count("audio_acquire"), 0);
    assert_eq!(event_count("request_poll"), 0);
    assert_eq!(
        LAST_WAIT_TIMEOUT_NANOSECONDS.load(Ordering::SeqCst),
        2_000_000
    );

    SNAPSHOT_MODE.store(14, Ordering::SeqCst);
    receive_completed_stream(&stream);
    worker.shutdown().expect("joined worker shutdown");
    assert_eq!(event_count("audio_acquire"), 1);
}

#[test]
fn inference_worker_services_cancel_between_bounded_request_waits() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(17, Ordering::SeqCst);
    CANCEL_MODE.store(3, Ordering::SeqCst);
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");

    wait_for_event_count("request_wait", 2);
    let started = Instant::now();
    stream.cancel().expect("bounded cancellation");
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "cancel must be serviced after at most one bounded native wait"
    );
    assert_eq!(
        LAST_WAIT_TIMEOUT_NANOSECONDS.load(Ordering::SeqCst),
        2_000_000
    );
    assert!(!REQUEST_OPERATION_OVERLAP.load(Ordering::SeqCst));
    assert_eq!(
        stream.recv().expect("cancelled event"),
        SynthesisEvent::Cancelled
    );
    worker.shutdown().expect("joined worker shutdown");
}

#[test]
fn inference_worker_idle_shutdown_is_wake_driven() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");

    thread::sleep(Duration::from_millis(20));
    assert_eq!(event_count("request_poll"), 0);
    assert_eq!(event_count("request_wait"), 0);

    let (result_sender, result_receiver) = std::sync::mpsc::sync_channel(1);
    let shutdown_thread = thread::spawn(move || {
        result_sender
            .send(worker.shutdown())
            .expect("shutdown result receiver remains live");
    });
    assert_eq!(
        result_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("idle shutdown wake must be bounded"),
        Ok(())
    );
    shutdown_thread.join().expect("shutdown thread");
}

#[test]
fn inference_worker_idle_sender_disconnect_wakes_teardown() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");

    drop(worker);
    wait_for_event_count("runtime_destroy", 1);
    assert_eq!(event_count("request_poll"), 0);
    assert_eq!(event_count("request_wait"), 0);
    assert_eq!(event_count("request_cancel"), 0);
}

#[test]
fn inference_worker_validates_the_model_request_boundary_before_native_start() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");

    for (tokens, seed, field) in [
        (Vec::new(), 0, "text_token_ids"),
        (vec![1; 513], 0, "text_token_ids"),
        (vec![-1], 0, "text_token_ids"),
        (vec![3_357], 0, "text_token_ids"),
        (vec![1], 0, "text_token_ids"),
        (vec![1, 3_358, 3_358], 0, "text_token_ids"),
        (vec![1, 3_357, 3_358], 0, "text_token_ids"),
        (vec![1, 3_359, 3_358], 0, "text_token_ids"),
        (vec![1, 3_358], u64::from(u32::MAX) + 1, "random_seed"),
    ] {
        match worker.synthesize(tokens, seed) {
            Err(magpie_tts_rt::WorkerError::InvalidRequest { field: actual, .. }) => {
                assert_eq!(actual, field)
            }
            Err(unexpected) => {
                panic!("expected typed request validation error: {unexpected:?}")
            }
            Ok(_) => panic!("invalid request unexpectedly started"),
        }
    }
    assert_eq!(event_count("request_start"), 0);

    SNAPSHOT_MODE.store(14, Ordering::SeqCst);
    let stream = worker
        .synthesize(vec![1, 3_358], u64::from(u32::MAX))
        .expect("boundary-valid request");
    receive_completed_stream(&stream);
    worker.shutdown().expect("worker remains usable");
}

#[test]
fn inference_worker_fails_closed_after_native_request_start_failure() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    REQUEST_START_MODE.store(1, Ordering::SeqCst);
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");

    match worker.synthesize(vec![7, 3_358], 1) {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        Err(unexpected) => panic!("expected native start CUDA error: {unexpected:?}"),
        Ok(_) => panic!("native start failure unexpectedly returned a stream"),
    }
    match worker.shutdown() {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("worker must retain native start failure: {unexpected:?}"),
    }
    assert_eq!(event_count("request_destroy"), 0);
    assert_eq!(event_count("request_cancel"), 0);
}

#[test]
fn inference_worker_forwards_zero_sample_final_before_completed() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(14, Ordering::SeqCst);
    AUDIO_ACQUIRE_MODE.store(7, Ordering::SeqCst);
    let config = WorkerConfig::new(
        RuntimeConfig::new(0).expect("valid device"),
        "/verified/bundle",
        [1; 32],
    )
    .expect("valid worker configuration");
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, config) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");
    match stream.recv().expect("first audio event") {
        SynthesisEvent::Audio(chunk) => {
            assert!(chunk.first);
            assert!(!chunk.final_chunk);
            assert_eq!(chunk.samples, AUDIO);
        }
        unexpected => panic!("unexpected first event: {unexpected:?}"),
    }
    match stream.recv().expect("FINAL marker event") {
        SynthesisEvent::Audio(chunk) => {
            assert!(!chunk.first);
            assert!(chunk.final_chunk);
            assert_eq!(chunk.sequence, 1);
            assert_eq!(chunk.first_sample_index, AUDIO.len() as u64);
            assert!(chunk.samples.is_empty());
        }
        unexpected => panic!("unexpected FINAL marker event: {unexpected:?}"),
    }
    assert_eq!(
        stream.recv().expect("completion event"),
        SynthesisEvent::Completed
    );
    worker.shutdown().expect("joined worker shutdown");
}

#[test]
fn inference_worker_reports_one_typed_runtime_error_before_failing_closed() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(15, Ordering::SeqCst);
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");
    assert!(matches!(
        stream.recv().expect("audio event"),
        SynthesisEvent::Audio(_)
    ));
    match stream.recv().expect("typed runtime failure") {
        SynthesisEvent::RuntimeError(Error::Native(native)) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("unexpected runtime failure event: {unexpected:?}"),
    }
    assert_eq!(
        stream.recv(),
        Err(magpie_tts_rt::WorkerError::EventChannelClosed)
    );
    match worker.shutdown() {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("worker must retain its original failure: {unexpected:?}"),
    }
}

#[test]
fn disconnected_event_receiver_does_not_erase_the_worker_runtime_error() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(15, Ordering::SeqCst);
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");
    drop(stream);
    wait_for_event_count("request_wait", 2);
    match worker.shutdown() {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("worker must retain its original failure: {unexpected:?}"),
    }
}

#[test]
fn unread_audio_does_not_block_shutdown_after_an_injected_runtime_failure() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(15, Ordering::SeqCst);
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let _unread_stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");
    wait_for_event_count("request_wait", 2);

    let (result_sender, result_receiver) = std::sync::mpsc::sync_channel(1);
    let shutdown_thread = thread::spawn(move || {
        let result = worker.shutdown();
        result_sender
            .send(result)
            .expect("shutdown result receiver remains live");
    });
    match result_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("fatal worker shutdown must remain bounded")
    {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("worker must return the injected CUDA error: {unexpected:?}"),
    }
    shutdown_thread.join().expect("shutdown thread");
    assert_eq!(event_count("request_cancel"), 0);
}

#[test]
fn inference_worker_sender_disconnect_cancels_and_destroys_active_request() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(0, Ordering::SeqCst);
    AUDIO_ACQUIRE_MODE.store(8, Ordering::SeqCst);
    CANCEL_MODE.store(3, Ordering::SeqCst);
    // SAFETY: mock_get_api and its backing storage remain live for the test.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");

    let started = Instant::now();
    drop(worker);
    assert!(
        started.elapsed() < Duration::from_millis(50),
        "Drop must detach instead of joining the inference thread"
    );
    assert_eq!(event_count("request_cancel"), 0);

    drop(stream);
    wait_for_event_count("runtime_destroy", 1);
    assert_eq!(event_count("request_cancel"), 1);
    let events = recorded_events();
    let cancel = events
        .iter()
        .position(|event| *event == "request_cancel")
        .expect("disconnect must cancel the active request");
    let request_destroy = events
        .iter()
        .position(|event| *event == "request_destroy")
        .expect("cancelled request must be destroyed");
    let session_destroy = events
        .iter()
        .position(|event| *event == "session_destroy")
        .expect("session must be destroyed");
    assert!(cancel < request_destroy);
    assert!(request_destroy < session_destroy);
}

#[test]
fn inference_worker_shutdown_does_not_block_behind_a_full_event_channel() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(14, Ordering::SeqCst);
    let config = WorkerConfig::new(
        RuntimeConfig::new(0).expect("valid device"),
        "/verified/bundle",
        [1; 32],
    )
    .expect("valid worker configuration");
    // SAFETY: mock_get_api returns the complete static mock ABI table above,
    // whose functions and backing audio remain live for the test process.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, config) }
        .expect("inference worker");
    let _unread_stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");

    // Keep the capacity-one event queue full by deliberately not receiving its
    // audio chunk. Shutdown must detach that queue, finish the native request,
    // and join instead of waiting for the consumer.
    let (result_sender, result_receiver) = std::sync::mpsc::sync_channel(1);
    let shutdown_thread = thread::spawn(move || {
        let result = worker.shutdown();
        result_sender
            .send(result)
            .expect("shutdown result receiver remains live");
    });
    assert_eq!(
        result_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("shutdown must not block behind backpressure"),
        Ok(())
    );
    shutdown_thread.join().expect("shutdown thread");
    assert_eq!(
        recorded_events(),
        vec![
            "runtime_create",
            "model_load",
            "model_get_info",
            "session_create",
            "request_start",
            "request_wait",
            "audio_acquire",
            "audio_release",
            "request_destroy",
            "session_destroy",
            "model_destroy",
            "runtime_destroy",
        ]
    );
}

#[test]
fn inference_worker_cancel_is_idempotent_across_a_queued_native_terminal() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(17, Ordering::SeqCst);
    CANCEL_MODE.store(4, Ordering::SeqCst);
    // SAFETY: mock_get_api returns the complete static mock ABI table above,
    // whose functions and backing audio remain live for the test process.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");

    // Leave the first AUDIO event in the capacity-one channel. The worker then
    // handles Cancel before it can enqueue Completed. Native cancellation
    // reports INVALID_ARGUMENT because the request has just completed; the
    // typed terminal poll must make this cancellation idempotent.
    stream
        .cancel()
        .expect("cancel racing native completion is idempotent");
    assert_eq!(event_count("request_cancel"), 1);
    assert_eq!(event_count("request_poll"), 1);

    // Completed still cannot enter the full event channel, so this second
    // Cancel exercises the terminal_snapshot path and must not call native
    // cancellation again.
    stream
        .cancel()
        .expect("cancel after terminal snapshot is idempotent");
    assert_eq!(event_count("request_cancel"), 1);

    receive_completed_stream(&stream);
    wait_for_event_count("request_destroy", 1);

    // Once the terminal has been queued and the native request closed, only
    // that exact most-recent terminal identifier remains idempotent.
    stream
        .cancel()
        .expect("cancel after queued terminal is idempotent");
    assert_eq!(event_count("request_cancel"), 1);
    worker.shutdown().expect("joined worker shutdown");
}

#[test]
fn inference_worker_preserves_invalid_cancel_when_poll_proves_request_is_running() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(0, Ordering::SeqCst);
    CANCEL_MODE.store(1, Ordering::SeqCst);
    // SAFETY: mock_get_api returns the complete static mock ABI table above,
    // whose functions and backing audio remain live for the test process.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");

    match stream.cancel() {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::INVALID_ARGUMENT);
            assert_eq!(native.stage(), ErrorStage::REQUEST);
        }
        unexpected => panic!("expected the original typed cancel error, got {unexpected:?}"),
    }

    CANCEL_MODE.store(0, Ordering::SeqCst);
    SNAPSHOT_MODE.store(14, Ordering::SeqCst);
    stream.cancel().expect("cleanup cancellation");
    receive_completed_stream(&stream);
    worker.shutdown().expect("joined worker shutdown");
}

#[test]
fn inference_worker_treats_a_cuda_cancel_failure_as_fatal() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(0, Ordering::SeqCst);
    CANCEL_MODE.store(2, Ordering::SeqCst);
    // SAFETY: mock_get_api returns the complete static mock ABI table above,
    // whose functions and backing audio remain live for the test process.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");
    let stream = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("start synthesis");

    match stream.cancel() {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("expected the original CUDA cancel error, got {unexpected:?}"),
    }

    // Start acknowledgement and the first worker advance are intentionally
    // independent. An immediate Cancel may therefore fail before any audio is
    // queued, or after one already-produced chunk. In both schedules the
    // invariant is the same: the stream reports the typed fatal error and
    // closes without a successful terminal event.
    let fatal_event = match stream.recv().expect("audio or fatal stream error") {
        SynthesisEvent::Audio(_) => stream.recv().expect("fatal stream error"),
        event => event,
    };
    match fatal_event {
        SynthesisEvent::RuntimeError(Error::Native(native)) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("expected fatal CUDA stream error, got {unexpected:?}"),
    }
    assert_eq!(
        stream.recv(),
        Err(magpie_tts_rt::WorkerError::EventChannelClosed)
    );
    match worker.shutdown() {
        Err(magpie_tts_rt::WorkerError::Runtime(Error::Native(native))) => {
            assert_eq!(native.status(), Status::CUDA_ERROR);
            assert_eq!(native.stage(), ErrorStage::CUDA);
        }
        unexpected => panic!("worker must retain the CUDA cancel error: {unexpected:?}"),
    }
}

#[test]
fn inference_worker_does_not_accept_an_arbitrary_old_request_identifier() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    SNAPSHOT_MODE.store(14, Ordering::SeqCst);
    // SAFETY: mock_get_api returns the complete static mock ABI table above,
    // whose functions and backing audio remain live for the test process.
    let worker = unsafe { InferenceWorker::spawn_with_get_api(mock_get_api, worker_config()) }
        .expect("inference worker");

    let first = worker
        .synthesize(vec![7, 3_358], 1)
        .expect("first synthesis");
    receive_completed_stream(&first);
    wait_for_event_count("request_destroy", 1);

    let second = worker
        .synthesize(vec![8, 3_358], 2)
        .expect("second synthesis");
    receive_completed_stream(&second);
    wait_for_event_count("request_destroy", 2);

    assert_eq!(
        first.cancel(),
        Err(magpie_tts_rt::WorkerError::UnknownRequest(
            first.request_id()
        ))
    );
    second
        .cancel()
        .expect("most recent terminal identifier remains idempotent");
    worker.shutdown().expect("joined worker shutdown");
}

#[test]
fn owns_the_full_hierarchy_and_releases_an_audio_lease() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();

    let api = api();
    let mut runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let mut model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let mut session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[11, 12, 3_358], 99)
        .expect("request start");

    let snapshot = request.poll().expect("request poll");
    assert_eq!(snapshot.state, RequestState::Running);
    assert_eq!(snapshot.terminal_status, Status::OK);
    assert_eq!(snapshot.terminal_error, None);

    let mut lease = request
        .acquire_audio()
        .expect("audio acquire")
        .expect("audio available");
    assert_eq!(lease.samples().expect("live lease samples"), &AUDIO);
    assert_eq!(lease.committed_text_tokens(), Some(1));
    assert_eq!(
        lease.alignment_events(),
        [AlignmentEvent {
            sample_index: 2_048,
            committed_text_tokens: 1,
        }]
    );
    lease.release().expect("audio release");
    assert_eq!(
        lease
            .samples()
            .expect_err("released PCM must be inaccessible"),
        Error::Closed {
            resource: "AudioLease"
        }
    );
    drop(lease);

    request.close().expect("request destroy");
    assert_eq!(
        request
            .acquire_audio()
            .err()
            .expect("a closed request must not hide behind the final-audio flag"),
        Error::Closed {
            resource: "Request"
        }
    );
    drop(request);
    session.close().expect("session destroy");
    drop(session);
    model.close().expect("model destroy");
    drop(model);
    runtime.close().expect("runtime destroy");
    drop(runtime);

    assert_eq!(
        recorded_events(),
        [
            "runtime_create",
            "model_load",
            "session_create",
            "request_start",
            "request_poll",
            "audio_acquire",
            "audio_release",
            "request_destroy",
            "session_destroy",
            "model_destroy",
            "runtime_destroy",
        ]
    );
}

#[test]
fn request_drop_destroys_but_never_cancels_or_waits() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();

    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    drop(request);
    drop(session);
    drop(model);
    drop(runtime);

    let events = recorded_events();
    assert!(events.contains(&"request_destroy"));
    assert!(!events.contains(&"request_cancel"));
    assert!(!events.contains(&"request_wait"));
}

#[test]
fn rejects_tokens_and_seeds_that_cannot_reach_the_int32_engine_contract() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();

    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");

    for tokens in [&[-1_i64][..], &[i64::from(i32::MAX) + 1][..]] {
        assert!(matches!(
            session.start_request(tokens, 0),
            Err(Error::InvalidInput {
                field: "text_token_ids",
                ..
            })
        ));
    }
    assert!(matches!(
        session.start_request(&[1, 3_358], u64::from(u32::MAX) + 1),
        Err(Error::InvalidInput {
            field: "random_seed",
            ..
        })
    ));
    assert!(
        !recorded_events().contains(&"request_start"),
        "invalid descriptors must be rejected before FFI"
    );
}

#[test]
fn audio_lease_drop_releases_exactly_once() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();

    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    let lease = request
        .acquire_audio()
        .expect("audio acquire")
        .expect("audio available");
    drop(lease);
    request.close().expect("request destroy");

    assert_eq!(
        recorded_events()
            .iter()
            .filter(|event| **event == "audio_release")
            .count(),
        1
    );
}

#[test]
fn invalid_zero_lease_id_poisons_and_aborts_on_drop() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    let output = Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "drop_failure_child", "--nocapture"])
        .env("MTT_DROP_FAILURE_CHILD", "poisoned-zero-lease")
        .output()
        .expect("launch poisoned lease child");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("abort diagnostic must be UTF-8");
    assert!(
        stderr.contains("MagpieTTS-RT fatal"),
        "missing poisoned-request diagnostic: {stderr}"
    );
}

#[test]
fn post_acquire_overflow_releases_lease_before_returning_error() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();

    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    AUDIO_ACQUIRE_MODE.store(2, Ordering::SeqCst);
    let error = request
        .acquire_audio()
        .err()
        .expect("overflowed sequence must be rejected");
    assert_eq!(
        error,
        Error::InvalidNativeData {
            operation: "audio_acquire",
            field: "sequence",
            reason: "overflowed",
        }
    );
    assert_eq!(
        recorded_events()
            .iter()
            .filter(|event| **event == "audio_release")
            .count(),
        1
    );

    AUDIO_ACQUIRE_MODE.store(0, Ordering::SeqCst);
    let mut lease = request
        .acquire_audio()
        .expect("valid retry")
        .expect("audio available");
    lease.release().expect("valid lease release");
    drop(lease);
    request.close().expect("request destroy");
}

#[test]
fn rejects_snapshot_revision_counter_and_terminal_regressions() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    for (mode, expected_field) in [(1, "revision"), (2, "state"), (6, "snapshot counters")] {
        with_request(mode, |request| {
            request.poll().expect("first snapshot");
            let error = request.poll().expect_err("second snapshot must fail");
            assert!(matches!(
                error,
                Error::InvalidNativeData { field, .. } if field == expected_field
            ));
        });
    }
}

#[test]
fn rejects_control_and_unknown_failed_terminal_statuses() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    for mode in [3, 4, 5] {
        with_request(mode, |request| {
            assert!(matches!(
                request.poll(),
                Err(Error::InvalidNativeData {
                    field: "terminal_status",
                    ..
                })
            ));
        });
    }
}

#[test]
fn preserves_valid_async_terminal_diagnostics() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    with_request(8, |request| {
        let snapshot = request.poll().expect("valid failed snapshot");
        assert_eq!(snapshot.state, RequestState::Failed);
        assert_eq!(snapshot.terminal_status, Status::CUDA_ERROR);
        let diagnostic = snapshot
            .terminal_error
            .expect("failed snapshot must expose its persistent diagnostic");
        assert_eq!(diagnostic.status(), Status::CUDA_ERROR);
        assert_eq!(diagnostic.stage(), ErrorStage::CUDA);
        assert_eq!(diagnostic.message_bytes(), b"kernel launch failed");
    });
}

#[test]
fn rejects_malformed_or_changing_terminal_diagnostics() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    for (mode, expected_field) in [
        (9, "terminal_error_stage"),
        (10, "terminal_error_message"),
        (11, "terminal diagnostic"),
        (13, "terminal_error_message"),
    ] {
        with_request(mode, |request| {
            assert!(matches!(
                request.poll(),
                Err(Error::InvalidNativeData { field, .. }) if field == expected_field
            ));
        });
    }

    with_request(12, |request| {
        request.poll().expect("first failed snapshot");
        assert!(matches!(
            request.poll(),
            Err(Error::InvalidNativeData {
                field: "terminal diagnostic",
                ..
            })
        ));
    });
}

#[test]
fn rejects_successful_wait_without_a_newer_revision() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    with_request(7, |request| {
        assert!(matches!(
            request.wait_after(9, Duration::from_millis(1)),
            Err(Error::InvalidNativeData {
                field: "revision",
                ..
            })
        ));
    });
}

#[test]
fn rejects_regressing_audio_alignment_after_releasing_native_lease() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 8, 3_358], 1)
        .expect("request start");

    AUDIO_ACQUIRE_MODE.store(3, Ordering::SeqCst);
    let mut first = request
        .acquire_audio()
        .expect("first acquire")
        .expect("first lease");
    assert_eq!(first.committed_text_tokens(), Some(2));
    first.release().expect("release first lease");
    drop(first);

    assert!(matches!(
        request.acquire_audio(),
        Err(Error::InvalidNativeData {
            field: "committed_text_tokens",
            ..
        })
    ));
    assert_eq!(
        recorded_events()
            .iter()
            .filter(|event| **event == "audio_release")
            .count(),
        2
    );
    request.close().expect("request destroy");
}

#[test]
fn accepts_alignment_after_a_single_frame_eos_tail() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    AUDIO_ACQUIRE_MODE.store(5, Ordering::SeqCst);
    let mut lease = request
        .acquire_audio()
        .expect("single-frame final acquire")
        .expect("single-frame final lease");
    assert_eq!(lease.samples().expect("live PCM").len(), 1_024);
    assert_eq!(
        lease.alignment_events(),
        [AlignmentEvent {
            sample_index: 1_024,
            committed_text_tokens: 1,
        }]
    );
    lease.release().expect("release final lease");
    drop(lease);
    request.close().expect("request destroy");
}

#[test]
fn accepts_a_zero_sample_final_control_marker_after_pcm() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    AUDIO_ACQUIRE_MODE.store(7, Ordering::SeqCst);
    let mut first = request
        .acquire_audio()
        .expect("first PCM acquire")
        .expect("first PCM lease");
    assert_eq!(first.samples().expect("first PCM"), &AUDIO);
    assert!(first.is_first());
    assert!(!first.is_final());
    first.release().expect("release first PCM");
    drop(first);

    let mut marker = request
        .acquire_audio()
        .expect("FINAL marker acquire")
        .expect("FINAL marker lease");
    assert_eq!(marker.samples().expect("empty FINAL marker"), &[]);
    assert_eq!(marker.first_sample_index(), AUDIO.len() as u64);
    assert!(!marker.is_first());
    assert!(marker.is_final());
    assert_eq!(marker.committed_text_tokens(), Some(1));
    assert!(marker.alignment_events().is_empty());
    marker.release().expect("release FINAL marker");
    drop(marker);
    assert!(request.acquire_audio().expect("after FINAL").is_none());
    request.close().expect("request destroy");
}

#[test]
fn rejects_alignment_at_the_start_of_its_audio_lease() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    AUDIO_ACQUIRE_MODE.store(6, Ordering::SeqCst);
    assert!(matches!(
        request.acquire_audio(),
        Err(Error::InvalidNativeData {
            field: "alignment event sample_index",
            ..
        })
    ));
    assert_eq!(
        recorded_events()
            .iter()
            .filter(|event| **event == "audio_release")
            .count(),
        1
    );
    request.close().expect("request destroy");
}

#[test]
fn rejects_a_reused_audio_lease_identifier() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();
    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    AUDIO_ACQUIRE_MODE.store(4, Ordering::SeqCst);
    let mut first = request
        .acquire_audio()
        .expect("first acquire")
        .expect("first lease");
    first.release().expect("release first lease");
    drop(first);

    assert!(matches!(
        request.acquire_audio(),
        Err(Error::InvalidNativeData {
            field: "lease_id",
            ..
        })
    ));
    assert_eq!(
        recorded_events()
            .iter()
            .filter(|event| **event == "audio_release")
            .count(),
        2
    );
    request.close().expect("request destroy");
}

#[test]
fn rejects_incomplete_function_tables_before_any_call() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    let mut table = mock_table();
    table.audio_release = None;

    // SAFETY: every present function has the exact C signature. The missing
    // function is expected to be rejected before the table can be used.
    let error = unsafe { Api::from_table(table) }
        .err()
        .expect("incomplete API must fail");
    assert_eq!(error, Error::MissingApiFunction("audio_release"));
}

#[test]
fn explicit_request_close_retains_ownership_after_busy() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();

    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    REQUEST_DESTROY_FAILURES.store(1, Ordering::SeqCst);
    let error = request.close().expect_err("first close must report BUSY");
    match error {
        Error::Native(native) => assert_eq!(native.status(), Status::BUSY),
        unexpected => panic!("unexpected close error: {unexpected}"),
    }

    request
        .close()
        .expect("caller-controlled second close succeeds");
    drop(request);
    assert_eq!(
        recorded_events()
            .iter()
            .filter(|event| **event == "request_destroy")
            .count(),
        2
    );
}

#[test]
fn explicit_audio_release_retains_lease_after_failure() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    clear_events();

    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");
    let mut lease = request
        .acquire_audio()
        .expect("audio acquire")
        .expect("audio available");

    AUDIO_RELEASE_FAILURES.store(1, Ordering::SeqCst);
    let error = lease
        .release()
        .expect_err("first release must report INTERNAL_ERROR");
    match error {
        Error::Native(native) => assert_eq!(native.status(), Status::INTERNAL_ERROR),
        unexpected => panic!("unexpected release error: {unexpected}"),
    }
    assert_eq!(
        lease
            .samples()
            .expect("failed release must preserve the PCM lease"),
        &AUDIO
    );

    lease
        .release()
        .expect("caller-controlled second release succeeds");
    drop(lease);
    request.close().expect("request destroy");
    assert_eq!(
        recorded_events()
            .iter()
            .filter(|event| **event == "audio_release")
            .count(),
        2
    );
}

#[test]
fn native_drop_failures_abort_instead_of_being_discarded() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    for failure in ["request", "audio"] {
        let output = Command::new(std::env::current_exe().expect("current test executable"))
            .args(["--exact", "drop_failure_child", "--nocapture"])
            .env("MTT_DROP_FAILURE_CHILD", failure)
            .output()
            .expect("launch abort child");
        assert!(
            !output.status.success(),
            "{failure} Drop failure was silently accepted"
        );
        let stderr = String::from_utf8(output.stderr).expect("abort diagnostic must be UTF-8");
        assert!(
            stderr.contains("MagpieTTS-RT fatal"),
            "missing fail-closed diagnostic for {failure}: {stderr}"
        );
    }
}

#[test]
fn drop_failure_child() {
    let Ok(failure) = std::env::var("MTT_DROP_FAILURE_CHILD") else {
        return;
    };
    clear_events();

    let api = api();
    let runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .expect("runtime create");
    let model = runtime
        .load_model("/verified/bundle", [1; 32])
        .expect("model load");
    let session = model.create_session().expect("session create");
    let mut request = session
        .start_request(&[7, 3_358], 1)
        .expect("request start");

    match failure.as_str() {
        "request" => {
            REQUEST_DESTROY_FAILURES.store(1, Ordering::SeqCst);
            drop(request);
        }
        "audio" => {
            let lease = request
                .acquire_audio()
                .expect("audio acquire")
                .expect("audio available");
            AUDIO_RELEASE_FAILURES.store(1, Ordering::SeqCst);
            drop(lease);
        }
        "poisoned-zero-lease" => {
            AUDIO_ACQUIRE_MODE.store(1, Ordering::SeqCst);
            let error = request
                .acquire_audio()
                .err()
                .expect("zero lease identifier must be rejected");
            match error {
                Error::InvalidAudioLeaseAndReleaseFailed {
                    lease_error,
                    release_error,
                } => {
                    assert_eq!(
                        *lease_error,
                        Error::InvalidNativeData {
                            operation: "audio_acquire",
                            field: "lease_id",
                            reason: "must be non-zero",
                        }
                    );
                    match *release_error {
                        Error::Native(native) => {
                            assert_eq!(native.status(), Status::INVALID_ARGUMENT)
                        }
                        unexpected => panic!("unexpected cleanup error: {unexpected}"),
                    }
                }
                unexpected => panic!("unexpected zero-lease error: {unexpected}"),
            }
            assert_eq!(
                request
                    .poll()
                    .expect_err("poisoned request must reject use"),
                Error::Poisoned {
                    resource: "Request",
                    reason: "an invalid native audio lease could not be released",
                }
            );
            drop(request);
        }
        unexpected => panic!("unknown child failure mode {unexpected}"),
    }
}

#[cfg(feature = "native-link")]
#[test]
fn explicitly_linked_library_negotiates_abi_v1() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    Api::linked().expect("linked MagpieTTS-RT library must expose ABI v1");
}

#[cfg(feature = "native-link")]
#[test]
fn linked_contract_shim_converts_the_runtime_error() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    if std::env::var_os("MAGPIE_TTS_RT_NATIVE_SHIM").is_none() {
        return;
    }
    let api = Api::linked().expect("linked contract shim must expose ABI v1");
    let error = Runtime::create(&api, RuntimeConfig::new(0).expect("valid device"))
        .err()
        .expect("contract shim must reject runtime creation");
    match error {
        Error::Native(native) => {
            assert_eq!(native.status(), Status::UNAVAILABLE);
            assert_eq!(native.stage(), ErrorStage::RUNTIME);
        }
        unexpected => panic!("unexpected runtime_create error: {unexpected}"),
    }
}

#[cfg(feature = "native-link")]
#[test]
fn linked_thor_runtime_rejects_a_missing_bundle_without_fallback() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    if std::env::var_os("MAGPIE_TTS_RT_RUN_THOR_TEST").is_none() {
        return;
    }
    let api = Api::linked().expect("linked MagpieTTS-RT library must expose ABI v1");
    let mut runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid CUDA device"))
        .expect("Thor TensorRT runtime creation");

    let error = runtime
        .load_model("/magpie-tts-rt-test/nonexistent-bundle", [1; 32])
        .err()
        .expect("missing bundle must fail closed");
    match error {
        Error::Native(native) => {
            assert_eq!(native.status(), Status::IO_ERROR);
            assert_eq!(native.stage(), ErrorStage::MODEL);
        }
        unexpected => panic!("unexpected model_load error: {unexpected}"),
    }

    runtime.close().expect("explicit runtime destruction");
}

use std::mem::size_of;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use magpie_tts_rt::{Api, Error, ErrorStage, RequestState, Runtime, RuntimeConfig, Status, sys};

static TEST_LOCK: Mutex<()> = Mutex::new(());
static EVENTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
static REQUEST_DESTROY_FAILURES: AtomicUsize = AtomicUsize::new(0);
static AUDIO_RELEASE_FAILURES: AtomicUsize = AtomicUsize::new(0);
static AUDIO_ACQUIRE_MODE: AtomicUsize = AtomicUsize::new(0);
static AUDIO_ACQUIRE_CALLS: AtomicUsize = AtomicUsize::new(0);
static SNAPSHOT_MODE: AtomicUsize = AtomicUsize::new(0);
static SNAPSHOT_CALLS: AtomicUsize = AtomicUsize::new(0);
static AUDIO: [f32; 4_096] = [0.25; 4_096];

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
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_start");
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
    _error: *mut sys::mtt_error_v1_t,
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
        _ => {}
    }
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn request_wait(
    _request: *mut sys::mtt_request_t,
    after_revision: u64,
    _timeout_nanoseconds: u64,
    snapshot: *mut sys::mtt_request_snapshot_v1_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_wait");
    const STALE_WAIT_MODE: usize = 7;
    let revision = if SNAPSHOT_MODE.load(Ordering::SeqCst) == STALE_WAIT_MODE {
        after_revision
    } else {
        after_revision.saturating_add(1)
    };
    // SAFETY: snapshot is the writable pointer supplied by Request::wait_after.
    unsafe { write_running_snapshot(snapshot, revision) };
    sys::MTT_STATUS_OK
}

unsafe extern "C" fn request_cancel(
    _request: *mut sys::mtt_request_t,
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("request_cancel");
    sys::MTT_STATUS_OK
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
    _error: *mut sys::mtt_error_v1_t,
) -> sys::mtt_status_t {
    record("audio_acquire");
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
    lease.reserved = [0; 4];
    match AUDIO_ACQUIRE_MODE.load(Ordering::SeqCst) {
        0 => {}
        1 => lease.lease_id = 0,
        2 => lease.sequence = u64::MAX,
        3 => {
            if call_index == 0 {
                lease.flags = sys::MTT_AUDIO_FLAG_FIRST | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
                lease.committed_text_tokens = 2;
            } else {
                lease.sequence = 1;
                lease.first_sample_index = AUDIO.len() as u64;
                lease.flags = sys::MTT_AUDIO_FLAG_FINAL | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID;
                lease.committed_text_tokens = 1;
            }
        }
        4 => {
            if call_index == 0 {
                lease.flags = sys::MTT_AUDIO_FLAG_FIRST;
                lease.committed_text_tokens = 0;
            } else {
                lease.lease_id = 17;
                lease.sequence = 1;
                lease.first_sample_index = AUDIO.len() as u64;
                lease.flags = sys::MTT_AUDIO_FLAG_FINAL;
                lease.committed_text_tokens = 0;
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
}

fn recorded_events() -> Vec<&'static str> {
    EVENTS.lock().expect("event lock poisoned").clone()
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
    let mut request = session.start_request(&[7], 1).expect("request start");
    SNAPSHOT_MODE.store(mode, Ordering::SeqCst);
    operation(&mut request);
    request.close().expect("request destroy");
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
    let mut request = session.start_request(&[11, 12], 99).expect("request start");

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
    let request = session.start_request(&[7], 1).expect("request start");

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
    let mut request = session.start_request(&[7], 1).expect("request start");

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
    let mut request = session.start_request(&[7], 1).expect("request start");

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
    let mut request = session.start_request(&[7, 8], 1).expect("request start");

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
    let mut request = session.start_request(&[7], 1).expect("request start");

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
    let mut request = session.start_request(&[7], 1).expect("request start");

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
    let mut request = session.start_request(&[7], 1).expect("request start");
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
    let mut request = session.start_request(&[7], 1).expect("request start");

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
fn linked_thor_runtime_converts_the_contract_only_model_error() {
    let _test = TEST_LOCK.lock().expect("test lock poisoned");
    if std::env::var_os("MAGPIE_TTS_RT_RUN_THOR_TEST").is_none() {
        return;
    }
    let api = Api::linked().expect("linked MagpieTTS-RT library must expose ABI v1");
    let mut runtime = Runtime::create(&api, RuntimeConfig::new(0).expect("valid CUDA device"))
        .expect("Thor TensorRT runtime creation");

    let error = runtime
        .load_model("/not-used-by-contract-only-runtime", [1; 32])
        .err()
        .expect("initial runtime must report unimplemented model loading");
    match error {
        Error::Native(native) => {
            assert_eq!(native.status(), Status::UNAVAILABLE);
            assert_eq!(native.stage(), ErrorStage::MODEL);
        }
        unexpected => panic!("unexpected model_load error: {unexpected}"),
    }

    runtime.close().expect("explicit runtime destruction");
}

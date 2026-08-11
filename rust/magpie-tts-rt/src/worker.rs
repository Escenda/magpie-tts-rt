use std::error;
use std::fmt;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::{
    AlignmentEvent, Api, Error, ErrorStage, GetApiFn, ModelInfo, NativeError, Request,
    RequestState, Runtime, RuntimeConfig, Session, Status,
};

const START_COMMAND_CAPACITY: usize = 1;
const CONTROL_COMMAND_CAPACITY: usize = 8;
const WAKE_CAPACITY: usize = 1;
const WORKER_CONTROL_SLICE: Duration = Duration::from_millis(2);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerConfig {
    runtime: RuntimeConfig,
    bundle_path: String,
    expected_manifest_sha256: [u8; 32],
}

impl WorkerConfig {
    pub fn new(
        runtime: RuntimeConfig,
        bundle_path: impl Into<String>,
        expected_manifest_sha256: [u8; 32],
    ) -> std::result::Result<Self, WorkerError> {
        let bundle_path = bundle_path.into();
        if bundle_path.is_empty() {
            return Err(WorkerError::InvalidConfiguration(
                "bundle_path must not be empty",
            ));
        }
        if expected_manifest_sha256.iter().all(|byte| *byte == 0) {
            return Err(WorkerError::InvalidConfiguration(
                "expected_manifest_sha256 must not be all zero",
            ));
        }
        Ok(Self {
            runtime,
            bundle_path,
            expected_manifest_sha256,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedAudioChunk {
    pub sequence: u64,
    pub first_sample_index: u64,
    pub sample_rate_hz: u32,
    pub first: bool,
    pub final_chunk: bool,
    pub committed_text_tokens: Option<u64>,
    pub alignment_events: Vec<AlignmentEvent>,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SynthesisEvent {
    Audio(OwnedAudioChunk),
    Completed,
    Cancelled,
    Failed(NativeError),
    RuntimeError(Error),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkerError {
    Runtime(Error),
    InvalidConfiguration(&'static str),
    InvalidRequest { field: &'static str, reason: String },
    CommandChannelClosed,
    EventChannelClosed,
    Busy,
    UnknownRequest(u64),
    RequestIdentifierExhausted,
    WorkerPanicked,
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => write!(formatter, "{error}"),
            Self::InvalidConfiguration(reason) => {
                write!(formatter, "invalid worker configuration: {reason}")
            }
            Self::InvalidRequest { field, reason } => {
                write!(formatter, "invalid synthesis request {field}: {reason}")
            }
            Self::CommandChannelClosed => {
                formatter.write_str("inference worker command channel is closed")
            }
            Self::EventChannelClosed => formatter.write_str("synthesis event channel is closed"),
            Self::Busy => formatter
                .write_str("the inference worker is active or its bounded start queue is full"),
            Self::UnknownRequest(identifier) => {
                write!(formatter, "unknown request identifier {identifier}")
            }
            Self::RequestIdentifierExhausted => {
                formatter.write_str("request identifier sequence is exhausted")
            }
            Self::WorkerPanicked => formatter.write_str("inference worker thread panicked"),
        }
    }
}

impl error::Error for WorkerError {}

impl From<Error> for WorkerError {
    fn from(error: Error) -> Self {
        Self::Runtime(error)
    }
}

pub struct SynthesisStream {
    request_id: u64,
    controls: SyncSender<ControlCommand>,
    wake: SyncSender<()>,
    events: Receiver<SynthesisEvent>,
}

impl SynthesisStream {
    #[must_use]
    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    pub fn recv(&self) -> std::result::Result<SynthesisEvent, WorkerError> {
        self.events
            .recv()
            .map_err(|_| WorkerError::EventChannelClosed)
    }

    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<Option<SynthesisEvent>, WorkerError> {
        match self.events.recv_timeout(timeout) {
            Ok(event) => Ok(Some(event)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(WorkerError::EventChannelClosed),
        }
    }

    /// Requests cancellation, or succeeds idempotently when this worker has
    /// already verified and queued this stream's terminal event.
    ///
    /// A different or older unknown request identifier remains an error.
    pub fn cancel(&self) -> std::result::Result<(), WorkerError> {
        let (reply_sender, reply_receiver) = mpsc::sync_channel(1);
        self.controls
            .send(ControlCommand::Cancel {
                request_id: self.request_id,
                reply: reply_sender,
            })
            .map_err(|_| WorkerError::CommandChannelClosed)?;
        notify_worker(&self.wake)?;
        reply_receiver
            .recv()
            .map_err(|_| WorkerError::CommandChannelClosed)?
    }
}

pub struct InferenceWorker {
    starts: SyncSender<StartCommand>,
    controls: SyncSender<ControlCommand>,
    wake: SyncSender<()>,
    thread: Option<JoinHandle<std::result::Result<(), WorkerError>>>,
    model_info: ModelInfo,
}

fn admit_start(
    starts: &SyncSender<StartCommand>,
    command: StartCommand,
) -> std::result::Result<(), WorkerError> {
    match starts.try_send(command) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_)) => Err(WorkerError::Busy),
        Err(TrySendError::Disconnected(_)) => Err(WorkerError::CommandChannelClosed),
    }
}

fn notify_worker(wake: &SyncSender<()>) -> std::result::Result<(), WorkerError> {
    match wake.try_send(()) {
        Ok(()) | Err(TrySendError::Full(())) => Ok(()),
        Err(TrySendError::Disconnected(())) => Err(WorkerError::CommandChannelClosed),
    }
}

impl InferenceWorker {
    /// Creates a worker whose native API is resolved by `get_api`. Runtime,
    /// model, session, request, and lease values are constructed and retained
    /// exclusively on the spawned inference thread.
    ///
    /// # Safety
    ///
    /// `get_api` must satisfy [`Api::negotiate`] and its containing library
    /// must remain loaded until this worker has been shut down and joined.
    pub unsafe fn spawn_with_get_api(
        get_api: GetApiFn,
        config: WorkerConfig,
    ) -> std::result::Result<Self, WorkerError> {
        let (start_sender, start_receiver) = mpsc::sync_channel(START_COMMAND_CAPACITY);
        let (control_sender, control_receiver) = mpsc::sync_channel(CONTROL_COMMAND_CAPACITY);
        let (wake_sender, wake_receiver) = mpsc::sync_channel(WAKE_CAPACITY);
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let worker_thread = thread::Builder::new()
            .name("magpie-tts-rt-inference".to_owned())
            .spawn(move || {
                // SAFETY: propagated from spawn_with_get_api; all native
                // owners remain inside this thread until explicit teardown.
                let api = unsafe { Api::negotiate(get_api) };
                let result = api.map_err(WorkerError::Runtime).and_then(|api| {
                    initialize_and_run(
                        api,
                        config,
                        start_receiver,
                        control_receiver,
                        wake_receiver,
                        &ready_sender,
                    )
                });
                if let Err(error) = &result {
                    let _ = ready_sender.send(Err(error.clone()));
                }
                result
            })
            .map_err(|_| WorkerError::CommandChannelClosed)?;
        match ready_receiver.recv() {
            Ok(Ok(model_info)) => Ok(Self {
                starts: start_sender,
                controls: control_sender,
                wake: wake_sender,
                thread: Some(worker_thread),
                model_info,
            }),
            Ok(Err(error)) => {
                let _ = worker_thread.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker_thread.join();
                Err(WorkerError::CommandChannelClosed)
            }
        }
    }

    #[cfg(feature = "native-link")]
    pub fn spawn(config: WorkerConfig) -> std::result::Result<Self, WorkerError> {
        // SAFETY: native-link keeps the selected runtime library loaded for
        // the process lifetime.
        unsafe { Self::spawn_with_get_api(crate::sys::mtt_get_api, config) }
    }

    pub fn synthesize(
        &self,
        text_token_ids: Vec<i64>,
        random_seed: u64,
    ) -> std::result::Result<SynthesisStream, WorkerError> {
        self.validate_synthesis_request(&text_token_ids, random_seed)?;
        let (event_sender, event_receiver) = mpsc::sync_channel(1);
        let (reply_sender, reply_receiver) = mpsc::sync_channel(1);
        admit_start(
            &self.starts,
            StartCommand {
                text_token_ids,
                random_seed,
                events: event_sender,
                reply: reply_sender,
            },
        )?;
        notify_worker(&self.wake)?;
        let request_id = reply_receiver
            .recv()
            .map_err(|_| WorkerError::CommandChannelClosed)??;
        Ok(SynthesisStream {
            request_id,
            controls: self.controls.clone(),
            wake: self.wake.clone(),
            events: event_receiver,
        })
    }

    #[must_use]
    pub const fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn validate_synthesis_request(
        &self,
        text_token_ids: &[i64],
        random_seed: u64,
    ) -> std::result::Result<(), WorkerError> {
        if text_token_ids.is_empty() {
            return Err(WorkerError::InvalidRequest {
                field: "text_token_ids",
                reason: "must contain at least one prepared token".to_owned(),
            });
        }
        if text_token_ids.len() > self.model_info.maximum_text_tokens as usize {
            return Err(WorkerError::InvalidRequest {
                field: "text_token_ids",
                reason: format!(
                    "count {} exceeds model maximum {}",
                    text_token_ids.len(),
                    self.model_info.maximum_text_tokens
                ),
            });
        }
        let final_index = text_token_ids.len() - 1;
        if text_token_ids[final_index] != i64::from(self.model_info.eos_token_id) {
            return Err(WorkerError::InvalidRequest {
                field: "text_token_ids",
                reason: format!(
                    "final token at index {final_index} must be authenticated EOS {}",
                    self.model_info.eos_token_id
                ),
            });
        }
        if let Some((index, token)) = text_token_ids[..final_index]
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| {
                *token < 0 || *token >= i64::from(self.model_info.tokenizer_vocabulary_size)
            })
        {
            return Err(WorkerError::InvalidRequest {
                field: "text_token_ids",
                reason: format!(
                    "non-final token at index {index} is {token}, outside normal tokenizer rows [0, {})",
                    self.model_info.tokenizer_vocabulary_size
                ),
            });
        }
        if random_seed > u64::from(u32::MAX) {
            return Err(WorkerError::InvalidRequest {
                field: "random_seed",
                reason: format!("{random_seed} exceeds uint32 maximum {}", u32::MAX),
            });
        }
        Ok(())
    }

    pub fn shutdown(mut self) -> std::result::Result<(), WorkerError> {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> std::result::Result<(), WorkerError> {
        let Some(worker_thread) = self.thread.take() else {
            return Ok(());
        };
        let shutdown_result = self
            .controls
            .send(ControlCommand::Shutdown)
            .map_err(|_| WorkerError::CommandChannelClosed)
            .and_then(|()| notify_worker(&self.wake));
        let worker_result = worker_thread
            .join()
            .map_err(|_| WorkerError::WorkerPanicked)?;
        match (shutdown_result, worker_result) {
            (_, Err(worker_error)) => Err(worker_error),
            (Err(channel_error), Ok(())) => Err(channel_error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        // Dropping JoinHandle detaches the inference thread. Drop deliberately
        // does not send Shutdown, join, or invoke any native operation. Once
        // the final command sender (including any SynthesisStream clone) is
        // dropped, the worker observes channel disconnection and tears its
        // native hierarchy down fail-closed on the inference thread.
        let _ = self.thread.take();
    }
}

struct StartCommand {
    text_token_ids: Vec<i64>,
    random_seed: u64,
    events: SyncSender<SynthesisEvent>,
    reply: SyncSender<std::result::Result<u64, WorkerError>>,
}

enum ControlCommand {
    Cancel {
        request_id: u64,
        reply: SyncSender<std::result::Result<(), WorkerError>>,
    },
    Shutdown,
}

enum AdmittedWorkerCommand {
    Control(ControlCommand),
    Start(StartCommand),
}

fn try_receive_prioritized(
    starts: &Receiver<StartCommand>,
    controls: &Receiver<ControlCommand>,
    starts_connected: &mut bool,
    controls_connected: &mut bool,
) -> Option<AdmittedWorkerCommand> {
    if *controls_connected {
        match controls.try_recv() {
            Ok(command) => return Some(AdmittedWorkerCommand::Control(command)),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => *controls_connected = false,
        }
    }
    if *starts_connected {
        match starts.try_recv() {
            Ok(command) => return Some(AdmittedWorkerCommand::Start(command)),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => *starts_connected = false,
        }
    }
    None
}

struct ActiveRequest {
    identifier: u64,
    request: Request,
    events: Option<SyncSender<SynthesisEvent>>,
    pending_event: Option<SynthesisEvent>,
    observed_revision: u64,
    known_available_audio: u32,
    terminal_snapshot: Option<crate::RequestSnapshot>,
    terminal_event_queued: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdvanceOutcome {
    Finished,
    Progressed,
    Waited,
    Backpressured,
}

fn initialize_and_run(
    api: Api,
    config: WorkerConfig,
    starts: Receiver<StartCommand>,
    controls: Receiver<ControlCommand>,
    wake: Receiver<()>,
    ready: &SyncSender<std::result::Result<ModelInfo, WorkerError>>,
) -> std::result::Result<(), WorkerError> {
    let mut runtime = Runtime::create(&api, config.runtime).map_err(WorkerError::Runtime)?;
    let mut model = runtime
        .load_model(&config.bundle_path, config.expected_manifest_sha256)
        .map_err(WorkerError::Runtime)?;
    let model_info = model.info().map_err(WorkerError::Runtime)?;
    let mut session = model.create_session().map_err(WorkerError::Runtime)?;
    ready
        .send(Ok(model_info))
        .map_err(|_| WorkerError::CommandChannelClosed)?;

    let loop_result = worker_loop(&session, starts, controls, wake);
    session.close().map_err(WorkerError::Runtime)?;
    drop(session);
    model.close().map_err(WorkerError::Runtime)?;
    drop(model);
    runtime.close().map_err(WorkerError::Runtime)?;
    loop_result
}

fn worker_loop(
    session: &Session,
    starts: Receiver<StartCommand>,
    controls: Receiver<ControlCommand>,
    wake: Receiver<()>,
) -> std::result::Result<(), WorkerError> {
    let mut active: Option<ActiveRequest> = None;
    let mut most_recent_terminal_identifier = None;
    let mut next_identifier = 1_u64;
    let mut shutdown_requested = false;
    let mut shutdown_error = None;
    let mut starts_connected = true;
    let mut controls_connected = true;

    loop {
        let command = try_receive_prioritized(
            &starts,
            &controls,
            &mut starts_connected,
            &mut controls_connected,
        );
        let handled_control = matches!(command, Some(AdmittedWorkerCommand::Control(_)));
        match command.as_ref() {
            Some(AdmittedWorkerCommand::Control(ControlCommand::Cancel { request_id, reply })) => {
                let mut fatal_error = None;
                let result = match active.as_mut() {
                    Some(request) if request.identifier == *request_id => {
                        let result = cancel_active_request(request);
                        if let Err(error) = &result
                            && cancel_error_is_fatal(error)
                        {
                            fatal_error = Some(error.clone());
                        }
                        result
                    }
                    // The worker closes a request only after its terminal event
                    // has entered the request's bounded event channel. Remember
                    // that exact identifier so a cancel racing the queued
                    // terminal remains idempotent without accepting an
                    // arbitrary unknown identifier.
                    _ if most_recent_terminal_identifier == Some(*request_id) => Ok(()),
                    _ => Err(WorkerError::UnknownRequest(*request_id)),
                };
                let _ = reply.send(result);
                if let Some(error) = fatal_error {
                    let Some(request) = active.as_mut() else {
                        return Err(error);
                    };
                    return Err(drive_fatal_request(request, &starts, &controls, error));
                }
            }
            Some(AdmittedWorkerCommand::Control(ControlCommand::Shutdown)) => {
                shutdown_requested = true;
                if let Some(request) = active.as_mut() {
                    request.events = None;
                    request.pending_event = None;
                }
            }
            Some(AdmittedWorkerCommand::Start(_)) | None => {}
        }

        let mut advance_outcome = None;
        if let Some(request) = active.as_mut() {
            match advance_request(request) {
                Ok(AdvanceOutcome::Finished) => {
                    advance_outcome = Some(AdvanceOutcome::Finished);
                    most_recent_terminal_identifier = Some(request.identifier);
                    active = None;
                    if shutdown_requested {
                        return shutdown_error.map_or(Ok(()), Err);
                    }
                }
                Ok(outcome) => advance_outcome = Some(outcome),
                Err(error) => {
                    return Err(drive_fatal_request(request, &starts, &controls, error));
                }
            }
        } else if shutdown_requested {
            return shutdown_error.map_or(Ok(()), Err);
        }

        let handled_start = matches!(command, Some(AdmittedWorkerCommand::Start(_)));
        if let Some(AdmittedWorkerCommand::Start(StartCommand {
            text_token_ids,
            random_seed,
            events,
            reply,
        })) = command
        {
            if active.is_some() || shutdown_requested {
                let _ = reply.send(Err(WorkerError::Busy));
            } else if next_identifier == u64::MAX {
                let _ = reply.send(Err(WorkerError::RequestIdentifierExhausted));
            } else {
                match session.start_request(&text_token_ids, random_seed) {
                    Ok(request) => {
                        let identifier = next_identifier;
                        next_identifier += 1;
                        active = Some(ActiveRequest {
                            identifier,
                            request,
                            events: Some(events),
                            pending_event: None,
                            observed_revision: 0,
                            known_available_audio: 0,
                            terminal_snapshot: None,
                            terminal_event_queued: false,
                        });
                        let _ = reply.send(Ok(identifier));
                    }
                    Err(error) => {
                        let error = WorkerError::Runtime(error);
                        let _ = reply.send(Err(error.clone()));
                        return Err(error);
                    }
                }
            }
        }

        if !starts_connected && !controls_connected && !shutdown_requested {
            // Every external command owner has gone away. Detach the bounded
            // event sink, then let advance_request explicitly cancel, observe
            // the drained terminal state, and destroy the native request
            // before the hierarchy is torn down.
            shutdown_requested = true;
            shutdown_error = Some(WorkerError::CommandChannelClosed);
            if let Some(request) = active.as_mut() {
                request.events = None;
                request.pending_event = None;
            }
        }

        if !handled_control && !handled_start {
            match advance_outcome {
                None => wake.recv().map_err(|_| WorkerError::CommandChannelClosed)?,
                Some(AdvanceOutcome::Backpressured) => thread::sleep(WORKER_CONTROL_SLICE),
                Some(
                    AdvanceOutcome::Finished | AdvanceOutcome::Progressed | AdvanceOutcome::Waited,
                ) => {}
            }
        }
    }
}

fn drive_fatal_request(
    request: &mut ActiveRequest,
    starts: &Receiver<StartCommand>,
    controls: &Receiver<ControlCommand>,
    error: WorkerError,
) -> WorkerError {
    let mut pending_error_event = match &error {
        WorkerError::Runtime(runtime_error) => {
            Some(SynthesisEvent::RuntimeError(runtime_error.clone()))
        }
        _ => None,
    };
    request.pending_event = None;

    loop {
        let Some(event) = pending_error_event.take() else {
            request.events = None;
            return error;
        };
        let Some(events) = request.events.as_ref() else {
            return error;
        };
        match events.try_send(event) {
            Ok(()) => {
                request.events = None;
                return error;
            }
            Err(TrySendError::Disconnected(_)) => {
                request.events = None;
                return error;
            }
            Err(TrySendError::Full(event)) => {
                pending_error_event = Some(event);
            }
        }

        let mut controls_connected = true;
        match controls.try_recv() {
            Ok(ControlCommand::Shutdown) => {
                request.events = None;
                return error;
            }
            Ok(ControlCommand::Cancel { reply, .. }) => {
                let _ = reply.send(Err(error.clone()));
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                controls_connected = false;
            }
        }

        let mut starts_connected = true;
        match starts.try_recv() {
            Ok(StartCommand { reply, .. }) => {
                let _ = reply.send(Err(error.clone()));
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                starts_connected = false;
            }
        }
        if !starts_connected && !controls_connected {
            request.events = None;
            return error;
        }
        thread::sleep(WORKER_CONTROL_SLICE);
    }
}

fn cancel_error_is_fatal(error: &WorkerError) -> bool {
    !matches!(
        error,
        WorkerError::Runtime(Error::Native(native))
            if native.status() == Status::INVALID_ARGUMENT
                && native.stage() == ErrorStage::REQUEST
    )
}

fn cancel_active_request(request: &mut ActiveRequest) -> std::result::Result<(), WorkerError> {
    if request.terminal_snapshot.is_some() {
        return Ok(());
    }

    match request.request.cancel() {
        Ok(()) => {
            // Accepted cancellation discards any native audio that has not
            // been acquired. Only a newer revision may advertise new leases.
            request.known_available_audio = 0;
            Ok(())
        }
        Err(cancel_error) if cancel_may_have_raced_terminal(&cancel_error) => {
            match request.request.poll() {
                Ok(snapshot) if snapshot.state != RequestState::Running => {
                    observe_snapshot(request, snapshot);
                    Ok(())
                }
                // A running snapshot proves that cancellation did not race a
                // terminal transition. Retain that real revision for the next
                // wait, but preserve the original typed cancel diagnostic.
                Ok(snapshot) => {
                    observe_snapshot(request, snapshot);
                    Err(WorkerError::Runtime(cancel_error))
                }
                Err(_) => Err(WorkerError::Runtime(cancel_error)),
            }
        }
        Err(error) => Err(WorkerError::Runtime(error)),
    }
}

fn cancel_may_have_raced_terminal(error: &Error) -> bool {
    matches!(
        error,
        Error::Native(native)
            if native.status() == Status::INVALID_ARGUMENT
                && native.stage() == ErrorStage::REQUEST
    )
}

fn advance_request(
    request: &mut ActiveRequest,
) -> std::result::Result<AdvanceOutcome, WorkerError> {
    flush_pending_event(request);
    if request.pending_event.is_some() {
        return Ok(AdvanceOutcome::Backpressured);
    }

    if request.events.is_none() && request.terminal_snapshot.is_none() {
        cancel_active_request(request)?;
    }

    if !request.terminal_event_queued && request.known_available_audio > 0 {
        match acquire_owned_audio(&mut request.request)? {
            Some(chunk) => {
                request.known_available_audio -= 1;
                if request.events.is_some() {
                    request.pending_event = Some(SynthesisEvent::Audio(chunk));
                    flush_pending_event(request);
                    if request.pending_event.is_some() {
                        return Ok(AdvanceOutcome::Backpressured);
                    }
                }
                return Ok(AdvanceOutcome::Progressed);
            }
            None => {
                // A cancellation or asynchronous failure may discard queued
                // native audio after the observed snapshot. Do not invent a
                // terminal result; wait for the revision that explains it.
                request.known_available_audio = 0;
            }
        }
    }

    if request.terminal_snapshot.is_none() {
        let snapshot = request
            .request
            .wait_after(request.observed_revision, WORKER_CONTROL_SLICE)
            .map_err(WorkerError::Runtime)?;
        let Some(snapshot) = snapshot else {
            return Ok(AdvanceOutcome::Waited);
        };
        observe_snapshot(request, snapshot);
        return Ok(AdvanceOutcome::Progressed);
    }
    let Some(snapshot) = request.terminal_snapshot.clone() else {
        return Ok(AdvanceOutcome::Progressed);
    };

    if request.known_available_audio > 0 {
        return Ok(AdvanceOutcome::Progressed);
    }

    if !request.terminal_event_queued {
        request.terminal_event_queued = true;
        if request.events.is_some() {
            request.pending_event = Some(terminal_event(snapshot)?);
            flush_pending_event(request);
            if request.pending_event.is_some() {
                return Ok(AdvanceOutcome::Backpressured);
            }
        }
    }

    request.request.close().map_err(WorkerError::Runtime)?;
    Ok(AdvanceOutcome::Finished)
}

fn observe_snapshot(request: &mut ActiveRequest, snapshot: crate::RequestSnapshot) {
    request.observed_revision = snapshot.revision;
    request.known_available_audio = snapshot.available_audio_leases;
    if snapshot.state != RequestState::Running {
        request.terminal_snapshot = Some(snapshot);
    }
}

fn acquire_owned_audio(request: &mut Request) -> crate::Result<Option<OwnedAudioChunk>> {
    let Some(mut lease) = request.acquire_audio()? else {
        return Ok(None);
    };
    let chunk_result = (|| {
        Ok(OwnedAudioChunk {
            sequence: lease.sequence(),
            first_sample_index: lease.first_sample_index(),
            sample_rate_hz: lease.sample_rate_hz(),
            first: lease.is_first(),
            final_chunk: lease.is_final(),
            committed_text_tokens: lease.committed_text_tokens(),
            alignment_events: lease.alignment_events().to_vec(),
            samples: lease.samples()?.to_vec(),
        })
    })();
    let release_result = lease.release();
    drop(lease);
    match (chunk_result, release_result) {
        (Ok(chunk), Ok(())) => Ok(Some(chunk)),
        (Err(chunk_error), Ok(())) => Err(chunk_error),
        (Ok(_), Err(release_error)) => Err(release_error),
        (Err(_), Err(release_error)) => Err(release_error),
    }
}

fn terminal_event(
    snapshot: crate::RequestSnapshot,
) -> std::result::Result<SynthesisEvent, WorkerError> {
    match snapshot.state {
        RequestState::Running => Err(WorkerError::Runtime(Error::InvalidNativeData {
            operation: "worker_terminal_event",
            field: "state",
            reason: "terminal event requires a terminal snapshot",
        })),
        RequestState::Completed => Ok(SynthesisEvent::Completed),
        RequestState::Cancelled => Ok(SynthesisEvent::Cancelled),
        RequestState::Failed => {
            let failure =
                snapshot
                    .terminal_error
                    .ok_or(WorkerError::Runtime(Error::InvalidNativeData {
                        operation: "worker_terminal_event",
                        field: "terminal_error",
                        reason: "FAILED snapshot omitted its diagnostic",
                    }))?;
            Ok(SynthesisEvent::Failed(failure))
        }
    }
}

fn flush_pending_event(request: &mut ActiveRequest) {
    let Some(event) = request.pending_event.take() else {
        return;
    };
    let Some(events) = request.events.as_ref() else {
        return;
    };
    match events.try_send(event) {
        Ok(()) => {}
        Err(TrySendError::Full(event)) => request.pending_event = Some(event),
        Err(TrySendError::Disconnected(_)) => request.events = None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start_command(seed: u64) -> StartCommand {
        let (events, _) = mpsc::sync_channel(1);
        let (reply, _) = mpsc::sync_channel(1);
        StartCommand {
            text_token_ids: vec![7, 3_358],
            random_seed: seed,
            events,
            reply,
        }
    }

    #[test]
    fn config_rejects_missing_trust_inputs() {
        let runtime = RuntimeConfig::new(0).expect("runtime config");
        assert_eq!(
            WorkerConfig::new(runtime, "", [1; 32]),
            Err(WorkerError::InvalidConfiguration(
                "bundle_path must not be empty",
            ))
        );
        assert_eq!(
            WorkerConfig::new(runtime, "/bundle", [0; 32]),
            Err(WorkerError::InvalidConfiguration(
                "expected_manifest_sha256 must not be all zero",
            ))
        );
    }

    #[test]
    fn start_admission_is_bounded_and_reports_busy_without_waiting() {
        let (starts, receiver) = mpsc::sync_channel(START_COMMAND_CAPACITY);
        admit_start(&starts, start_command(1)).expect("first bounded admission");
        assert_eq!(
            admit_start(&starts, start_command(2)),
            Err(WorkerError::Busy)
        );
        let admitted = receiver.try_recv().expect("admitted start");
        assert_eq!(admitted.random_seed, 1);
        drop(receiver);
        assert_eq!(
            admit_start(&starts, start_command(3)),
            Err(WorkerError::CommandChannelClosed)
        );
    }

    #[test]
    fn control_admission_wins_under_repeated_start_saturation() {
        let (starts, start_receiver) = mpsc::sync_channel(START_COMMAND_CAPACITY);
        let (controls, control_receiver) = mpsc::sync_channel(CONTROL_COMMAND_CAPACITY);
        let mut starts_connected = true;
        let mut controls_connected = true;

        for iteration in 0..4_096_u64 {
            admit_start(&starts, start_command(iteration)).expect("fill start queue");
            controls
                .send(ControlCommand::Shutdown)
                .expect("admit priority control");

            assert!(matches!(
                try_receive_prioritized(
                    &start_receiver,
                    &control_receiver,
                    &mut starts_connected,
                    &mut controls_connected,
                ),
                Some(AdmittedWorkerCommand::Control(ControlCommand::Shutdown))
            ));
            match try_receive_prioritized(
                &start_receiver,
                &control_receiver,
                &mut starts_connected,
                &mut controls_connected,
            ) {
                Some(AdmittedWorkerCommand::Start(command)) => {
                    assert_eq!(command.random_seed, iteration);
                }
                _ => panic!("saturated start was not retained after priority control"),
            }
        }
    }

    #[test]
    fn one_slot_wake_coalesces_without_changing_control_priority() {
        let (starts, start_receiver) = mpsc::sync_channel(START_COMMAND_CAPACITY);
        let (controls, control_receiver) = mpsc::sync_channel(CONTROL_COMMAND_CAPACITY);
        let (wake, wake_receiver) = mpsc::sync_channel(WAKE_CAPACITY);
        let mut starts_connected = true;
        let mut controls_connected = true;

        admit_start(&starts, start_command(1)).expect("admit start payload");
        notify_worker(&wake).expect("wake for start");
        controls
            .send(ControlCommand::Shutdown)
            .expect("admit control payload");
        notify_worker(&wake).expect("coalesce control wake");

        wake_receiver.try_recv().expect("one coalesced wake");
        assert_eq!(wake_receiver.try_recv(), Err(TryRecvError::Empty));
        assert!(matches!(
            try_receive_prioritized(
                &start_receiver,
                &control_receiver,
                &mut starts_connected,
                &mut controls_connected,
            ),
            Some(AdmittedWorkerCommand::Control(ControlCommand::Shutdown))
        ));
        assert!(matches!(
            try_receive_prioritized(
                &start_receiver,
                &control_receiver,
                &mut starts_connected,
                &mut controls_connected,
            ),
            Some(AdmittedWorkerCommand::Start(_))
        ));
    }

    #[test]
    fn wake_disconnection_is_an_explicit_command_channel_error() {
        let (wake, wake_receiver) = mpsc::sync_channel(WAKE_CAPACITY);
        drop(wake_receiver);
        assert_eq!(notify_worker(&wake), Err(WorkerError::CommandChannelClosed));
    }
}

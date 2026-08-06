#![doc = "Safe Rust ownership layer for the MagpieTTS-RT C ABI."]

mod api;
mod error;
mod runtime;
mod worker;

pub use api::{Api, GetApiFn};
pub use error::{Error, ErrorStage, NativeError, Result, Status};
pub use magpie_tts_rt_sys as sys;
pub use runtime::{
    AlignmentEvent, AudioLease, Model, ModelInfo, Request, RequestSnapshot, RequestState, Runtime,
    RuntimeConfig, Session,
};
pub use worker::{
    InferenceWorker, OwnedAudioChunk, SynthesisEvent, SynthesisStream, WorkerConfig, WorkerError,
};

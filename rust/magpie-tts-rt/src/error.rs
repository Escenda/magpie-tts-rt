use std::error;
use std::fmt;

use crate::sys;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Status(i32);

impl Status {
    pub const OK: Self = Self(sys::MTT_STATUS_OK);
    pub const INVALID_ARGUMENT: Self = Self(sys::MTT_STATUS_INVALID_ARGUMENT);
    pub const ABI_MISMATCH: Self = Self(sys::MTT_STATUS_ABI_MISMATCH);
    pub const BUSY: Self = Self(sys::MTT_STATUS_BUSY);
    pub const IO_ERROR: Self = Self(sys::MTT_STATUS_IO_ERROR);
    pub const MANIFEST_ERROR: Self = Self(sys::MTT_STATUS_MANIFEST_ERROR);
    pub const RUNTIME_MISMATCH: Self = Self(sys::MTT_STATUS_RUNTIME_MISMATCH);
    pub const HASH_MISMATCH: Self = Self(sys::MTT_STATUS_HASH_MISMATCH);
    pub const ENGINE_ERROR: Self = Self(sys::MTT_STATUS_ENGINE_ERROR);
    pub const CUDA_ERROR: Self = Self(sys::MTT_STATUS_CUDA_ERROR);
    pub const CANCELLED: Self = Self(sys::MTT_STATUS_CANCELLED);
    pub const WOULD_BLOCK: Self = Self(sys::MTT_STATUS_WOULD_BLOCK);
    pub const TIMEOUT: Self = Self(sys::MTT_STATUS_TIMEOUT);
    pub const POISONED: Self = Self(sys::MTT_STATUS_POISONED);
    pub const UNAVAILABLE: Self = Self(sys::MTT_STATUS_UNAVAILABLE);
    pub const INTERNAL_ERROR: Self = Self(sys::MTT_STATUS_INTERNAL_ERROR);

    #[must_use]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self.0
    }

    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.0 == sys::MTT_STATUS_OK
    }

    pub(crate) const fn is_declared(self) -> bool {
        self.0 >= sys::MTT_STATUS_OK && self.0 <= sys::MTT_STATUS_INTERNAL_ERROR
    }

    pub(crate) const fn is_terminal_failure(self) -> bool {
        matches!(
            self,
            Self::INVALID_ARGUMENT
                | Self::ABI_MISMATCH
                | Self::IO_ERROR
                | Self::MANIFEST_ERROR
                | Self::RUNTIME_MISMATCH
                | Self::HASH_MISMATCH
                | Self::ENGINE_ERROR
                | Self::CUDA_ERROR
                | Self::POISONED
                | Self::UNAVAILABLE
                | Self::INTERNAL_ERROR
        )
    }

    fn name(self) -> Option<&'static str> {
        match self {
            Self::OK => Some("OK"),
            Self::INVALID_ARGUMENT => Some("INVALID_ARGUMENT"),
            Self::ABI_MISMATCH => Some("ABI_MISMATCH"),
            Self::BUSY => Some("BUSY"),
            Self::IO_ERROR => Some("IO_ERROR"),
            Self::MANIFEST_ERROR => Some("MANIFEST_ERROR"),
            Self::RUNTIME_MISMATCH => Some("RUNTIME_MISMATCH"),
            Self::HASH_MISMATCH => Some("HASH_MISMATCH"),
            Self::ENGINE_ERROR => Some("ENGINE_ERROR"),
            Self::CUDA_ERROR => Some("CUDA_ERROR"),
            Self::CANCELLED => Some("CANCELLED"),
            Self::WOULD_BLOCK => Some("WOULD_BLOCK"),
            Self::TIMEOUT => Some("TIMEOUT"),
            Self::POISONED => Some("POISONED"),
            Self::UNAVAILABLE => Some("UNAVAILABLE"),
            Self::INTERNAL_ERROR => Some("INTERNAL_ERROR"),
            _ => None,
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(formatter, "{name} ({})", self.0),
            None => write!(formatter, "unknown status ({})", self.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ErrorStage(i32);

impl ErrorStage {
    pub const NONE: Self = Self(sys::MTT_ERROR_STAGE_NONE);
    pub const ABI: Self = Self(sys::MTT_ERROR_STAGE_ABI);
    pub const RUNTIME: Self = Self(sys::MTT_ERROR_STAGE_RUNTIME);
    pub const MANIFEST: Self = Self(sys::MTT_ERROR_STAGE_MANIFEST);
    pub const MODEL: Self = Self(sys::MTT_ERROR_STAGE_MODEL);
    pub const SESSION: Self = Self(sys::MTT_ERROR_STAGE_SESSION);
    pub const REQUEST: Self = Self(sys::MTT_ERROR_STAGE_REQUEST);
    pub const TENSORRT: Self = Self(sys::MTT_ERROR_STAGE_TENSORRT);
    pub const CUDA: Self = Self(sys::MTT_ERROR_STAGE_CUDA);
    pub const PLUGIN: Self = Self(sys::MTT_ERROR_STAGE_PLUGIN);
    pub const ALIGNMENT: Self = Self(sys::MTT_ERROR_STAGE_ALIGNMENT);
    pub const CODEC: Self = Self(sys::MTT_ERROR_STAGE_CODEC);

    #[must_use]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn as_raw(self) -> i32 {
        self.0
    }

    pub(crate) const fn is_declared(self) -> bool {
        self.0 >= sys::MTT_ERROR_STAGE_NONE && self.0 <= sys::MTT_ERROR_STAGE_CODEC
    }

    fn name(self) -> Option<&'static str> {
        match self {
            Self::NONE => Some("NONE"),
            Self::ABI => Some("ABI"),
            Self::RUNTIME => Some("RUNTIME"),
            Self::MANIFEST => Some("MANIFEST"),
            Self::MODEL => Some("MODEL"),
            Self::SESSION => Some("SESSION"),
            Self::REQUEST => Some("REQUEST"),
            Self::TENSORRT => Some("TENSORRT"),
            Self::CUDA => Some("CUDA"),
            Self::PLUGIN => Some("PLUGIN"),
            Self::ALIGNMENT => Some("ALIGNMENT"),
            Self::CODEC => Some("CODEC"),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(formatter, "{name} ({})", self.0),
            None => write!(formatter, "unknown stage ({})", self.0),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeError {
    operation: &'static str,
    status: Status,
    stage: ErrorStage,
    message: Vec<u8>,
}

impl NativeError {
    pub(crate) fn new(
        operation: &'static str,
        status: Status,
        stage: ErrorStage,
        message: Vec<u8>,
    ) -> Self {
        Self {
            operation,
            status,
            stage,
            message,
        }
    }

    #[must_use]
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    #[must_use]
    pub const fn status(&self) -> Status {
        self.status
    }

    #[must_use]
    pub const fn stage(&self) -> ErrorStage {
        self.stage
    }

    #[must_use]
    pub fn message_bytes(&self) -> &[u8] {
        &self.message
    }

    pub fn message(&self) -> std::result::Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.message)
    }
}

impl fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} failed with {} at {}",
            self.operation, self.status, self.stage
        )?;
        if !self.message.is_empty() {
            formatter.write_str(": ")?;
            for byte in &self.message {
                for escaped in std::ascii::escape_default(*byte) {
                    write!(formatter, "{}", char::from(escaped))?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    AbiNegotiation {
        status: Status,
    },
    InvalidApiHeader {
        struct_size: u32,
        abi_version: u32,
    },
    MissingApiFunction(&'static str),
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    ResourceInUse {
        resource: &'static str,
    },
    Poisoned {
        resource: &'static str,
        reason: &'static str,
    },
    Closed {
        resource: &'static str,
    },
    NullHandle {
        operation: &'static str,
    },
    Native(NativeError),
    InvalidNativeError {
        operation: &'static str,
        returned_status: Status,
        struct_size: u32,
        abi_version: u32,
        reported_status: Status,
        reported_stage: ErrorStage,
        message: Vec<u8>,
        nul_terminated: bool,
        reason: &'static str,
    },
    InvalidNativeData {
        operation: &'static str,
        field: &'static str,
        reason: &'static str,
    },
    InvalidAudioLeaseAndReleaseFailed {
        lease_error: Box<Error>,
        release_error: Box<Error>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AbiNegotiation { status } => {
                write!(formatter, "mtt_get_api failed with {status}")
            }
            Self::InvalidApiHeader {
                struct_size,
                abi_version,
            } => write!(
                formatter,
                "invalid API header: struct_size={struct_size}, abi_version={abi_version}"
            ),
            Self::MissingApiFunction(name) => {
                write!(
                    formatter,
                    "the negotiated API omitted required function {name}"
                )
            }
            Self::InvalidInput { field, reason } => {
                write!(formatter, "invalid {field}: {reason}")
            }
            Self::ResourceInUse { resource } => {
                write!(formatter, "{resource} still owns live child handles")
            }
            Self::Poisoned { resource, reason } => {
                write!(formatter, "{resource} is poisoned: {reason}")
            }
            Self::Closed { resource } => write!(formatter, "{resource} is already closed"),
            Self::NullHandle { operation } => {
                write!(formatter, "{operation} returned a null handle")
            }
            Self::Native(error) => error.fmt(formatter),
            Self::InvalidNativeError {
                operation,
                returned_status,
                struct_size,
                abi_version,
                reported_status,
                reported_stage,
                message,
                nul_terminated,
                reason,
            } => {
                write!(
                    formatter,
                    "{operation} returned {returned_status} with malformed error data \
                     (struct_size={struct_size}, abi_version={abi_version}, \
                     reported_status={reported_status}, reported_stage={reported_stage}, \
                     nul_terminated={nul_terminated}, reason={reason}, message="
                )?;
                for byte in message {
                    for escaped in std::ascii::escape_default(*byte) {
                        write!(formatter, "{}", char::from(escaped))?;
                    }
                }
                formatter.write_str(")")
            }
            Self::InvalidNativeData {
                operation,
                field,
                reason,
            } => write!(formatter, "{operation} returned invalid {field}: {reason}"),
            Self::InvalidAudioLeaseAndReleaseFailed {
                lease_error,
                release_error,
            } => write!(
                formatter,
                "invalid audio lease ({lease_error}); releasing that lease also failed \
                 ({release_error})"
            ),
        }
    }
}

impl error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

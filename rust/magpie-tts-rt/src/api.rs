use std::mem::size_of;
use std::rc::Rc;

use crate::error::{Error, ErrorStage, NativeError, Result, Status};
use crate::sys;

pub type GetApiFn = unsafe extern "C" fn(u32, *mut sys::mtt_api_v1_t) -> sys::mtt_status_t;

/// A validated ABI v1 function table.
///
/// ABI v1 does not declare native handles thread-safe. The internal `Rc`
/// therefore intentionally makes this type, and every owner derived from it,
/// neither `Send` nor `Sync`.
///
/// ```compile_fail
/// use magpie_tts_rt::Api;
///
/// fn require_send<T: Send>() {}
/// require_send::<Api>();
/// ```
#[derive(Clone)]
pub struct Api {
    raw: Rc<sys::mtt_api_v1_t>,
}

impl Api {
    /// Negotiates ABI v1 through an externally resolved `mtt_get_api` symbol.
    ///
    /// # Safety
    ///
    /// `get_api` and every function it returns must implement the complete ABI
    /// v1 contract declared by `magpie_tts_rt.h`: exact signatures, valid and
    /// correctly aligned output pointers, declared buffer lengths and
    /// lifetimes, handle ownership, lease lifetime, and error-buffer semantics.
    /// The library containing those functions must remain loaded for this
    /// `Api` and every handle and lease created from it. Signature and library
    /// lifetime alone are not sufficient.
    pub unsafe fn negotiate(get_api: GetApiFn) -> Result<Self> {
        let mut table = sys::mtt_api_v1_t {
            struct_size: size_of::<sys::mtt_api_v1_t>() as u32,
            abi_version: sys::MTT_ABI_VERSION_1,
            runtime_create: None,
            runtime_destroy: None,
            model_load: None,
            model_destroy: None,
            session_create: None,
            session_destroy: None,
            request_start: None,
            request_poll: None,
            request_wait: None,
            request_cancel: None,
            request_destroy: None,
            audio_acquire: None,
            audio_release: None,
        };

        // SAFETY: the caller guarantees the symbol signature and library lifetime.
        let raw_status = unsafe { get_api(sys::MTT_ABI_VERSION_1, &mut table) };
        let status = Status::from_raw(raw_status);
        if !status.is_ok() {
            return Err(Error::AbiNegotiation { status });
        }

        // SAFETY: the successful negotiation populated function pointers owned by
        // a library whose lifetime is guaranteed by the caller.
        unsafe { Self::from_table(table) }
    }

    /// Constructs an API from a function table supplied by a native library.
    ///
    /// # Safety
    ///
    /// Every non-null function pointer in `table` must implement the complete
    /// ABI v1 contract declared by `magpie_tts_rt.h`, including valid aligned
    /// output memory, buffer lengths and lifetimes, handle/lease ownership, and
    /// error-buffer semantics. Its containing library must outlive all values
    /// constructed from this `Api`.
    pub unsafe fn from_table(table: sys::mtt_api_v1_t) -> Result<Self> {
        if table.struct_size != size_of::<sys::mtt_api_v1_t>() as u32
            || table.abi_version != sys::MTT_ABI_VERSION_1
        {
            return Err(Error::InvalidApiHeader {
                struct_size: table.struct_size,
                abi_version: table.abi_version,
            });
        }

        require_function(table.runtime_create, "runtime_create")?;
        require_function(table.runtime_destroy, "runtime_destroy")?;
        require_function(table.model_load, "model_load")?;
        require_function(table.model_destroy, "model_destroy")?;
        require_function(table.session_create, "session_create")?;
        require_function(table.session_destroy, "session_destroy")?;
        require_function(table.request_start, "request_start")?;
        require_function(table.request_poll, "request_poll")?;
        require_function(table.request_wait, "request_wait")?;
        require_function(table.request_cancel, "request_cancel")?;
        require_function(table.request_destroy, "request_destroy")?;
        require_function(table.audio_acquire, "audio_acquire")?;
        require_function(table.audio_release, "audio_release")?;

        Ok(Self {
            raw: Rc::new(table),
        })
    }

    #[cfg(feature = "native-link")]
    /// Negotiates with the library selected by the `native-link` feature.
    ///
    /// Building this feature requires `MAGPIE_TTS_RT_LIB_DIR` to be an absolute
    /// Linux directory containing `libmagpie_tts_rt.so`.
    /// Runtime library discovery remains the application's responsibility, for
    /// example through its installed rpath or platform loader configuration.
    pub fn linked() -> Result<Self> {
        // SAFETY: the `native-link` feature resolves this symbol at link time and
        // keeps the linked library loaded for the process lifetime.
        unsafe { Self::negotiate(sys::mtt_get_api) }
    }

    pub(crate) fn raw(&self) -> &sys::mtt_api_v1_t {
        &self.raw
    }

    pub(crate) fn error_buffer() -> sys::mtt_error_v1_t {
        sys::mtt_error_v1_t {
            struct_size: size_of::<sys::mtt_error_v1_t>() as u32,
            abi_version: sys::MTT_ABI_VERSION_1,
            code: sys::MTT_STATUS_OK,
            stage: sys::MTT_ERROR_STAGE_NONE,
            message: [0; sys::MTT_ERROR_MESSAGE_CAPACITY as usize],
        }
    }

    pub(crate) fn result_from_call(
        operation: &'static str,
        raw_status: sys::mtt_status_t,
        raw_error: &sys::mtt_error_v1_t,
    ) -> Result<()> {
        let status = Self::validate_call_status(operation, raw_status, raw_error)?;
        if status.is_ok() {
            return Ok(());
        }

        Err(Self::error_from_validated_call(
            operation, status, raw_error,
        ))
    }

    pub(crate) fn validate_call_status(
        operation: &'static str,
        raw_status: sys::mtt_status_t,
        raw_error: &sys::mtt_error_v1_t,
    ) -> Result<Status> {
        let returned_status = Status::from_raw(raw_status);
        let reported_status = Status::from_raw(raw_error.code);
        let reported_stage = ErrorStage::from_raw(raw_error.stage);
        let (message, nul_terminated) = message_bytes(raw_error);

        let invalid_reason = if raw_error.struct_size != size_of::<sys::mtt_error_v1_t>() as u32 {
            Some("error struct_size does not match ABI v1")
        } else if raw_error.abi_version != sys::MTT_ABI_VERSION_1 {
            Some("error abi_version does not match ABI v1")
        } else if !returned_status.is_declared() {
            Some("returned status is not declared by ABI v1")
        } else if reported_status != returned_status {
            Some("error code does not match the returned status")
        } else if !reported_stage.is_declared() {
            Some("error stage is not declared by ABI v1")
        } else if !nul_terminated {
            Some("error message is not NUL-terminated")
        } else if returned_status.is_ok()
            && (reported_stage != ErrorStage::NONE || !message.is_empty())
        {
            Some("successful calls require stage NONE and an empty message")
        } else if !returned_status.is_ok() && reported_stage == ErrorStage::NONE {
            Some("non-success status requires a declared error stage")
        } else if matches!(returned_status, Status::TIMEOUT | Status::WOULD_BLOCK)
            && reported_stage != ErrorStage::REQUEST
        {
            Some("request control status requires stage REQUEST")
        } else if !returned_status.is_ok()
            && !matches!(returned_status, Status::TIMEOUT | Status::WOULD_BLOCK)
            && message.is_empty()
        {
            Some("non-control failure requires a diagnostic message")
        } else {
            None
        };

        if let Some(reason) = invalid_reason {
            return Err(Error::InvalidNativeError {
                operation,
                returned_status,
                struct_size: raw_error.struct_size,
                abi_version: raw_error.abi_version,
                reported_status,
                reported_stage,
                message,
                nul_terminated,
                reason,
            });
        }
        Ok(returned_status)
    }

    pub(crate) fn error_from_validated_call(
        operation: &'static str,
        status: Status,
        raw_error: &sys::mtt_error_v1_t,
    ) -> Error {
        let (message, nul_terminated) = message_bytes(raw_error);
        debug_assert!(nul_terminated);
        Error::Native(NativeError::new(
            operation,
            status,
            ErrorStage::from_raw(raw_error.stage),
            message,
        ))
    }
}

fn require_function<T>(function: Option<T>, name: &'static str) -> Result<()> {
    if function.is_some() {
        Ok(())
    } else {
        Err(Error::MissingApiFunction(name))
    }
}

fn message_bytes(raw: &sys::mtt_error_v1_t) -> (Vec<u8>, bool) {
    let bytes: Vec<u8> = raw
        .message
        .iter()
        .map(|byte| byte.to_ne_bytes()[0])
        .collect();
    match bytes.iter().position(|byte| *byte == 0) {
        Some(terminator) => (bytes[..terminator].to_vec(), true),
        None => (bytes, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_with(
        status: sys::mtt_status_t,
        stage: sys::mtt_error_stage_t,
        message: &[u8],
    ) -> sys::mtt_error_v1_t {
        let mut error = Api::error_buffer();
        error.code = status;
        error.stage = stage;
        for (target, source) in error.message.iter_mut().zip(message.iter().copied()) {
            *target = source as core::ffi::c_char;
        }
        error
    }

    #[test]
    fn rejects_success_with_diagnostic_payload() {
        let error = error_with(sys::MTT_STATUS_OK, sys::MTT_ERROR_STAGE_NONE, b"unexpected");
        assert!(matches!(
            Api::validate_call_status("test", sys::MTT_STATUS_OK, &error),
            Err(Error::InvalidNativeError { .. })
        ));
    }

    #[test]
    fn rejects_control_status_without_request_stage() {
        for status in [sys::MTT_STATUS_WOULD_BLOCK, sys::MTT_STATUS_TIMEOUT] {
            let error = error_with(status, sys::MTT_ERROR_STAGE_NONE, b"");
            assert!(matches!(
                Api::validate_call_status("test", status, &error),
                Err(Error::InvalidNativeError { .. })
            ));
        }
    }

    #[test]
    fn accepts_well_formed_empty_control_status() {
        for status in [sys::MTT_STATUS_WOULD_BLOCK, sys::MTT_STATUS_TIMEOUT] {
            let error = error_with(status, sys::MTT_ERROR_STAGE_REQUEST, b"");
            assert_eq!(
                Api::validate_call_status("test", status, &error).expect("valid control status"),
                Status::from_raw(status)
            );
        }
    }

    #[test]
    fn rejects_unknown_status_and_stage_values() {
        let unknown_status = 10_000;
        let error = error_with(unknown_status, sys::MTT_ERROR_STAGE_REQUEST, b"unknown");
        assert!(matches!(
            Api::validate_call_status("test", unknown_status, &error),
            Err(Error::InvalidNativeError { .. })
        ));

        let error = error_with(sys::MTT_STATUS_IO_ERROR, 10_000, b"unknown");
        assert!(matches!(
            Api::validate_call_status("test", sys::MTT_STATUS_IO_ERROR, &error),
            Err(Error::InvalidNativeError { .. })
        ));
    }
}

use core::mem::{align_of, offset_of, size_of};
use magpie_tts_rt_sys as sys;

#[test]
fn v1_struct_layout_matches_the_public_c_header_on_64_bit_targets() {
    assert_eq!(
        size_of::<usize>(),
        8,
        "v1 layout receipt is for 64-bit targets"
    );

    assert_eq!(size_of::<sys::mtt_error_v1_t>(), 528);
    assert_eq!(align_of::<sys::mtt_error_v1_t>(), 4);
    assert_eq!(offset_of!(sys::mtt_error_v1_t, message), 16);

    assert_eq!(size_of::<sys::mtt_runtime_desc_v1_t>(), 48);
    assert_eq!(align_of::<sys::mtt_runtime_desc_v1_t>(), 8);
    assert_eq!(offset_of!(sys::mtt_runtime_desc_v1_t, reserved), 16);

    assert_eq!(size_of::<sys::mtt_model_desc_v1_t>(), 96);
    assert_eq!(align_of::<sys::mtt_model_desc_v1_t>(), 8);
    assert_eq!(offset_of!(sys::mtt_model_desc_v1_t, bundle_path), 8);
    assert_eq!(
        offset_of!(sys::mtt_model_desc_v1_t, expected_manifest_sha256),
        24
    );
    assert_eq!(offset_of!(sys::mtt_model_desc_v1_t, reserved), 64);

    assert_eq!(size_of::<sys::mtt_session_desc_v1_t>(), 48);
    assert_eq!(align_of::<sys::mtt_session_desc_v1_t>(), 8);
    assert_eq!(offset_of!(sys::mtt_session_desc_v1_t, reserved), 16);

    assert_eq!(size_of::<sys::mtt_request_desc_v1_t>(), 72);
    assert_eq!(align_of::<sys::mtt_request_desc_v1_t>(), 8);
    assert_eq!(offset_of!(sys::mtt_request_desc_v1_t, text_token_ids), 8);
    assert_eq!(offset_of!(sys::mtt_request_desc_v1_t, reserved), 40);

    assert_eq!(size_of::<sys::mtt_request_snapshot_v1_t>(), 600);
    assert_eq!(align_of::<sys::mtt_request_snapshot_v1_t>(), 8);
    assert_eq!(offset_of!(sys::mtt_request_snapshot_v1_t, revision), 8);
    assert_eq!(
        offset_of!(sys::mtt_request_snapshot_v1_t, terminal_status),
        48
    );
    assert_eq!(
        offset_of!(sys::mtt_request_snapshot_v1_t, terminal_error_stage),
        52
    );
    assert_eq!(
        offset_of!(sys::mtt_request_snapshot_v1_t, terminal_error_message),
        56
    );
    assert_eq!(offset_of!(sys::mtt_request_snapshot_v1_t, reserved), 568);

    assert_eq!(size_of::<sys::mtt_audio_lease_v1_t>(), 104);
    assert_eq!(align_of::<sys::mtt_audio_lease_v1_t>(), 8);
    assert_eq!(offset_of!(sys::mtt_audio_lease_v1_t, samples), 16);
    assert_eq!(offset_of!(sys::mtt_audio_lease_v1_t, reserved), 72);

    assert_eq!(size_of::<sys::mtt_api_v1_t>(), 112);
    assert_eq!(align_of::<sys::mtt_api_v1_t>(), 8);
    assert_eq!(offset_of!(sys::mtt_api_v1_t, runtime_create), 8);
    assert_eq!(offset_of!(sys::mtt_api_v1_t, audio_release), 104);
}

#[test]
fn v1_constants_match_the_public_c_header() {
    assert_eq!(sys::MTT_ABI_VERSION_1, 1);
    assert_eq!(sys::MTT_ERROR_MESSAGE_CAPACITY, 512);
    assert_eq!(sys::MTT_SHA256_BYTES, 32);
    assert_eq!(sys::MTT_STATUS_OK, 0);
    assert_eq!(sys::MTT_STATUS_INTERNAL_ERROR, 15);
    assert_eq!(sys::MTT_REQUEST_STATE_RUNNING, 1);
    assert_eq!(sys::MTT_REQUEST_STATE_FAILED, 4);
    assert_eq!(sys::MTT_PCM_FORMAT_F32_MONO, 1);
    assert_eq!(
        sys::MTT_AUDIO_FLAG_FIRST | sys::MTT_AUDIO_FLAG_FINAL | sys::MTT_AUDIO_FLAG_ALIGNMENT_VALID,
        7
    );
}

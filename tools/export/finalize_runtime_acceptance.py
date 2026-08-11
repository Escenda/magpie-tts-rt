#!/usr/bin/env python3
"""Promote exact streaming candidates and create startup golden evidence.

Text Encoder, Main Decoder, and Local AR build receipts are deliberately
unaccepted candidates.  This command authenticates those candidates and an
accepted NanoCodec export, requires an accepted three-fixture closed-loop
sequence receipt, authenticates the deterministic build receipt for the one
plugin shared by every plugin-bearing generation plan, executes the canonical
generated codes through the three stateful NanoCodec plans, and publishes all
promotion and golden receipts in one atomic no-replace directory.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TOOLS = PROJECT_ROOT / "tools" / "export"
BUNDLE_TOOLS = PROJECT_ROOT / "tools" / "bundle"
ORACLE_TOOLS = PROJECT_ROOT / "tools" / "oracle"
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))
if str(BUNDLE_TOOLS) not in sys.path:
    sys.path.insert(0, str(BUNDLE_TOOLS))
if str(ORACLE_TOOLS) not in sys.path:
    sys.path.insert(0, str(ORACLE_TOOLS))

from cublas_runtime_identity import (  # noqa: E402
    CublasRuntimeIdentity,
    parse_cublas_runtime_identity,
)
from cuda_runtime_identity import (  # noqa: E402
    CudaRuntimeIdentity,
    parse_cuda_runtime_identity,
)

from export_main_decoder import (  # noqa: E402
    PREFILL_PLAN,
    STEP_PLAN,
    canonical_json_bytes,
    load_fixture,
    publish_directory_no_replace,
    tensor_from_fixture,
)
from export_nanocodec import (  # noqa: E402
    INITIAL_FRAMES,
    PLAN_FILES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
    STEADY_FRAMES,
    TAIL_MAX_FRAMES,
    TAIL_MIN_FRAMES,
    TENSORRT_FP32_ATOL,
    TENSORRT_FP32_RTOL,
    StateContract,
    compare_tensor,
    execute_plan,
    inspect_plan,
    route_inputs,
    tensor_from_fixture as nanocodec_tensor_from_fixture,
    verify_nanocodec_fixture,
)
from nanocodec_contract import CANONICAL_STATE_BINDINGS  # noqa: E402
from package_runtime_bundle import (  # noqa: E402
    FileArtifact,
    ZERO_FRAME_FINALIZATION,
    canonical_json_bytes as compact_json_bytes,
    tokenizer_identity_projection,
    validate_plugin_build_receipt,
    validate_plugin_build_source_tree,
)
from source_acceptance_receipt import (  # noqa: E402
    validate_public_acceptance,
)
from validate_main_decoder_plans import (  # noqa: E402
    AuthenticatedCodecRestore,
    AuthenticatedPlanExport,
    JsonValue,
    authenticate_codec_restore,
    authenticate_plan_export,
    load_json,
    require_integer,
    require_list,
    require_mapping,
    require_nonnegative_integer,
    require_safe_artifact_path,
    require_sha256,
    require_string,
    sha256_bytes,
    sha256_file,
)
from validate_main_local_ar_sequence import (  # noqa: E402
    AuthenticatedLocalARExport,
    AuthenticatedTextEncoderExport,
    LOCAL_AR_PLAN,
    LOCAL_AR_PLUGIN,
    SEQUENCE_RECEIPT,
    SEQUENCE_RECEIPT_CHECKSUM,
    authenticate_local_ar_export,
    authenticate_text_encoder_export,
)


NANOCODEC_RECEIPT = "export-receipt.json"
NANOCODEC_RECEIPT_CHECKSUM = "export-receipt.json.sha256"
NANOCODEC_CONTRACT = "nanocodec-contract.json"
TEXT_PROMOTION_RECEIPT = "text-encoder-acceptance-receipt.json"
MAIN_PROMOTION_RECEIPT = "main-decoder-acceptance-receipt.json"
LOCAL_PROMOTION_RECEIPT = "local-ar-acceptance-receipt.json"
GOLDEN_FIXTURE = "golden-fixture.json"
GOLDEN_RECEIPT = "golden-receipt.json"
CONSOLIDATED_RECEIPT = "consolidated-export-receipt.json"
PLUGIN_BUILD_RECEIPT = "plugin-build-receipt.json"
GOLDEN_PCM = "golden.pcm.f32le.bin"
MINIMUM_SEQUENCE_FIXTURES = 3
ENGINE_ROLES = (
    "text_encoder",
    "main_decoder_prefill",
    "main_decoder_step",
    "local_ar_16",
    "nanocodec_initial_4",
    "nanocodec_steady_8",
    "nanocodec_tail_1_8",
)


@dataclass(frozen=True)
class AuthenticatedArtifact:
    path: Path
    relative_path: str
    size_bytes: int
    sha256: str

    def file_reference(self) -> dict[str, JsonValue]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class AuthenticatedReceiptDirectory:
    root: Path
    receipt: dict[str, JsonValue]
    receipt_sha256: str
    artifacts: dict[str, AuthenticatedArtifact]


@dataclass(frozen=True)
class AuthenticatedSequenceReceipt:
    receipt_sha256: str
    locked_magpie_restore_sha256: str
    codec_restore: AuthenticatedCodecRestore
    cuda_identity: CudaRuntimeIdentity
    cublas_identity: CublasRuntimeIdentity
    mode8_class_table_sha256: str


@dataclass(frozen=True)
class AuthenticatedNanoCodecExport:
    root: Path
    receipt_sha256: str
    oracle_lock_sha256: str
    canonical_fixture_manifest_sha256: str
    runtime: dict[str, JsonValue]
    plans: dict[str, AuthenticatedArtifact]
    contract: AuthenticatedArtifact


@dataclass(frozen=True)
class GoldenHashes:
    normalized_text_sha256: str
    token_ids_sha256: str
    baked_context_sha256: str
    decoder_tokens_sha256: str
    codec_codes_sha256: str
    codec_frame_count: int
    pcm_f32le_sha256: str
    pcm_sample_count: int


def require_common_plugin_sha256(
    *,
    text_encoder: str,
    main_decoder: str,
    local_ar: str,
) -> None:
    if text_encoder != local_ar or main_decoder != local_ar:
        raise RuntimeError(
            "Text Encoder, Main Decoder, and Local AR require different plugins"
        )


def utc_timestamp() -> str:
    return (
        datetime.datetime.now(datetime.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    return resolved


def authenticate_receipt_directory(
    path: Path,
    *,
    receipt_name: str,
    checksum_name: str,
    artifact_role: str,
    status: str,
) -> AuthenticatedReceiptDirectory:
    if path.is_symlink():
        raise RuntimeError(f"receipt directory must not be a symbolic link: {path}")
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"receipt path is not a directory: {root}")
    receipt_path = require_regular_file(root / receipt_name, receipt_name)
    checksum_path = require_regular_file(root / checksum_name, checksum_name)
    receipt, receipt_payload = load_json(receipt_path)
    receipt_sha256 = sha256_bytes(receipt_payload)
    expected_checksum = f"{receipt_sha256}  {receipt_name}\n".encode("ascii")
    if checksum_path.read_bytes() != expected_checksum:
        raise RuntimeError(f"{checksum_name} does not authenticate {receipt_name}")
    if require_integer(receipt.get("schema_version"), "receipt.schema_version") != 1:
        raise RuntimeError("unsupported receipt schema")
    if (
        require_string(receipt.get("artifact_role"), "receipt.artifact_role")
        != artifact_role
    ):
        raise RuntimeError(
            f"receipt artifact role mismatch: expected {artifact_role}"
        )
    if require_string(receipt.get("status"), "receipt.status") != status:
        raise RuntimeError(f"receipt status mismatch: expected {status}")

    expected_files = {receipt_name, checksum_name}
    artifacts: dict[str, AuthenticatedArtifact] = {}
    for index, raw_record in enumerate(
        require_list(receipt.get("artifacts"), "receipt.artifacts")
    ):
        label = f"receipt.artifacts[{index}]"
        record = require_mapping(raw_record, label)
        relative = require_safe_artifact_path(record.get("path"), f"{label}.path")
        if relative in artifacts:
            raise RuntimeError(f"duplicate receipt artifact: {relative}")
        expected_size = require_nonnegative_integer(
            record.get("size_bytes"),
            f"{label}.size_bytes",
        )
        expected_sha256 = require_sha256(
            record.get("sha256"),
            f"{label}.sha256",
        )
        unresolved = root / PurePosixPath(relative)
        artifact_path = require_regular_file(unresolved, relative)
        if not artifact_path.is_relative_to(root):
            raise RuntimeError(f"receipt artifact escapes its root: {relative}")
        if artifact_path.stat().st_size != expected_size:
            raise RuntimeError(f"receipt artifact size mismatch: {relative}")
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"receipt artifact SHA-256 mismatch: {relative}")
        artifacts[relative] = AuthenticatedArtifact(
            path=artifact_path,
            relative_path=relative,
            size_bytes=expected_size,
            sha256=actual_sha256,
        )
        expected_files.add(relative)

    actual_files: set[str] = set()
    actual_directories = {""}
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"receipt directory contains symlink: {relative}")
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(
                f"receipt directory contains non-file entry: {relative}"
            )
    expected_directories = {""}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            "receipt directory entry set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}, "
            "extra_directories="
            f"{sorted(actual_directories - expected_directories)}"
        )
    return AuthenticatedReceiptDirectory(
        root=root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        artifacts=artifacts,
    )


def authenticate_nanocodec_export(
    path: Path,
    *,
    lock_sha256: str,
    canonical_fixture_manifest_sha256: str,
) -> AuthenticatedNanoCodecExport:
    authenticated = authenticate_receipt_directory(
        path,
        receipt_name=NANOCODEC_RECEIPT,
        checksum_name=NANOCODEC_RECEIPT_CHECKSUM,
        artifact_role="stateful_nanocodec",
        status="accepted",
    )
    source = require_mapping(
        authenticated.receipt.get("source"),
        "NanoCodec receipt.source",
    )
    actual_lock = require_sha256(
        source.get("oracle_lock_sha256"),
        "NanoCodec receipt.source.oracle_lock_sha256",
    )
    actual_fixture = require_sha256(
        source.get("boundary_fixture_manifest_sha256"),
        "NanoCodec receipt.source.boundary_fixture_manifest_sha256",
    )
    if actual_lock != lock_sha256:
        raise RuntimeError("NanoCodec export uses a different oracle lock")
    if actual_fixture != canonical_fixture_manifest_sha256:
        raise RuntimeError("NanoCodec export uses a different canonical fixture")
    plans: dict[str, AuthenticatedArtifact] = {}
    for route, relative in PLAN_FILES.items():
        try:
            plans[route] = authenticated.artifacts[relative]
        except KeyError as error:
            raise RuntimeError(
                f"NanoCodec export is missing plan {relative}"
            ) from error
    try:
        contract = authenticated.artifacts[NANOCODEC_CONTRACT]
    except KeyError as error:
        raise RuntimeError("NanoCodec export is missing its state contract") from error
    return AuthenticatedNanoCodecExport(
        root=authenticated.root,
        receipt_sha256=authenticated.receipt_sha256,
        oracle_lock_sha256=actual_lock,
        canonical_fixture_manifest_sha256=actual_fixture,
        runtime=require_mapping(
            authenticated.receipt.get("runtime"),
            "NanoCodec receipt.runtime",
        ),
        plans=plans,
        contract=contract,
    )


def authenticate_sequence_receipt(
    path: Path,
    *,
    lock_sha256: str,
    canonical_fixture_manifest_sha256: str,
    text: AuthenticatedTextEncoderExport,
    main: AuthenticatedPlanExport,
    local: AuthenticatedLocalARExport,
) -> AuthenticatedSequenceReceipt:
    authenticated = authenticate_receipt_directory(
        path,
        receipt_name=SEQUENCE_RECEIPT,
        checksum_name=SEQUENCE_RECEIPT_CHECKSUM,
        artifact_role="text_main_local_ar_sequence_validation",
        status="accepted",
    )
    receipt = authenticated.receipt
    session_policy = require_mapping(
        receipt.get("session_policy"),
        "sequence.session_policy",
    )
    if (
        session_policy.get("main_execution_status_recurrence")
        != "int32-device-scalar-sticky-12-layer"
        or session_policy.get(
            "main_execution_status_checked_before_next_local_ar"
        )
        is not True
    ):
        raise RuntimeError(
            "sequence receipt lacks the canonical Main execution-status gate"
        )
    fixture_count = require_nonnegative_integer(
        receipt.get("fixture_count"),
        "sequence.fixture_count",
    )
    exact_count = require_nonnegative_integer(
        receipt.get("exact_code_case_count"),
        "sequence.exact_code_case_count",
    )
    fixtures = require_list(receipt.get("fixtures"), "sequence.fixtures")
    if (
        fixture_count != len(fixtures)
        or fixture_count < MINIMUM_SEQUENCE_FIXTURES
        or exact_count != fixture_count
        or receipt.get("all_codes_exact") is not True
    ):
        raise RuntimeError(
            "sequence receipt does not prove exact generation for every fixture"
        )
    manifest_sha256s: set[str] = set()
    canonical_found = False
    for index, raw_case in enumerate(fixtures):
        case = require_mapping(raw_case, f"sequence.fixtures[{index}]")
        manifest_sha256 = require_sha256(
            case.get("fixture_manifest_sha256"),
            f"sequence.fixtures[{index}].fixture_manifest_sha256",
        )
        if manifest_sha256 in manifest_sha256s:
            raise RuntimeError(
                f"sequence receipt duplicates fixture {manifest_sha256}"
            )
        manifest_sha256s.add(manifest_sha256)
        if case.get("code_exact") is not True:
            raise RuntimeError(
                f"sequence fixture is not exact: {manifest_sha256}"
            )
        local_ar_invocations = require_nonnegative_integer(
            case.get("local_ar_invocations"),
            f"sequence.fixtures[{index}].local_ar_invocations",
        )
        status_checks = require_nonnegative_integer(
            case.get("main_execution_status_check_count"),
            (
                f"sequence.fixtures[{index}]."
                "main_execution_status_check_count"
            ),
        )
        if (
            case.get("main_execution_status_all_zero") is not True
            or local_ar_invocations < 1
            or status_checks != local_ar_invocations - 1
        ):
            raise RuntimeError(
                "sequence fixture does not prove zero Main execution status "
                f"at every generation boundary: {manifest_sha256}"
            )
        generated_sha256 = require_sha256(
            case.get("generated_codes_sha256"),
            f"sequence.fixtures[{index}].generated_codes_sha256",
        )
        expected_sha256 = require_sha256(
            case.get("expected_codes_sha256"),
            f"sequence.fixtures[{index}].expected_codes_sha256",
        )
        if generated_sha256 != expected_sha256:
            raise RuntimeError(
                f"sequence fixture code digests differ: {manifest_sha256}"
            )
        if manifest_sha256 == canonical_fixture_manifest_sha256:
            canonical_found = True
    if not canonical_found:
        raise RuntimeError("sequence receipt omits the canonical golden fixture")

    source = require_mapping(receipt.get("source"), "sequence.source")
    locked_magpie_restore_sha256 = require_sha256(
        source.get("locked_magpie_restore_sha256"),
        "sequence.source.locked_magpie_restore_sha256",
    )
    expected_restore_sha256 = sha256_file(
        (EXPORT_TOOLS / "locked_magpie_restore.py").resolve(strict=True)
    )
    if (
        locked_magpie_restore_sha256 != expected_restore_sha256
        or text.locked_magpie_restore_sha256 != expected_restore_sha256
        or main.locked_magpie_restore_sha256 != expected_restore_sha256
        or local.locked_magpie_restore_sha256 != expected_restore_sha256
    ):
        raise RuntimeError(
            "sequence/export locked Magpie restore helper digest mismatch"
        )
    codec_restore = authenticate_codec_restore(source, "sequence.source")
    if (
        codec_restore != text.codec_restore
        or codec_restore != main.codec_restore
        or codec_restore != local.codec_restore
    ):
        raise RuntimeError("sequence/export codec restore identity mismatch")
    expected_source = {
        "oracle_lock_sha256": lock_sha256,
        "text_encoder_export_receipt_sha256": text.receipt_sha256,
        "text_encoder_plan_sha256": text.plan_sha256,
        "main_export_receipt_sha256": main.receipt_sha256,
        "main_mode8_validation_receipt_sha256": (
            main.mode8_validation_receipt_sha256
        ),
        "main_prefill_plan_sha256": main.prefill_plan_sha256,
        "main_step_plan_sha256": main.step_plan_sha256,
        "local_ar_export_receipt_sha256": local.receipt_sha256,
        "local_ar_plan_sha256": local.plan_sha256,
        "local_ar_plugin_sha256": local.plugin_sha256,
    }
    for key, expected in expected_source.items():
        actual = require_sha256(source.get(key), f"sequence.source.{key}")
        if actual != expected:
            raise RuntimeError(
                f"sequence source {key} mismatch: expected {expected}, got {actual}"
            )
    runtime = require_mapping(receipt.get("runtime"), "sequence.runtime")
    cuda_identity = parse_cuda_runtime_identity(
        runtime.get("cuda"),
        "sequence.runtime.cuda",
    )
    cublas_identity = parse_cublas_runtime_identity(
        runtime.get("cublas"),
        "sequence.runtime.cublas",
    )
    mode8_class_table_sha256 = require_sha256(
        runtime.get("mode8_class_table_sha256"),
        "sequence.runtime.mode8_class_table_sha256",
    )
    if mode8_class_table_sha256 != main.mode8_class_table_sha256:
        raise RuntimeError(
            "sequence mode-8 class-table digest differs from the "
            "authenticated Main Decoder export"
        )
    return AuthenticatedSequenceReceipt(
        receipt_sha256=authenticated.receipt_sha256,
        locked_magpie_restore_sha256=locked_magpie_restore_sha256,
        codec_restore=codec_restore,
        cuda_identity=cuda_identity,
        cublas_identity=cublas_identity,
        mode8_class_table_sha256=mode8_class_table_sha256,
    )


def require_locked_source_receipt(
    path: Path,
    lock: dict[str, JsonValue],
) -> AuthenticatedArtifact:
    acceptance = require_mapping(lock.get("acceptance"), "lock.acceptance")
    resolved = require_regular_file(path, "source model acceptance receipt")
    expected_sha256 = require_sha256(
        acceptance.get("receipt_sha256"),
        "lock.acceptance.receipt_sha256",
    )
    expected_size = require_nonnegative_integer(
        acceptance.get("receipt_size_bytes"),
        "lock.acceptance.receipt_size_bytes",
    )
    expected_name = require_string(
        acceptance.get("receipt_name"),
        "lock.acceptance.receipt_name",
    )
    if resolved.name != expected_name:
        raise RuntimeError(
            "source model acceptance receipt name differs from oracle lock"
        )
    actual_sha256 = sha256_file(resolved)
    if resolved.stat().st_size != expected_size or actual_sha256 != expected_sha256:
        raise RuntimeError("source model acceptance receipt differs from oracle lock")
    receipt, payload = load_json(resolved)
    if len(payload) != expected_size or sha256_bytes(payload) != expected_sha256:
        raise RuntimeError(
            "source model acceptance receipt changed while it was parsed"
        )
    validate_public_acceptance(receipt, lock)
    return AuthenticatedArtifact(
        path=resolved,
        relative_path=resolved.name,
        size_bytes=expected_size,
        sha256=actual_sha256,
    )


def tokenizer_identity(
    path: Path,
) -> tuple[AuthenticatedArtifact, str]:
    resolved = require_regular_file(path, "tokenizer identity receipt")
    receipt, payload = load_json(resolved)
    if require_integer(receipt.get("schema_version"), "tokenizer.schema_version") != 1:
        raise RuntimeError("unsupported tokenizer identity receipt schema")
    projection = tokenizer_identity_projection(receipt)
    identity_sha256 = sha256_bytes(compact_json_bytes(projection))
    return (
        AuthenticatedArtifact(
            path=resolved,
            relative_path=resolved.name,
            size_bytes=len(payload),
            sha256=sha256_bytes(payload),
        ),
        identity_sha256,
    )


def little_endian_int32_values(payload: bytes) -> tuple[int, ...]:
    if not payload or len(payload) % 4 != 0:
        raise RuntimeError("prepared token payload is not non-empty INT32")
    return tuple(
        int.from_bytes(payload[offset : offset + 4], "little", signed=True)
        for offset in range(0, len(payload), 4)
    )


def chunked_codec_code_bytes(
    complete_codes,
    frame_schedule: tuple[int, ...],
) -> bytes:
    import numpy

    if (
        complete_codes.ndim != 3
        or complete_codes.shape[0] != 1
        or complete_codes.shape[1] != 8
        or complete_codes.shape[2] != sum(frame_schedule)
    ):
        raise RuntimeError(
            f"complete codec code shape does not match schedule: "
            f"{complete_codes.shape}/{frame_schedule}"
        )
    offset = 0
    chunks: list[bytes] = []
    for frame_count in frame_schedule:
        chunk = complete_codes[:, :, offset : offset + frame_count]
        chunks.append(
            numpy.ascontiguousarray(chunk, dtype="<i8").tobytes()
        )
        offset += frame_count
    return b"".join(chunks)


def execute_golden_nanocodec(
    *,
    torch,
    tensorrt,
    numpy,
    export: AuthenticatedNanoCodecExport,
    fixture,
    staging: Path,
) -> tuple[str, int]:
    state_contracts = tuple(
        StateContract(binding.logical_name, binding.shape)
        for binding in CANONICAL_STATE_BINDINGS
    )
    runtimes = []
    engines = {}
    for route in ("initial_4", "steady_8", "tail_1_8"):
        runtime, engine, _ = inspect_plan(
            tensorrt,
            export.plans[route].path,
            route=route,
            state_contracts=state_contracts,
        )
        runtimes.append(runtime)
        engines[route] = engine

    codes_record = fixture.records["generation.codes"]
    complete_codes = nanocodec_tensor_from_fixture(
        torch,
        numpy,
        codes_record,
    )
    current_states = None
    output_chunks: list[bytes] = []
    contexts = []
    frame_offset = 0
    for sequence, frame_count in enumerate(fixture.frame_schedule):
        if sequence == 0:
            route = "initial_4"
            if frame_count != INITIAL_FRAMES:
                raise RuntimeError("golden schedule does not start with four frames")
        elif sequence == len(fixture.frame_schedule) - 1:
            route = "tail_1_8"
            if frame_count < TAIL_MIN_FRAMES or frame_count > TAIL_MAX_FRAMES:
                raise RuntimeError("golden terminal chunk is outside tail profile")
        else:
            route = "steady_8"
            if frame_count != STEADY_FRAMES:
                raise RuntimeError("golden steady chunk is not eight frames")
        codec_tokens = complete_codes[
            :,
            :,
            frame_offset : frame_offset + frame_count,
        ].contiguous()
        outputs, context = execute_plan(
            torch,
            tensorrt,
            engines[route],
            route_inputs(
                state_contracts,
                codec_tokens,
                current_states,
            ),
        )
        contexts.append(context)
        expected_names = {
            "pcm",
            "valid_sample_length",
            *(binding.output_binding for binding in state_contracts),
        }
        if set(outputs) != expected_names:
            raise RuntimeError(f"NanoCodec {route} output set mismatch")
        valid_samples = int(outputs["valid_sample_length"].cpu().item())
        expected_samples = frame_count * SAMPLES_PER_FRAME
        if valid_samples != expected_samples:
            raise RuntimeError(
                f"NanoCodec {route} valid length mismatch: "
                f"{valid_samples} != {expected_samples}"
            )
        expected_pcm = nanocodec_tensor_from_fixture(
            torch,
            numpy,
            fixture.records[f"codec.chunk_{sequence:03d}.pcm"],
        )
        actual_pcm = outputs["pcm"].to(device="cuda")[:, :valid_samples]
        compare_tensor(
            torch,
            actual_pcm,
            expected_pcm,
            name=f"golden.chunk_{sequence:03d}.pcm",
            absolute_tolerance=TENSORRT_FP32_ATOL,
            relative_tolerance=TENSORRT_FP32_RTOL,
        )
        output_chunks.append(
            numpy.ascontiguousarray(
                actual_pcm.cpu().numpy(),
                dtype="<f4",
            ).tobytes()
        )
        current_states = tuple(
            outputs[binding.output_binding].to(device="cuda")
            for binding in state_contracts
        )
        frame_offset += frame_count
    if frame_offset != fixture.valid_codec_frames:
        raise RuntimeError("NanoCodec golden execution did not consume every frame")
    pcm_payload = b"".join(output_chunks)
    sample_count = len(pcm_payload) // 4
    if sample_count != fixture.valid_codec_frames * SAMPLES_PER_FRAME:
        raise RuntimeError("NanoCodec golden PCM sample count is inconsistent")
    expected_complete_pcm = nanocodec_tensor_from_fixture(
        torch,
        numpy,
        fixture.records["codec.complete_pcm"],
    )
    actual_complete_pcm = torch.from_numpy(
        numpy.frombuffer(pcm_payload, dtype="<f4").copy()
    ).reshape(expected_complete_pcm.shape).to(device="cuda")
    compare_tensor(
        torch,
        actual_complete_pcm,
        expected_complete_pcm,
        name="golden.complete_pcm",
        absolute_tolerance=TENSORRT_FP32_ATOL,
        relative_tolerance=TENSORRT_FP32_RTOL,
    )
    (staging / GOLDEN_PCM).write_bytes(pcm_payload)
    if len(runtimes) != 3 or len(contexts) != len(fixture.frame_schedule):
        raise AssertionError("NanoCodec TensorRT ownership is incomplete")
    return sha256_bytes(pcm_payload), sample_count


def build_golden_documents(
    *,
    fixture,
    frame_schedule: tuple[int, ...],
    valid_codec_frames: int,
    manifest: dict[str, JsonValue],
    tokenizer_identity_sha256: str,
    lock_sha256: str,
    pcm_sha256: str,
    pcm_sample_count: int,
    created_at_utc: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], GoldenHashes]:
    token_record = fixture.tensors["input.text_token_ids"]
    token_payload = token_record.path.read_bytes()
    token_ids = little_endian_int32_values(token_payload)
    if len(token_ids) != fixture.text_tokens or any(token_id < 0 for token_id in token_ids):
        raise RuntimeError("canonical prepared token IDs are invalid")
    text = require_string(manifest.get("text"), "fixture.text")
    normalized_text_sha256 = sha256_bytes(text.encode("utf-8"))
    baked_context = fixture.tensors.get("cfg.sofia_prefix")
    if (
        baked_context is None
        or baked_context.dtype != "bf16"
        or baked_context.shape != (1, 217, 768)
    ):
        raise RuntimeError("canonical Sofia baked context contract mismatch")
    generation_codes = fixture.tensors["generation.codes"]
    if generation_codes.dtype != "int64":
        raise RuntimeError("canonical generated codes are not INT64")

    import numpy

    complete_codes = numpy.fromfile(generation_codes.path, dtype="<i8").reshape(
        generation_codes.shape
    )
    codec_code_payload = chunked_codec_code_bytes(
        complete_codes,
        frame_schedule,
    )
    hashes = GoldenHashes(
        normalized_text_sha256=normalized_text_sha256,
        token_ids_sha256=sha256_bytes(token_payload),
        baked_context_sha256=baked_context.sha256,
        decoder_tokens_sha256=sha256_file(generation_codes.path),
        codec_codes_sha256=sha256_bytes(codec_code_payload),
        codec_frame_count=valid_codec_frames,
        pcm_f32le_sha256=pcm_sha256,
        pcm_sample_count=pcm_sample_count,
    )
    seed = require_nonnegative_integer(
        manifest.get("local_ar_seed"),
        "fixture.local_ar_seed",
    )
    if seed >= 2**32:
        raise RuntimeError("fixture.local_ar_seed is outside uint32")
    fixture_id = require_string(manifest.get("fixture_id"), "fixture.fixture_id")
    golden_fixture: dict[str, JsonValue] = {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "prepared_token_ids": list(token_ids),
        "seed": seed,
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "oracle_lock_sha256": lock_sha256,
        "normalized_text_sha256": hashes.normalized_text_sha256,
        "token_ids_sha256": hashes.token_ids_sha256,
        "baked_context_sha256": hashes.baked_context_sha256,
        "expected": {
            "decoder_tokens_sha256": hashes.decoder_tokens_sha256,
            "codec_codes_sha256": hashes.codec_codes_sha256,
            "codec_frame_count": hashes.codec_frame_count,
            "pcm_f32le_sha256": hashes.pcm_f32le_sha256,
            "pcm_sample_count": hashes.pcm_sample_count,
        },
    }
    golden_receipt: dict[str, JsonValue] = {
        "receipt_version": 1,
        "created_at_utc": created_at_utc,
        "normalized_text_sha256": hashes.normalized_text_sha256,
        "token_ids_sha256": hashes.token_ids_sha256,
        "baked_context_sha256": hashes.baked_context_sha256,
        "seed": seed,
        "decoder_tokens_sha256": hashes.decoder_tokens_sha256,
        "codec_codes_sha256": hashes.codec_codes_sha256,
        "codec_frame_count": hashes.codec_frame_count,
        "pcm_f32le_sha256": hashes.pcm_f32le_sha256,
        "sample_count": hashes.pcm_sample_count,
        "initial_frames": INITIAL_FRAMES,
        "steady_frames": STEADY_FRAMES,
        "tail_min_frames": TAIL_MIN_FRAMES,
        "tail_max_frames": TAIL_MAX_FRAMES,
        "eos_frame_is_audio": False,
        "zero_frame_finalization": ZERO_FRAME_FINALIZATION,
    }
    return golden_fixture, golden_receipt, hashes


def write_receipt(
    staging: Path,
    name: str,
    document: dict[str, JsonValue],
) -> AuthenticatedArtifact:
    payload = canonical_json_bytes(document)
    path = staging / name
    path.write_bytes(payload)
    checksum_name = f"{name}.sha256"
    (staging / checksum_name).write_text(
        f"{sha256_bytes(payload)}  {name}\n",
        encoding="ascii",
    )
    return AuthenticatedArtifact(
        path=path,
        relative_path=name,
        size_bytes=len(payload),
        sha256=sha256_bytes(payload),
    )


def promotion_receipt(
    *,
    artifact_role: str,
    candidate_receipt_sha256: str,
    sequence_receipt_sha256: str,
    lock_sha256: str,
    canonical_fixture_manifest_sha256: str,
    engines: list[tuple[str, AuthenticatedArtifact]],
    created_at_utc: str,
    plugin: AuthenticatedArtifact | None,
) -> dict[str, JsonValue]:
    document: dict[str, JsonValue] = {
        "schema_version": 1,
        "artifact_role": artifact_role,
        "status": "accepted",
        "created_at_utc": created_at_utc,
        "source": {
            "candidate_receipt_sha256": candidate_receipt_sha256,
            "sequence_receipt_sha256": sequence_receipt_sha256,
            "oracle_lock_sha256": lock_sha256,
            "canonical_fixture_manifest_sha256": (
                canonical_fixture_manifest_sha256
            ),
        },
        "acceptance": {
            "method": "three_predeclared_japanese_closed_loop_exact_v1",
            "fixture_count": 3,
            "generated_codes": "bit_exact",
            "eos_boundaries": "exact",
            "rng_counters": "exact",
            "alignment": "monotonic",
        },
        "engines": [
            {
                "role": role,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for role, artifact in engines
        ],
    }
    if plugin is not None:
        document["plugin"] = {
            "sha256": plugin.sha256,
            "size_bytes": plugin.size_bytes,
        }
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--sequence-validation", type=Path, required=True)
    parser.add_argument("--text-encoder-export", type=Path, required=True)
    parser.add_argument("--main-export", type=Path, required=True)
    parser.add_argument("--local-ar-export", type=Path, required=True)
    parser.add_argument("--nanocodec-export", type=Path, required=True)
    parser.add_argument(
        "--source-model-acceptance-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-identity-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--plugin-build-receipt", type=Path, required=True)
    parser.add_argument("--tensorrt-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_path = require_regular_file(args.lock, "oracle lock")
    lock, lock_payload = load_json(lock_path)
    lock_sha256 = sha256_bytes(lock_payload)
    model = require_mapping(lock.get("model"), "lock.model")
    model_id = require_string(model.get("model_id"), "lock.model.model_id")
    model_version = require_string(
        model.get("version"),
        "lock.model.version",
    )
    model_revision = require_string(
        model.get("revision"),
        "lock.model.revision",
    )
    if (
        len(model_revision) != 40
        or any(character not in "0123456789abcdef" for character in model_revision)
    ):
        raise RuntimeError("lock.model.revision must be a lowercase Git SHA-1")
    model_sha256 = require_sha256(model.get("sha256"), "lock.model.sha256")

    fixture = load_fixture(args.fixture, lock_path)
    canonical_fixture_manifest_sha256 = fixture.manifest_sha256
    manifest, _ = load_json(fixture.root / "manifest.json")
    text = authenticate_text_encoder_export(args.text_encoder_export)
    main_export = authenticate_plan_export(args.main_export)
    local = authenticate_local_ar_export(args.local_ar_export)
    if (
        text.oracle_lock_sha256 != lock_sha256
        or main_export.oracle_lock_sha256 != lock_sha256
        or local.oracle_lock_sha256 != lock_sha256
    ):
        raise RuntimeError("candidate component oracle locks differ")
    if (
        text.source_fixture_manifest_sha256
        != canonical_fixture_manifest_sha256
        or main_export.source_fixture_manifest_sha256
        != canonical_fixture_manifest_sha256
        or canonical_fixture_manifest_sha256
        not in local.source_fixture_manifest_sha256s
    ):
        raise RuntimeError("candidate component canonical fixtures differ")
    require_common_plugin_sha256(
        text_encoder=text.required_plugin_sha256,
        main_decoder=main_export.required_plugin_sha256,
        local_ar=local.plugin_sha256,
    )
    sequence = authenticate_sequence_receipt(
        args.sequence_validation,
        lock_sha256=lock_sha256,
        canonical_fixture_manifest_sha256=canonical_fixture_manifest_sha256,
        text=text,
        main=main_export,
        local=local,
    )
    nanocodec = authenticate_nanocodec_export(
        args.nanocodec_export,
        lock_sha256=lock_sha256,
        canonical_fixture_manifest_sha256=canonical_fixture_manifest_sha256,
    )
    source_model_receipt = require_locked_source_receipt(
        args.source_model_acceptance_receipt,
        lock,
    )
    tokenizer_receipt, tokenizer_identity_sha256 = tokenizer_identity(
        args.tokenizer_identity_receipt
    )
    plugin_input = require_regular_file(local.plugin, "runtime plugin")
    plugin_build_receipt_input = require_regular_file(
        args.plugin_build_receipt,
        "plugin build receipt",
    )
    plugin_build_evidence = validate_plugin_build_receipt(
        plugin_build_receipt_input,
        FileArtifact(
            path=plugin_input.name,
            sha256=local.plugin_sha256,
            size_bytes=plugin_input.stat().st_size,
        ),
    )
    validate_plugin_build_source_tree(
        plugin_build_evidence,
        PROJECT_ROOT,
    )
    if sequence.cublas_identity != plugin_build_evidence.cublas_identity:
        raise RuntimeError(
            "sequence cuBLAS identity differs from the authenticated "
            "plugin build receipt"
        )
    if sequence.cuda_identity != main_export.cuda_identity:
        raise RuntimeError(
            "sequence CUDA runtime identity differs from the authenticated "
            "Main Decoder export"
        )
    if sequence.cublas_identity != main_export.cublas_identity:
        raise RuntimeError(
            "sequence cuBLAS identity differs from the authenticated Main "
            "Decoder export"
        )

    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        import numpy
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; final acceptance is forbidden")
        if torch.cuda.get_device_capability(0) != (11, 0):
            raise RuntimeError(
                "final acceptance requires Thor sm_110, got "
                f"{torch.cuda.get_device_capability(0)}"
            )
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from export_main_decoder import import_tensorrt

        tensorrt = import_tensorrt(args.tensorrt_python_path)
        nanocodec_fixture = verify_nanocodec_fixture(
            fixture.root,
            lock,
            lock_sha256,
        )
        pcm_sha256, pcm_sample_count = execute_golden_nanocodec(
            torch=torch,
            tensorrt=tensorrt,
            numpy=numpy,
            export=nanocodec,
            fixture=nanocodec_fixture,
            staging=staging,
        )
        created_at_utc = utc_timestamp()
        golden_fixture, golden_receipt, hashes = build_golden_documents(
            fixture=fixture,
            frame_schedule=nanocodec_fixture.frame_schedule,
            valid_codec_frames=nanocodec_fixture.valid_codec_frames,
            manifest=manifest,
            tokenizer_identity_sha256=tokenizer_identity_sha256,
            lock_sha256=lock_sha256,
            pcm_sha256=pcm_sha256,
            pcm_sample_count=pcm_sample_count,
            created_at_utc=created_at_utc,
        )
        golden_fixture_artifact = write_receipt(
            staging,
            GOLDEN_FIXTURE,
            golden_fixture,
        )
        golden_receipt_artifact = write_receipt(
            staging,
            GOLDEN_RECEIPT,
            golden_receipt,
        )

        text_plan = AuthenticatedArtifact(
            path=text.plan,
            relative_path=text.plan.name,
            size_bytes=text.plan.stat().st_size,
            sha256=text.plan_sha256,
        )
        main_prefill = AuthenticatedArtifact(
            path=main_export.prefill_plan,
            relative_path=PREFILL_PLAN,
            size_bytes=main_export.prefill_plan.stat().st_size,
            sha256=main_export.prefill_plan_sha256,
        )
        main_step = AuthenticatedArtifact(
            path=main_export.step_plan,
            relative_path=STEP_PLAN,
            size_bytes=main_export.step_plan.stat().st_size,
            sha256=main_export.step_plan_sha256,
        )
        local_plan = AuthenticatedArtifact(
            path=local.plan,
            relative_path=LOCAL_AR_PLAN,
            size_bytes=local.plan.stat().st_size,
            sha256=local.plan_sha256,
        )
        plugin = AuthenticatedArtifact(
            path=plugin_input,
            relative_path=LOCAL_AR_PLUGIN,
            size_bytes=plugin_input.stat().st_size,
            sha256=local.plugin_sha256,
        )
        plugin_build_receipt_path = staging / PLUGIN_BUILD_RECEIPT
        shutil.copyfile(
            plugin_build_receipt_input,
            plugin_build_receipt_path,
        )
        if (
            plugin_build_receipt_path.stat().st_size
            != plugin_build_evidence.receipt_size_bytes
            or sha256_file(plugin_build_receipt_path)
            != plugin_build_evidence.receipt_sha256
        ):
            raise RuntimeError(
                "plugin build receipt changed while final acceptance was staged"
            )
        plugin_build_receipt_artifact = AuthenticatedArtifact(
            path=plugin_build_receipt_path,
            relative_path=PLUGIN_BUILD_RECEIPT,
            size_bytes=plugin_build_evidence.receipt_size_bytes,
            sha256=plugin_build_evidence.receipt_sha256,
        )
        (staging / f"{PLUGIN_BUILD_RECEIPT}.sha256").write_text(
            f"{plugin_build_receipt_artifact.sha256}  {PLUGIN_BUILD_RECEIPT}\n",
            encoding="ascii",
        )
        text_promotion = write_receipt(
            staging,
            TEXT_PROMOTION_RECEIPT,
            promotion_receipt(
                artifact_role="text_encoder_plan",
                candidate_receipt_sha256=text.receipt_sha256,
                sequence_receipt_sha256=sequence.receipt_sha256,
                lock_sha256=lock_sha256,
                canonical_fixture_manifest_sha256=(
                    canonical_fixture_manifest_sha256
                ),
                engines=[("text_encoder", text_plan)],
                created_at_utc=created_at_utc,
                plugin=plugin,
            ),
        )
        main_promotion = write_receipt(
            staging,
            MAIN_PROMOTION_RECEIPT,
            promotion_receipt(
                artifact_role="main_decoder",
                candidate_receipt_sha256=main_export.receipt_sha256,
                sequence_receipt_sha256=sequence.receipt_sha256,
                lock_sha256=lock_sha256,
                canonical_fixture_manifest_sha256=(
                    canonical_fixture_manifest_sha256
                ),
                engines=[
                    ("main_decoder_prefill", main_prefill),
                    ("main_decoder_step", main_step),
                ],
                created_at_utc=created_at_utc,
                plugin=plugin,
            ),
        )
        local_promotion = write_receipt(
            staging,
            LOCAL_PROMOTION_RECEIPT,
            promotion_receipt(
                artifact_role="local_ar_fixed_16",
                candidate_receipt_sha256=local.receipt_sha256,
                sequence_receipt_sha256=sequence.receipt_sha256,
                lock_sha256=lock_sha256,
                canonical_fixture_manifest_sha256=(
                    canonical_fixture_manifest_sha256
                ),
                engines=[("local_ar_16", local_plan)],
                created_at_utc=created_at_utc,
                plugin=plugin,
            ),
        )
        nanocodec_initial = nanocodec.plans["initial_4"]
        nanocodec_steady = nanocodec.plans["steady_8"]
        nanocodec_tail = nanocodec.plans["tail_1_8"]
        engine_artifacts = (
            ("text_encoder", text_plan),
            ("main_decoder_prefill", main_prefill),
            ("main_decoder_step", main_step),
            ("local_ar_16", local_plan),
            ("nanocodec_initial_4", nanocodec_initial),
            ("nanocodec_steady_8", nanocodec_steady),
            ("nanocodec_tail_1_8", nanocodec_tail),
        )
        if tuple(role for role, _ in engine_artifacts) != ENGINE_ROLES:
            raise AssertionError("consolidated engine role order changed")
        consolidated: dict[str, JsonValue] = {
            "schema_version": 1,
            "artifact_role": "runtime_bundle_export",
            "status": "accepted",
            "created_at_utc": created_at_utc,
            "source": {
                "model_id": model_id,
                "model_version": model_version,
                "model_revision": model_revision,
                "model_sha256": model_sha256,
                "oracle_lock_sha256": lock_sha256,
                "canonical_fixture_manifest_sha256": (
                    canonical_fixture_manifest_sha256
                ),
                "source_model_acceptance_receipt_sha256": (
                    source_model_receipt.sha256
                ),
                "tokenizer_identity_sha256": tokenizer_identity_sha256,
                "tokenizer_identity_receipt_sha256": tokenizer_receipt.sha256,
                "locked_magpie_restore_sha256": (
                    sequence.locked_magpie_restore_sha256
                ),
                "codec_restore": {
                    "embedded_codec_model_id": (
                        sequence.codec_restore.embedded_codec_model_id
                    ),
                    "codec_model_sha256": (
                        sequence.codec_restore.codec_model_sha256
                    ),
                    "codec_model_size_bytes": (
                        sequence.codec_restore.codec_model_size_bytes
                    ),
                    "codec_resolution": "authenticated_local_file",
                    "use_scl_loss": False,
                    "network_resolution": False,
                },
            },
            "runtime_dependencies": {
                "cuda": sequence.cuda_identity.to_json(),
                "cublas": sequence.cublas_identity.to_json(),
                "mode8_class_table_sha256": (
                    sequence.mode8_class_table_sha256
                ),
            },
            "component_receipts": [
                {
                    "role": "text_encoder",
                    "artifact_role": "text_encoder_plan",
                    "receipt_sha256": text_promotion.sha256,
                },
                {
                    "role": "main_decoder",
                    "artifact_role": "main_decoder",
                    "receipt_sha256": main_promotion.sha256,
                },
                {
                    "role": "local_ar",
                    "artifact_role": "local_ar_fixed_16",
                    "receipt_sha256": local_promotion.sha256,
                },
                {
                    "role": "nanocodec",
                    "artifact_role": "stateful_nanocodec",
                    "receipt_sha256": nanocodec.receipt_sha256,
                },
            ],
            "sequence_receipt_sha256": sequence.receipt_sha256,
            "complete_generation_receipt_sha256": (
                golden_receipt_artifact.sha256
            ),
            "eos_frame_is_audio": False,
            "zero_frame_finalization": ZERO_FRAME_FINALIZATION,
            "plugin": {
                "role": "runtime_plugin",
                "sha256": plugin.sha256,
                "size_bytes": plugin.size_bytes,
                "build_receipt_sha256": (
                    plugin_build_receipt_artifact.sha256
                ),
                "build_receipt_size_bytes": (
                    plugin_build_receipt_artifact.size_bytes
                ),
            },
            "engines": [
                {
                    "role": role,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                }
                for role, artifact in engine_artifacts
            ],
            "golden_fixture": golden_fixture_artifact.file_reference(),
            "golden_receipt": golden_receipt_artifact.file_reference(),
        }
        consolidated_artifact = write_receipt(
            staging,
            CONSOLIDATED_RECEIPT,
            consolidated,
        )

        lock_recheck, lock_recheck_payload = load_json(lock_path)
        if (
            lock_recheck != lock
            or sha256_bytes(lock_recheck_payload) != lock_sha256
        ):
            raise RuntimeError(
                "oracle lock changed during final acceptance"
            )
        fixture_recheck = load_fixture(args.fixture, lock_path)
        if (
            fixture_recheck.manifest_sha256
            != canonical_fixture_manifest_sha256
        ):
            raise RuntimeError(
                "canonical fixture changed during final acceptance"
            )
        text_recheck = authenticate_text_encoder_export(
            args.text_encoder_export
        )
        main_recheck = authenticate_plan_export(args.main_export)
        local_recheck = authenticate_local_ar_export(args.local_ar_export)
        if (
            text_recheck != text
            or main_recheck != main_export
            or local_recheck != local
        ):
            raise RuntimeError(
                "generation plan inputs changed during final acceptance"
            )
        sequence_recheck = authenticate_sequence_receipt(
            args.sequence_validation,
            lock_sha256=lock_sha256,
            canonical_fixture_manifest_sha256=(
                canonical_fixture_manifest_sha256
            ),
            text=text_recheck,
            main=main_recheck,
            local=local_recheck,
        )
        if sequence_recheck != sequence:
            raise RuntimeError(
                "closed-loop sequence evidence changed during final acceptance"
            )
        nanocodec_recheck = authenticate_nanocodec_export(
            args.nanocodec_export,
            lock_sha256=lock_sha256,
            canonical_fixture_manifest_sha256=(
                canonical_fixture_manifest_sha256
            ),
        )
        if nanocodec_recheck != nanocodec:
            raise RuntimeError(
                "NanoCodec inputs changed during final acceptance"
            )
        if require_locked_source_receipt(
            args.source_model_acceptance_receipt,
            lock,
        ) != source_model_receipt:
            raise RuntimeError(
                "source-model receipt changed during final acceptance"
            )
        tokenizer_recheck = tokenizer_identity(
            args.tokenizer_identity_receipt
        )
        if tokenizer_recheck != (
            tokenizer_receipt,
            tokenizer_identity_sha256,
        ):
            raise RuntimeError(
                "tokenizer receipt changed during final acceptance"
            )
        plugin_build_recheck = validate_plugin_build_receipt(
            plugin_build_receipt_input,
            FileArtifact(
                path=plugin_input.name,
                sha256=local_recheck.plugin_sha256,
                size_bytes=plugin_input.stat().st_size,
            ),
        )
        validate_plugin_build_source_tree(
            plugin_build_recheck,
            PROJECT_ROOT,
        )
        if plugin_build_recheck != plugin_build_evidence:
            raise RuntimeError(
                "plugin build evidence changed during final acceptance"
            )

        publish_directory_no_replace(staging, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "consolidated_receipt_sha256": consolidated_artifact.sha256,
                    "sequence_receipt_sha256": sequence.receipt_sha256,
                    "golden_receipt_sha256": golden_receipt_artifact.sha256,
                    "golden_fixture_sha256": golden_fixture_artifact.sha256,
                    "pcm_f32le_sha256": hashes.pcm_f32le_sha256,
                    "pcm_sample_count": hashes.pcm_sample_count,
                    "plugin_build_receipt_sha256": (
                        plugin_build_receipt_artifact.sha256
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

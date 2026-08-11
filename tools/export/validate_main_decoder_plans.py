#!/usr/bin/env python3
"""Measure one locked Main Decoder plan pair over multiple oracle fixtures.

This tool never builds or modifies a TensorRT plan. It authenticates an
existing measured export, executes the exact same prefill and one-step plans
for at least two distinct fixtures, and publishes a checksummed receipt. The
result remains ``measured-not-accepted`` until sequence and semantic evidence
justify predeclared tolerances.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch

from alignment_controller import SofiaAlignmentController
from build_text_encoder_plan import register_plugin
from cublas_runtime_identity import (
    CublasRuntimeIdentity,
    parse_cublas_runtime_identity,
)
from cuda_runtime_identity import (
    CudaRuntimeIdentity,
    parse_cuda_runtime_identity,
)
from export_main_decoder import (
    DECODER_LAYERS,
    PREFILL_PLAN,
    RECEIPT,
    RECEIPT_CHECKSUM,
    STEP_PLAN,
    DecoderFixture,
    canonical_json_bytes,
    execute_plan,
    fixture_inputs,
    grouped_parity,
    import_tensorrt,
    inspect_plan,
    load_fixture,
    publish_directory_no_replace,
    step_input_names,
    tensor_from_fixture,
)


type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list[JsonValue]
    | dict[str, JsonValue]
)


HARNESS_RECEIPT = "validation-receipt.json"
HARNESS_RECEIPT_CHECKSUM = "validation-receipt.json.sha256"
MINIMUM_DISTINCT_FIXTURES = 2


@dataclass(frozen=True)
class AuthenticatedCodecRestore:
    embedded_codec_model_id: str
    codec_model_sha256: str
    codec_model_size_bytes: int


def authenticate_codec_restore(
    source: dict[str, JsonValue],
    label: str,
) -> AuthenticatedCodecRestore:
    restore = require_mapping(source.get("codec_restore"), f"{label}.codec_restore")
    if (
        require_string(
            restore.get("codec_resolution"),
            f"{label}.codec_restore.codec_resolution",
        )
        != "authenticated_local_file"
        or require_boolean(
            restore.get("use_scl_loss"),
            f"{label}.codec_restore.use_scl_loss",
        )
        or require_boolean(
            restore.get("network_resolution"),
            f"{label}.codec_restore.network_resolution",
        )
    ):
        raise RuntimeError(
            f"{label}.codec_restore is not the authenticated offline contract"
        )
    if set(restore) != {
        "embedded_codec_model_id",
        "codec_model_sha256",
        "codec_model_size_bytes",
        "codec_resolution",
        "use_scl_loss",
        "network_resolution",
    }:
        raise RuntimeError(f"{label}.codec_restore has an unexpected field set")
    return AuthenticatedCodecRestore(
        embedded_codec_model_id=require_string(
            restore.get("embedded_codec_model_id"),
            f"{label}.codec_restore.embedded_codec_model_id",
        ),
        codec_model_sha256=require_sha256(
            restore.get("codec_model_sha256"),
            f"{label}.codec_restore.codec_model_sha256",
        ),
        codec_model_size_bytes=require_nonnegative_integer(
            restore.get("codec_model_size_bytes"),
            f"{label}.codec_restore.codec_model_size_bytes",
        ),
    )


@dataclass(frozen=True)
class AuthenticatedPlanExport:
    root: Path
    receipt_sha256: str
    oracle_lock_sha256: str
    locked_magpie_restore_sha256: str
    codec_restore: AuthenticatedCodecRestore
    prefill_plan: Path
    prefill_plan_sha256: str
    step_plan: Path
    step_plan_sha256: str
    source_fixture_manifest_sha256: str
    required_plugin_sha256: str
    mode8_validation_receipt_sha256: str
    mode8_class_table_sha256: str
    cuda_identity: CudaRuntimeIdentity
    cublas_identity: CublasRuntimeIdentity
    tensorrt_version: str
    torch_cuda_build: str
    gpu_name: str
    gpu_compute_capability: tuple[int, int]


@dataclass(frozen=True)
class FixtureMetadata:
    fixture_id: str
    text: str
    language: str
    local_ar_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--plan-export", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each independently captured oracle fixture.",
    )
    parser.add_argument("--tensorrt-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_pairs(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> tuple[dict[str, JsonValue], bytes]:
    payload = path.read_bytes()
    parsed: JsonValue = json.loads(
        payload,
        object_pairs_hook=reject_duplicate_pairs,
    )
    if not isinstance(parsed, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return parsed, payload


def require_mapping(
    value: JsonValue | None,
    label: str,
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def require_list(value: JsonValue | None, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be an array")
    return value


def require_string(value: JsonValue | None, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string")
    return value


def require_integer(value: JsonValue | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_boolean(value: JsonValue | None, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} must be a boolean")
    return value


def require_nonnegative_integer(value: JsonValue | None, label: str) -> int:
    result = require_integer(value, label)
    if result < 0:
        raise RuntimeError(f"{label} must be non-negative")
    return result


def require_sha256(value: JsonValue | None, label: str) -> str:
    digest = require_string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def require_safe_artifact_path(value: JsonValue | None, label: str) -> str:
    relative = require_string(value, label)
    path = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or "\0" in relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RuntimeError(f"{label} is not a safe normalized relative path")
    return relative


def authenticate_plan_export(path: Path) -> AuthenticatedPlanExport:
    if path.is_symlink():
        raise RuntimeError(f"plan export must not be a symbolic link: {path}")
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"plan export is not a directory: {root}")
    receipt_path = root / RECEIPT
    checksum_path = root / RECEIPT_CHECKSUM
    if receipt_path.is_symlink() or checksum_path.is_symlink():
        raise RuntimeError("plan export receipt files must not be symbolic links")
    receipt, receipt_payload = load_json(receipt_path)
    receipt_sha256 = sha256_bytes(receipt_payload)
    expected_checksum = f"{receipt_sha256}  {RECEIPT}\n".encode("ascii")
    if checksum_path.read_bytes() != expected_checksum:
        raise RuntimeError("plan export receipt checksum does not match")
    if require_integer(receipt.get("schema_version"), "receipt.schema_version") != 1:
        raise RuntimeError("unsupported plan export receipt schema")
    if (
        require_string(receipt.get("artifact_role"), "receipt.artifact_role")
        != "main_decoder_prefill_and_step"
    ):
        raise RuntimeError("plan export has the wrong artifact role")
    if (
        require_string(receipt.get("status"), "receipt.status")
        != "measured-not-accepted"
    ):
        raise RuntimeError("only an explicitly measured-not-accepted export is valid")

    plan_inspection = require_mapping(
        receipt.get("plan_inspection"),
        "receipt.plan_inspection",
    )
    step_inspection = require_mapping(
        plan_inspection.get("step"),
        "receipt.plan_inspection.step",
    )
    step_tensors = require_list(
        step_inspection.get("tensors"),
        "receipt.plan_inspection.step.tensors",
    )
    position_records = [
        require_mapping(
            value,
            f"receipt.plan_inspection.step.tensors[{index}]",
        )
        for index, value in enumerate(step_tensors)
        if isinstance(value, dict) and value.get("name") == "position"
    ]
    if len(position_records) != 1:
        raise RuntimeError(
            "plan export must inspect exactly one step position tensor"
        )
    position = position_records[0]
    position_shape = require_list(
        position.get("shape"),
        "receipt.plan_inspection.step.position.shape",
    )
    if (
        require_string(
            position.get("dtype"),
            "receipt.plan_inspection.step.position.dtype",
        )
        != "int64"
        or position_shape
        or require_string(
            position.get("mode"),
            "receipt.plan_inspection.step.position.mode",
        )
        != "input"
        or require_string(
            position.get("location"),
            "receipt.plan_inspection.step.position.location",
        )
        != "device"
        or require_boolean(
            position.get("shape_inference_io"),
            "receipt.plan_inspection.step.position.shape_inference_io",
        )
        or "profile_values" in position
    ):
        raise RuntimeError(
            "plan export step position must be a scalar int64 DEVICE "
            "execution input without profile values"
        )

    status_records = {
        name: [
            require_mapping(
                value,
                f"receipt.plan_inspection.step.tensors[{index}]",
            )
            for index, value in enumerate(step_tensors)
            if isinstance(value, dict) and value.get("name") == name
        ]
        for name in ("execution_status_in", "execution_status_out")
    }
    for name, expected_mode in (
        ("execution_status_in", "input"),
        ("execution_status_out", "output"),
    ):
        records = status_records[name]
        if len(records) != 1:
            raise RuntimeError(
                f"plan export must inspect exactly one step {name} tensor"
            )
        record = records[0]
        if (
            require_string(
                record.get("dtype"),
                f"receipt.plan_inspection.step.{name}.dtype",
            )
            != "int32"
            or require_list(
                record.get("shape"),
                f"receipt.plan_inspection.step.{name}.shape",
            )
            or require_string(
                record.get("mode"),
                f"receipt.plan_inspection.step.{name}.mode",
            )
            != expected_mode
            or require_string(
                record.get("location"),
                f"receipt.plan_inspection.step.{name}.location",
            )
            != "device"
            or require_boolean(
                record.get("shape_inference_io"),
                f"receipt.plan_inspection.step.{name}.shape_inference_io",
            )
            or "profile_values" in record
        ):
            raise RuntimeError(
                f"plan export step {name} must be a scalar int32 DEVICE "
                f"{expected_mode} without profile values"
            )

    artifact_values = require_list(receipt.get("artifacts"), "receipt.artifacts")
    authenticated: dict[str, tuple[Path, str]] = {}
    expected_files = {RECEIPT, RECEIPT_CHECKSUM}
    for index, raw_record in enumerate(artifact_values):
        label = f"receipt.artifacts[{index}]"
        record = require_mapping(raw_record, label)
        relative = require_safe_artifact_path(record.get("path"), f"{label}.path")
        if relative in authenticated:
            raise RuntimeError(f"duplicate plan export artifact: {relative}")
        size_bytes = require_nonnegative_integer(
            record.get("size_bytes"),
            f"{label}.size_bytes",
        )
        expected_digest = require_sha256(record.get("sha256"), f"{label}.sha256")
        unresolved_artifact = root / PurePosixPath(relative)
        if unresolved_artifact.is_symlink():
            raise RuntimeError(f"plan export artifact is a symbolic link: {relative}")
        artifact = unresolved_artifact.resolve(strict=True)
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise RuntimeError(f"invalid plan export artifact: {relative}")
        if artifact.stat().st_size != size_bytes:
            raise RuntimeError(f"plan export artifact size mismatch: {relative}")
        actual_digest = sha256_file(artifact)
        if actual_digest != expected_digest:
            raise RuntimeError(f"plan export artifact digest mismatch: {relative}")
        authenticated[relative] = (artifact, actual_digest)
        expected_files.add(relative)

    expected_directories = {""}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_directories = {""}
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"plan export contains a symbolic link: {relative}")
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(f"plan export contains a non-file entry: {relative}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            "plan export entry set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}, "
            f"extra_directories={sorted(actual_directories - expected_directories)}"
        )
    try:
        prefill_plan, prefill_digest = authenticated[PREFILL_PLAN]
        step_plan, step_digest = authenticated[STEP_PLAN]
    except KeyError as error:
        raise RuntimeError(f"plan export is missing required plan: {error.args[0]}") from error
    source = require_mapping(receipt.get("source"), "receipt.source")
    runtime = require_mapping(receipt.get("runtime"), "receipt.runtime")
    export = require_mapping(receipt.get("export"), "receipt.export")
    if (
        require_string(
            export.get("execution_status_contract"),
            "receipt.export.execution_status_contract",
        )
        != "int32-device-scalar-sticky-12-layer-recurrence"
    ):
        raise RuntimeError(
            "plan export has no canonical Main execution-status recurrence"
        )
    mode8_class_table_sha256 = require_sha256(
        runtime.get("mode8_class_table_sha256"),
        "receipt.runtime.mode8_class_table_sha256",
    )
    if (
        require_sha256(
            export.get("mode8_class_table_sha256"),
            "receipt.export.mode8_class_table_sha256",
        )
        != mode8_class_table_sha256
    ):
        raise RuntimeError(
            "plan export mode-8 class-table digests do not match"
        )
    cuda_identity = parse_cuda_runtime_identity(
        runtime.get("cuda"),
        "receipt.runtime.cuda",
    )
    cublas_identity = parse_cublas_runtime_identity(
        runtime.get("cublas"),
        "receipt.runtime.cublas",
    )
    capability_values = require_list(
        runtime.get("gpu_compute_capability"),
        "receipt.runtime.gpu_compute_capability",
    )
    if len(capability_values) != 2:
        raise RuntimeError("plan export GPU capability must have two components")
    capability = tuple(
        require_nonnegative_integer(
            value,
            f"receipt.runtime.gpu_compute_capability[{index}]",
        )
        for index, value in enumerate(capability_values)
    )
    return AuthenticatedPlanExport(
        root=root,
        receipt_sha256=receipt_sha256,
        oracle_lock_sha256=require_sha256(
            source.get("oracle_lock_sha256"),
            "receipt.source.oracle_lock_sha256",
        ),
        locked_magpie_restore_sha256=require_sha256(
            source.get("locked_magpie_restore_sha256"),
            "receipt.source.locked_magpie_restore_sha256",
        ),
        codec_restore=authenticate_codec_restore(source, "receipt.source"),
        prefill_plan=prefill_plan,
        prefill_plan_sha256=prefill_digest,
        step_plan=step_plan,
        step_plan_sha256=step_digest,
        source_fixture_manifest_sha256=require_sha256(
            source.get("boundary_fixture_manifest_sha256"),
            "receipt.source.boundary_fixture_manifest_sha256",
        ),
        required_plugin_sha256=require_sha256(
            source.get("plugin_sha256"),
            "receipt.source.plugin_sha256",
        ),
        mode8_validation_receipt_sha256=require_sha256(
            source.get("mode8_validation_receipt_sha256"),
            "receipt.source.mode8_validation_receipt_sha256",
        ),
        mode8_class_table_sha256=mode8_class_table_sha256,
        cuda_identity=cuda_identity,
        cublas_identity=cublas_identity,
        tensorrt_version=require_string(
            runtime.get("tensorrt"),
            "receipt.runtime.tensorrt",
        ),
        torch_cuda_build=require_string(
            runtime.get("torch_cuda_build"),
            "receipt.runtime.torch_cuda_build",
        ),
        gpu_name=require_string(
            runtime.get("gpu_name"),
            "receipt.runtime.gpu_name",
        ),
        gpu_compute_capability=(capability[0], capability[1]),
    )


def load_fixture_metadata(root: Path) -> FixtureMetadata:
    manifest, _ = load_json(root / "manifest.json")
    local_ar_seed = require_integer(
        manifest.get("local_ar_seed"),
        "fixture.local_ar_seed",
    )
    if local_ar_seed < 0 or local_ar_seed >= 2**32:
        raise RuntimeError(
            "fixture.local_ar_seed must be in [0, 2^32), "
            f"got {local_ar_seed}"
        )
    language = require_string(manifest.get("language"), "fixture.language")
    if language != "ja":
        raise RuntimeError(f"Main Decoder acceptance fixture must be Japanese: {language}")
    return FixtureMetadata(
        fixture_id=require_string(manifest.get("fixture_id"), "fixture.fixture_id"),
        text=require_string(manifest.get("text"), "fixture.text"),
        language=language,
        local_ar_seed=local_ar_seed,
    )


def write_validation_tensor(
    staging: Path,
    relative: Path,
    value: torch.Tensor,
) -> dict[str, JsonValue]:
    if value.dtype != torch.bfloat16:
        raise RuntimeError(f"validation tensor must be BF16: {relative}")
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe validation tensor path: {relative}")
    payload = (
        value.detach()
        .cpu()
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
        .tobytes()
    )
    output = staging / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "path": relative.as_posix(),
        "dtype": "bf16",
        "shape": list(value.shape),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def verify_alignment_decision(
    *,
    fixture: DecoderFixture,
    oracle_alignment: torch.Tensor,
    plan_alignment: torch.Tensor,
) -> dict[str, JsonValue]:
    expected_prior = tensor_from_fixture(fixture, "step_000.next_prior")
    expected_attended = tensor_from_fixture(fixture, "step_000.attended")
    expected_counters = tensor_from_fixture(
        fixture,
        "step_000.attention_counters",
    )
    expected_shapes = {
        "step_000.next_prior": (2, 1, fixture.text_tokens),
        "step_000.attended": (1,),
        "step_000.attention_counters": (1, fixture.text_tokens),
    }
    for name, expected_shape in expected_shapes.items():
        record = fixture.tensors.get(name)
        if record is None:
            raise RuntimeError(f"alignment fixture tensor is missing: {name}")
        if record.shape != expected_shape:
            raise RuntimeError(
                f"alignment fixture tensor {name} shape mismatch: "
                f"expected={expected_shape}, actual={record.shape}"
            )

    oracle_controller = SofiaAlignmentController(
        text_length=fixture.text_tokens,
        device=oracle_alignment.device,
        dtype=oracle_alignment.dtype,
    )
    oracle_update = oracle_controller.update(oracle_alignment)
    oracle_exact = (
        torch.equal(oracle_update.prior, expected_prior)
        and torch.equal(oracle_update.attended, expected_attended)
        and torch.equal(oracle_controller.counters, expected_counters)
    )
    if not oracle_exact:
        raise RuntimeError(
            "alignment controller does not reproduce the locked oracle "
            f"decision for fixture {fixture.manifest_sha256}"
        )

    plan_controller = SofiaAlignmentController(
        text_length=fixture.text_tokens,
        device=plan_alignment.device,
        dtype=plan_alignment.dtype,
    )
    plan_update = plan_controller.update(plan_alignment)
    plan_decision_exact = (
        torch.equal(plan_update.prior, expected_prior)
        and torch.equal(plan_update.attended, expected_attended)
        and torch.equal(plan_controller.counters, expected_counters)
    )
    if not plan_decision_exact:
        raise RuntimeError(
            "TensorRT prefill alignment changed the locked discrete alignment "
            f"decision for fixture {fixture.manifest_sha256}"
        )
    return {
        "oracle_fixture_bit_exact": True,
        "plan_alignment_discrete_decision_matches_oracle": True,
        "attended_position": int(expected_attended.item()),
        "attention_counter_total": int(expected_counters.sum().item()),
    }


def measure_fixture(
    *,
    tensorrt,
    plan_export: AuthenticatedPlanExport,
    fixture_path: Path,
    lock_path: Path,
    staging: Path,
    case_index: int,
) -> dict[str, JsonValue]:
    fixture = load_fixture(fixture_path, lock_path)
    metadata = load_fixture_metadata(fixture.root)
    (
        prefill_inputs,
        step_inputs,
        prefill_expected,
        step_expected,
    ) = fixture_inputs(fixture)
    prefill_actual = execute_plan(
        tensorrt,
        plan_export.prefill_plan,
        {
            "condition": prefill_inputs[0],
            "condition_mask": prefill_inputs[1],
        },
    )
    alignment_decision = verify_alignment_decision(
        fixture=fixture,
        oracle_alignment=prefill_expected["alignment"],
        plan_alignment=prefill_actual["alignment"],
    )
    step_input_map = dict(zip(step_input_names(), step_inputs, strict=True))
    for layer_index in range(DECODER_LAYERS):
        for stem in (
            "self_key",
            "self_value",
            "self_mask",
            "cross_key",
            "cross_value",
        ):
            step_input_map[f"step_{stem}_in_{layer_index}"] = prefill_actual[
                f"prefill_{stem}_{layer_index}"
            ]
    step_actual = execute_plan(
        tensorrt,
        plan_export.step_plan,
        step_input_map,
    )
    case_name = f"case-{case_index:03d}-{fixture.manifest_sha256[:12]}"
    artifacts = [
        write_validation_tensor(
            staging,
            Path("validation") / case_name / "prefill.hidden.bf16.bin",
            prefill_actual["last_hidden"],
        ),
        write_validation_tensor(
            staging,
            Path("validation") / case_name / "step-001.hidden.bf16.bin",
            step_actual["decoder_hidden"],
        ),
    ]
    result: dict[str, JsonValue] = {
        "case": case_name,
        "fixture_id": metadata.fixture_id,
        "fixture_manifest_sha256": fixture.manifest_sha256,
        "text": metadata.text,
        "language": metadata.language,
        "local_ar_seed": metadata.local_ar_seed,
        "text_tokens": fixture.text_tokens,
        "step_cache_source": "TensorRT prefill outputs",
        "alignment_decision": alignment_decision,
        "prefill": grouped_parity(
            "prefill",
            prefill_actual,
            prefill_expected,
        ),
        "step": grouped_parity("step", step_actual, step_expected),
        "artifacts": artifacts,
    }
    del (
        prefill_inputs,
        step_inputs,
        prefill_expected,
        step_expected,
        prefill_actual,
        step_input_map,
        step_actual,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def metric_envelope(cases: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    envelope: dict[str, JsonValue] = {}
    for role in ("prefill", "step"):
        role_result: dict[str, JsonValue] = {}
        group_names: set[str] = set()
        for case in cases:
            role_metrics = require_mapping(case.get(role), f"case.{role}")
            group_names.update(role_metrics)
        for group in sorted(group_names):
            metrics = [
                require_mapping(
                    require_mapping(case.get(role), f"case.{role}").get(group),
                    f"case.{role}.{group}",
                )
                for case in cases
            ]
            if "max_abs" in metrics[0]:
                role_result[group] = {
                    "maximum_max_abs": max(
                        float(metric["max_abs"]) for metric in metrics
                    ),
                    "maximum_mean_abs": max(
                        float(metric["mean_abs"]) for metric in metrics
                    ),
                    "maximum_p99_abs": max(
                        float(metric["p99_abs"]) for metric in metrics
                    ),
                    "minimum_cosine_similarity": min(
                        float(metric["cosine_similarity"]) for metric in metrics
                    ),
                    "maximum_bit_mismatch_ratio": max(
                        float(metric["bit_mismatch_ratio"]) for metric in metrics
                    ),
                }
            else:
                role_result[group] = {
                    "maximum_mismatch_ratio": max(
                        float(metric["mismatch_ratio"]) for metric in metrics
                    )
                }
        envelope[role] = role_result
    return envelope


def main() -> int:
    args = parse_args()
    fixture_paths = [fixture.resolve(strict=True) for fixture in args.fixture]
    if len(fixture_paths) < MINIMUM_DISTINCT_FIXTURES:
        raise RuntimeError(
            f"at least {MINIMUM_DISTINCT_FIXTURES} fixtures are required"
        )
    fixture_inputs_seen: set[tuple[str, int]] = set()
    for fixture_path in fixture_paths:
        metadata = load_fixture_metadata(fixture_path)
        fixture_input = (metadata.text, metadata.local_ar_seed)
        if fixture_input in fixture_inputs_seen:
            raise RuntimeError(
                "fixture text/seed input is duplicated and is not independent: "
                f"{fixture_input!r}"
            )
        fixture_inputs_seen.add(fixture_input)
    plan_export = authenticate_plan_export(args.plan_export)
    plugin_path = args.plugin.resolve(strict=True)
    plugin_sha256 = sha256_file(plugin_path)
    if plugin_sha256 != plan_export.required_plugin_sha256:
        raise RuntimeError(
            "plan export plugin digest mismatch: "
            f"expected={plan_export.required_plugin_sha256}, "
            f"actual={plugin_sha256}"
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        plugin_library, _ = register_plugin(plugin_path)
        tensorrt = import_tensorrt(args.tensorrt_python_path)
        current_fingerprint = (
            tensorrt.__version__,
            str(torch.version.cuda),
            torch.cuda.get_device_name(0),
            tuple(torch.cuda.get_device_capability(0)),
        )
        expected_fingerprint = (
            plan_export.tensorrt_version,
            plan_export.torch_cuda_build,
            plan_export.gpu_name,
            plan_export.gpu_compute_capability,
        )
        if current_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "plan export runtime fingerprint mismatch: "
                f"expected={expected_fingerprint}, actual={current_fingerprint}"
            )
        prefill_inspection = inspect_plan(
            tensorrt,
            "prefill",
            plan_export.prefill_plan,
        )
        step_inspection = inspect_plan(
            tensorrt,
            "step",
            plan_export.step_plan,
        )
        cases: list[dict[str, JsonValue]] = []
        manifest_digests: set[str] = set()
        for index, fixture_path in enumerate(fixture_paths):
            case = measure_fixture(
                tensorrt=tensorrt,
                plan_export=plan_export,
                fixture_path=fixture_path,
                lock_path=args.lock,
                staging=staging,
                case_index=index,
            )
            manifest_digest = require_sha256(
                case.get("fixture_manifest_sha256"),
                "case.fixture_manifest_sha256",
            )
            if manifest_digest in manifest_digests:
                raise RuntimeError(
                    f"duplicate fixture manifest is not independent: {manifest_digest}"
                )
            manifest_digests.add(manifest_digest)
            cases.append(case)

        artifact_records: list[JsonValue] = []
        for case in cases:
            artifact_records.extend(
                require_list(case.get("artifacts"), "case.artifacts")
            )
        receipt: dict[str, JsonValue] = {
            "schema_version": 1,
            "artifact_role": "main_decoder_multi_fixture_validation",
            "status": "measured-not-accepted",
            "reason": (
                "no tolerance is inferred from these fixtures; closed-loop "
                "Local AR code/EOS and sequence evidence remain required"
            ),
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "source": {
                "validator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
                "oracle_lock_sha256": sha256_file(args.lock.resolve(strict=True)),
                "plan_export_receipt_sha256": plan_export.receipt_sha256,
                "plan_export_source_fixture_manifest_sha256": (
                    plan_export.source_fixture_manifest_sha256
                ),
                "prefill_plan_sha256": plan_export.prefill_plan_sha256,
                "step_plan_sha256": plan_export.step_plan_sha256,
                "plugin_sha256": plugin_sha256,
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "tensorrt": tensorrt.__version__,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "plan_inspection": {
                "prefill": prefill_inspection,
                "step": step_inspection,
            },
            "fixture_count": len(cases),
            "fixtures": cases,
            "metric_envelope": metric_envelope(cases),
            "artifacts": artifact_records,
        }
        receipt_path = staging / HARNESS_RECEIPT
        receipt_payload = canonical_json_bytes(receipt)
        receipt_path.write_bytes(receipt_payload)
        (staging / HARNESS_RECEIPT_CHECKSUM).write_text(
            f"{sha256_bytes(receipt_payload)}  {HARNESS_RECEIPT}\n",
            encoding="ascii",
        )
        if plugin_library is None:
            raise RuntimeError("plugin library lifetime was not retained")
        publish_directory_no_replace(staging, output)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    sys.exit(main())

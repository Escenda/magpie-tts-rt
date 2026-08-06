#!/usr/bin/env python3
"""Finalize and verify a portable source-model acceptance receipt.

The CUDA acceptance probe intentionally records host-local paths so an operator
can audit the machine that executed it.  Those paths are not part of a
redistributable runtime bundle.  This tool validates the complete raw probe
against ``oracle-lock.json`` and emits a minimal v2 receipt containing only
portable, authenticated facts.  It never edits either input and never replaces
an existing output.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

SCHEMA_VERSION = 2
ARTIFACT_ROLE = "source_model_acceptance"
ACCEPTED_STATUS = "accepted"
RAW_SCHEMA = "magpie-final-native-bf16-streaming-acceptance-v1"
RAW_ACCEPTED_STATUS = "passed_runtime_artifacts_ready_for_parakeet"
MAX_JSON_BYTES = 16 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
EXPECTED_CASE_NAMES = (
    "short_conversation",
    "punctuation",
    "latin_and_numbers",
    "long_multi_chunk",
)
REQUIRED_PASSED_GATES = (
    "all_cases_exercised",
    "callback_contracts_passed",
    "codec_fp32",
    "default_schedule_exact",
    "eos_alignment_passed",
    "explicit_outer_warmup_only",
    "finite_audio_artifacts_passed",
    "general_followup_case_passed",
    "generation_rtf_p95_below_one",
    "local_ar_graph_passed",
    "native_bf16_no_adapter",
    "native_codec_graphs_passed",
    "packed_first4_cases_passed",
    "runtime_acceptance_passed",
    "same_seed_replay_passed",
    "source_model_probe_unchanged",
    "total_rtf_p95_below_one",
    "zero_positive_playback_lateness",
)
ACCEPTANCE_CONTRACT_FIELDS = (
    "speaker_name",
    "speaker_index",
    "local_ar_seed",
    "sample_rate_hz",
    "samples_per_codec_frame",
    "first_codec_frames",
    "steady_codec_frames",
    "tail_codec_frames_min",
    "tail_codec_frames_max",
)
METRIC_SUMMARY_FIELDS = ("median", "p95", "minimum", "maximum")
AGGREGATE_FIELDS = (
    "case_count",
    "raw_ttfa_ms",
    "gapless_start_ms",
    "max_positive_playback_lateness_ms",
    "generation_rtf",
    "total_rtf",
    "maximum_peak_cuda_allocated_bytes",
    "maximum_peak_cuda_reserved_bytes",
)
GATE_FIELDS = (
    *REQUIRED_PASSED_GATES,
    "parakeet_gate_pending",
    "failures",
)


class SourceAcceptanceError(RuntimeError):
    """Fail-closed source acceptance validation error."""


def reject_duplicate_pairs(
    pairs: list[tuple[str, JsonValue]],
) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise SourceAcceptanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> tuple[JsonObject, bytes]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise SourceAcceptanceError(f"JSON input must be a regular file: {path}")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise SourceAcceptanceError(
            f"JSON input size is outside 1..{MAX_JSON_BYTES}: {resolved}"
        )
    payload = resolved.read_bytes()
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceAcceptanceError(f"invalid JSON: {resolved}: {error}") from error
    if not isinstance(decoded, dict):
        raise SourceAcceptanceError(f"JSON root must be an object: {resolved}")
    return decoded, payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_object(value: JsonValue, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise SourceAcceptanceError(f"{path}: expected an object")
    return value


def require_array(value: JsonValue, path: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise SourceAcceptanceError(f"{path}: expected an array")
    return value


def require_string(value: JsonValue, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceAcceptanceError(f"{path}: expected a non-empty string")
    return value


def require_integer(value: JsonValue, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceAcceptanceError(f"{path}: expected an integer")
    return value


def require_number(value: JsonValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceAcceptanceError(f"{path}: expected a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise SourceAcceptanceError(f"{path}: expected a finite JSON number")
    return result


def require_sha256(value: JsonValue, path: str) -> str:
    result = require_string(value, path)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise SourceAcceptanceError(f"{path}: expected lowercase SHA-256")
    return result


def require_git_sha1(value: JsonValue, path: str) -> str:
    result = require_string(value, path)
    if GIT_SHA1_PATTERN.fullmatch(result) is None:
        raise SourceAcceptanceError(f"{path}: expected lowercase Git SHA-1")
    return result


def require_exact_keys(
    value: JsonObject,
    expected: tuple[str, ...],
    path: str,
) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise SourceAcceptanceError(
            f"{path}: key set mismatch: "
            f"missing={sorted(wanted - actual)}, extra={sorted(actual - wanted)}"
        )


def ordered_source_bundle_sha256(files: JsonObject) -> str:
    digest = hashlib.sha256()
    for relative, raw_sha256 in files.items():
        file_sha256 = require_sha256(
            raw_sha256,
            f"/oracle_source/files/{relative}",
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_from_lock(lock: JsonObject) -> JsonObject:
    source = require_object(lock.get("oracle_source"), "/oracle_source")
    files = require_object(source.get("files"), "/oracle_source/files")
    if len(files) != 11:
        raise SourceAcceptanceError(
            f"/oracle_source/files: expected exactly 11 files, got {len(files)}"
        )
    expected_bundle = require_sha256(
        source.get("optimized_source_bundle_sha256"),
        "/oracle_source/optimized_source_bundle_sha256",
    )
    actual_bundle = ordered_source_bundle_sha256(files)
    if actual_bundle != expected_bundle:
        raise SourceAcceptanceError(
            "/oracle_source/optimized_source_bundle_sha256: "
            f"ordered digest mismatch: expected {expected_bundle}, got {actual_bundle}"
        )
    return {
        "repository": require_string(
            source.get("repository"),
            "/oracle_source/repository",
        ),
        "base_revision": require_git_sha1(
            source.get("base_revision"),
            "/oracle_source/base_revision",
        ),
        "optimized_source_bundle_sha256": expected_bundle,
        "files": dict(files),
    }


def model_from_lock(lock: JsonObject) -> JsonObject:
    model = require_object(lock.get("model"), "/model")
    license_document = require_object(model.get("license"), "/model/license")
    require_exact_keys(
        license_document,
        (
            "name",
            "version",
            "document_file_name",
            "document_sha256",
            "required_notice_file_name",
            "required_notice_sha256",
        ),
        "/model/license",
    )
    return {
        "model_id": require_string(model.get("model_id"), "/model/model_id"),
        "version": require_string(model.get("version"), "/model/version"),
        "revision": require_git_sha1(model.get("revision"), "/model/revision"),
        "file_name": require_string(model.get("file_name"), "/model/file_name"),
        "sha256": require_sha256(model.get("sha256"), "/model/sha256"),
        "size_bytes": require_integer(model.get("size_bytes"), "/model/size_bytes"),
        "license": {
            "name": require_string(
                license_document.get("name"),
                "/model/license/name",
            ),
            "version": require_string(
                license_document.get("version"),
                "/model/license/version",
            ),
            "document_file_name": require_string(
                license_document.get("document_file_name"),
                "/model/license/document_file_name",
            ),
            "document_sha256": require_sha256(
                license_document.get("document_sha256"),
                "/model/license/document_sha256",
            ),
            "required_notice_file_name": require_string(
                license_document.get("required_notice_file_name"),
                "/model/license/required_notice_file_name",
            ),
            "required_notice_sha256": require_sha256(
                license_document.get("required_notice_sha256"),
                "/model/license/required_notice_sha256",
            ),
        },
    }


def acceptance_contract_from_lock(lock: JsonObject) -> JsonObject:
    acceptance = require_object(lock.get("acceptance"), "/acceptance")
    result: JsonObject = {}
    for field in ACCEPTANCE_CONTRACT_FIELDS:
        value = acceptance.get(field)
        if field == "speaker_name":
            result[field] = require_string(value, f"/acceptance/{field}")
        else:
            result[field] = require_integer(value, f"/acceptance/{field}")
    return result


def validate_raw_source_section(
    section: JsonObject,
    lock: JsonObject,
    path: str,
) -> None:
    model = model_from_lock(lock)
    expected_source = source_from_lock(lock)
    if (
        require_sha256(section.get("model_sha256"), f"{path}/model_sha256")
        != require_sha256(model.get("sha256"), "/model/sha256")
    ):
        raise SourceAcceptanceError(f"{path}/model_sha256: differs from oracle lock")
    if (
        require_integer(section.get("model_size_bytes"), f"{path}/model_size_bytes")
        != require_integer(model.get("size_bytes"), "/model/size_bytes")
    ):
        raise SourceAcceptanceError(
            f"{path}/model_size_bytes: differs from oracle lock"
        )
    if (
        require_git_sha1(
            section.get("speech_git_head"),
            f"{path}/speech_git_head",
        )
        != expected_source["base_revision"]
    ):
        raise SourceAcceptanceError(f"{path}/speech_git_head: differs from oracle lock")
    actual_files = require_object(
        section.get("optimized_source_file_sha256"),
        f"{path}/optimized_source_file_sha256",
    )
    expected_files = require_object(
        expected_source["files"],
        "/oracle_source/files",
    )
    if tuple(actual_files) != tuple(expected_files):
        raise SourceAcceptanceError(
            f"{path}/optimized_source_file_sha256: ordered file set differs "
            "from oracle lock"
        )
    for relative, expected_sha256 in expected_files.items():
        actual_sha256 = require_sha256(
            actual_files.get(relative),
            f"{path}/optimized_source_file_sha256/{relative}",
        )
        if actual_sha256 != expected_sha256:
            raise SourceAcceptanceError(
                f"{path}/optimized_source_file_sha256/{relative}: "
                "differs from oracle lock"
            )
    actual_bundle = require_sha256(
        section.get("optimized_source_bundle_sha256"),
        f"{path}/optimized_source_bundle_sha256",
    )
    expected_bundle = require_sha256(
        expected_source["optimized_source_bundle_sha256"],
        "/oracle_source/optimized_source_bundle_sha256",
    )
    if actual_bundle != expected_bundle:
        raise SourceAcceptanceError(
            f"{path}/optimized_source_bundle_sha256: differs from oracle lock"
        )
    if ordered_source_bundle_sha256(actual_files) != actual_bundle:
        raise SourceAcceptanceError(
            f"{path}/optimized_source_bundle_sha256: does not authenticate "
            "the ordered internal file map"
        )


def canonical_utc_timestamp(unix_seconds: float) -> str:
    if unix_seconds <= 0.0:
        raise SourceAcceptanceError(
            "/created_unix_seconds: expected a positive timestamp"
        )
    return (
        datetime.datetime.fromtimestamp(unix_seconds, tz=datetime.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def validate_cases(
    raw: JsonObject,
    contract: JsonObject,
) -> tuple[str, ...]:
    cases = require_array(raw.get("cases"), "/cases")
    names: list[str] = []
    expected_schedule = [
        contract["first_codec_frames"],
        contract["steady_codec_frames"],
        True,
    ]
    for index, raw_case in enumerate(cases):
        path = f"/cases/{index}"
        case = require_object(raw_case, path)
        names.append(require_string(case.get("name"), f"{path}/name"))
        if (
            require_integer(case.get("speaker_index"), f"{path}/speaker_index")
            != contract["speaker_index"]
        ):
            raise SourceAcceptanceError(
                f"{path}/speaker_index: differs from oracle lock"
            )
        if (
            require_integer(case.get("seed"), f"{path}/seed")
            != contract["local_ar_seed"]
        ):
            raise SourceAcceptanceError(f"{path}/seed: differs from oracle lock")
        if require_array(case.get("schedule"), f"{path}/schedule") != expected_schedule:
            raise SourceAcceptanceError(
                f"{path}/schedule: expected {expected_schedule}"
            )
    if tuple(names) != EXPECTED_CASE_NAMES:
        raise SourceAcceptanceError(
            f"/cases: expected {EXPECTED_CASE_NAMES}, got {tuple(names)}"
        )
    return tuple(names)


def validate_metric_summary(value: JsonValue, path: str) -> JsonObject:
    summary = require_object(value, path)
    require_exact_keys(summary, METRIC_SUMMARY_FIELDS, path)
    metrics = {
        field: require_number(summary.get(field), f"{path}/{field}")
        for field in METRIC_SUMMARY_FIELDS
    }
    if any(metric < 0.0 for metric in metrics.values()):
        raise SourceAcceptanceError(f"{path}: metrics must be nonnegative")
    if metrics["minimum"] > metrics["median"]:
        raise SourceAcceptanceError(f"{path}: minimum exceeds median")
    if metrics["median"] > metrics["maximum"]:
        raise SourceAcceptanceError(f"{path}: median exceeds maximum")
    if metrics["minimum"] > metrics["p95"]:
        raise SourceAcceptanceError(f"{path}: minimum exceeds p95")
    if metrics["p95"] > metrics["maximum"]:
        raise SourceAcceptanceError(f"{path}: p95 exceeds maximum")
    return summary


def validate_aggregate(value: JsonValue, path: str) -> JsonObject:
    aggregate = require_object(value, path)
    require_exact_keys(aggregate, AGGREGATE_FIELDS, path)
    if require_integer(aggregate.get("case_count"), f"{path}/case_count") != 4:
        raise SourceAcceptanceError(f"{path}/case_count: expected 4")
    for field in (
        "raw_ttfa_ms",
        "gapless_start_ms",
        "max_positive_playback_lateness_ms",
        "generation_rtf",
        "total_rtf",
    ):
        validate_metric_summary(aggregate.get(field), f"{path}/{field}")
    for field in (
        "maximum_peak_cuda_allocated_bytes",
        "maximum_peak_cuda_reserved_bytes",
    ):
        if require_integer(aggregate.get(field), f"{path}/{field}") <= 0:
            raise SourceAcceptanceError(f"{path}/{field}: expected a positive integer")
    return aggregate


def validate_gates(value: JsonValue, path: str) -> JsonObject:
    gates = require_object(value, path)
    require_exact_keys(gates, GATE_FIELDS, path)
    for gate in REQUIRED_PASSED_GATES:
        if gates.get(gate) is not True:
            raise SourceAcceptanceError(f"{path}/{gate}: expected true")
    if gates.get("parakeet_gate_pending") is not True:
        raise SourceAcceptanceError(f"{path}/parakeet_gate_pending: expected true")
    if require_array(gates.get("failures"), f"{path}/failures"):
        raise SourceAcceptanceError(f"{path}/failures: expected an empty array")
    return gates


def validate_determinism(
    value: JsonValue,
    contract: JsonObject,
    path: str,
) -> list[JsonValue]:
    records = require_array(value, path)
    if len(records) != len(EXPECTED_CASE_NAMES):
        raise SourceAcceptanceError(
            f"{path}: expected {len(EXPECTED_CASE_NAMES)} records"
        )
    expected_keys = (
        "case_name",
        "local_ar_seed",
        "first_codes_sha256",
        "replay_codes_sha256",
        "codes_exact",
        "first_pcm_f32le_sha256",
        "replay_pcm_f32le_sha256",
        "pcm_exact",
        "first_codec_frame_count",
        "replay_codec_frame_count",
        "codec_frame_count_exact",
        "passed",
    )
    for index, (raw_record, expected_name) in enumerate(
        zip(records, EXPECTED_CASE_NAMES)
    ):
        record_path = f"{path}/{index}"
        record = require_object(raw_record, record_path)
        require_exact_keys(record, expected_keys, record_path)
        if require_string(record.get("case_name"), f"{record_path}/case_name") != (
            expected_name
        ):
            raise SourceAcceptanceError(
                f"{record_path}/case_name: expected {expected_name}"
            )
        if require_integer(
            record.get("local_ar_seed"),
            f"{record_path}/local_ar_seed",
        ) != contract["local_ar_seed"]:
            raise SourceAcceptanceError(
                f"{record_path}/local_ar_seed: differs from oracle lock"
            )
        first_codes = require_sha256(
            record.get("first_codes_sha256"),
            f"{record_path}/first_codes_sha256",
        )
        replay_codes = require_sha256(
            record.get("replay_codes_sha256"),
            f"{record_path}/replay_codes_sha256",
        )
        first_pcm = require_sha256(
            record.get("first_pcm_f32le_sha256"),
            f"{record_path}/first_pcm_f32le_sha256",
        )
        replay_pcm = require_sha256(
            record.get("replay_pcm_f32le_sha256"),
            f"{record_path}/replay_pcm_f32le_sha256",
        )
        first_frames = require_integer(
            record.get("first_codec_frame_count"),
            f"{record_path}/first_codec_frame_count",
        )
        replay_frames = require_integer(
            record.get("replay_codec_frame_count"),
            f"{record_path}/replay_codec_frame_count",
        )
        if (
            record.get("codes_exact") is not True
            or first_codes != replay_codes
            or record.get("pcm_exact") is not True
            or first_pcm != replay_pcm
            or record.get("codec_frame_count_exact") is not True
            or first_frames <= 0
            or first_frames != replay_frames
            or record.get("passed") is not True
        ):
            raise SourceAcceptanceError(
                f"{record_path}: same-seed replay is not exact"
            )
    return records


def validate_raw_acceptance(
    raw: JsonObject,
    raw_payload: bytes,
    lock: JsonObject,
) -> JsonObject:
    if require_string(raw.get("schema"), "/schema") != RAW_SCHEMA:
        raise SourceAcceptanceError(f"/schema: expected {RAW_SCHEMA}")
    if require_string(raw.get("status"), "/status") != RAW_ACCEPTED_STATUS:
        raise SourceAcceptanceError(f"/status: expected {RAW_ACCEPTED_STATUS}")
    if require_string(raw.get("phase"), "/phase") != "complete":
        raise SourceAcceptanceError("/phase: acceptance probe did not complete")
    if raw.get("source_unchanged") is not True:
        raise SourceAcceptanceError("/source_unchanged: expected true")
    if raw.get("parakeet_transcription_evaluated") is not False:
        raise SourceAcceptanceError(
            "/parakeet_transcription_evaluated: expected false for this gate"
        )

    source_start = require_object(raw.get("source_start"), "/source_start")
    source_end = require_object(raw.get("source_end"), "/source_end")
    if source_start != source_end:
        raise SourceAcceptanceError("/source_end: source changed during acceptance")
    validate_raw_source_section(source_start, lock, "/source_start")
    validate_raw_source_section(source_end, lock, "/source_end")

    gates = validate_gates(raw.get("gates"), "/gates")

    contract = acceptance_contract_from_lock(lock)
    case_names = validate_cases(raw, contract)
    aggregate = validate_aggregate(raw.get("aggregate"), "/aggregate")
    if aggregate["case_count"] != len(case_names):
        raise SourceAcceptanceError("/aggregate/case_count: differs from cases")
    determinism = validate_determinism(
        raw.get("determinism"),
        contract,
        "/determinism",
    )

    static_audit = require_object(raw.get("static_audit"), "/static_audit")
    audit_case_names = require_array(
        static_audit.get("case_names"),
        "/static_audit/case_names",
    )
    if audit_case_names != list(case_names):
        raise SourceAcceptanceError(
            "/static_audit/case_names: differs from executed cases"
        )
    runtime = require_object(raw.get("runtime_environment"), "/runtime_environment")
    if require_array(
        runtime.get("foreign_site_packages"),
        "/runtime_environment/foreign_site_packages",
    ):
        raise SourceAcceptanceError(
            "/runtime_environment/foreign_site_packages: expected an empty array"
        )

    model = model_from_lock(lock)
    public_receipt: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "status": ACCEPTED_STATUS,
        "created_at_utc": canonical_utc_timestamp(
            require_number(raw.get("created_unix_seconds"), "/created_unix_seconds")
        ),
        "source_model": model,
        "oracle_source": source_from_lock(lock),
        "acceptance_contract": contract,
        "probe": {
            "raw_receipt_sha256": sha256_bytes(raw_payload),
            "raw_receipt_size_bytes": len(raw_payload),
            "probe_sha256": require_sha256(
                source_start.get("probe_sha256"),
                "/source_start/probe_sha256",
            ),
            "helper_sha256": require_sha256(
                source_start.get("helper_sha256"),
                "/source_start/helper_sha256",
            ),
            "case_definition_sha256": require_sha256(
                static_audit.get("case_definition_sha256"),
                "/static_audit/case_definition_sha256",
            ),
        },
        "runtime_environment": {
            "torch_version": require_string(
                runtime.get("torch_version"),
                "/runtime_environment/torch_version",
            ),
            "torch_cuda_build": require_string(
                runtime.get("torch_cuda_build"),
                "/runtime_environment/torch_cuda_build",
            ),
            "cudnn_version": require_integer(
                runtime.get("cudnn_version"),
                "/runtime_environment/cudnn_version",
            ),
            "cuda_driver_version": require_string(
                runtime.get("cuda_driver_version"),
                "/runtime_environment/cuda_driver_version",
            ),
            "gpu_name": require_string(
                runtime.get("gpu_name"),
                "/runtime_environment/gpu_name",
            ),
            "gpu_compute_capability": list(
                require_array(
                    runtime.get("gpu_compute_capability"),
                    "/runtime_environment/gpu_compute_capability",
                )
            ),
        },
        "evidence": {
            "case_names": list(case_names),
            "determinism": list(determinism),
            "aggregate": dict(aggregate),
            "gates": dict(gates),
        },
    }
    reject_absolute_paths(public_receipt)
    validate_public_acceptance(public_receipt, lock)
    return public_receipt


def reject_absolute_paths(value: JsonValue, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            reject_absolute_paths(child, f"{path}/{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            reject_absolute_paths(child, f"{path}/{index}")
        return
    if not isinstance(value, str):
        return
    if (
        value.startswith("/")
        or value.startswith("file:")
        or WINDOWS_ABSOLUTE_PATH_PATTERN.match(value) is not None
    ):
        raise SourceAcceptanceError(
            f"{path or '/'}: public receipt contains an absolute path"
        )


def validate_public_acceptance(receipt: JsonObject, lock: JsonObject) -> None:
    require_exact_keys(
        receipt,
        (
            "schema_version",
            "artifact_role",
            "status",
            "created_at_utc",
            "source_model",
            "oracle_source",
            "acceptance_contract",
            "probe",
            "runtime_environment",
            "evidence",
        ),
        "/",
    )
    if require_integer(receipt.get("schema_version"), "/schema_version") != 2:
        raise SourceAcceptanceError("/schema_version: expected 2")
    if require_string(receipt.get("artifact_role"), "/artifact_role") != ARTIFACT_ROLE:
        raise SourceAcceptanceError(f"/artifact_role: expected {ARTIFACT_ROLE}")
    if require_string(receipt.get("status"), "/status") != ACCEPTED_STATUS:
        raise SourceAcceptanceError(f"/status: expected {ACCEPTED_STATUS}")
    created_at = require_string(receipt.get("created_at_utc"), "/created_at_utc")
    try:
        parsed_timestamp = datetime.datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise SourceAcceptanceError(
            "/created_at_utc: expected an ISO-8601 timestamp"
        ) from error
    if parsed_timestamp.tzinfo != datetime.UTC or not created_at.endswith("Z"):
        raise SourceAcceptanceError("/created_at_utc: expected canonical UTC")

    model = model_from_lock(lock)
    receipt_model = require_object(receipt.get("source_model"), "/source_model")
    if receipt_model != model:
        raise SourceAcceptanceError("/source_model: differs from oracle lock")
    if require_object(receipt.get("oracle_source"), "/oracle_source") != source_from_lock(
        lock
    ):
        raise SourceAcceptanceError("/oracle_source: differs from oracle lock")
    if require_object(
        receipt.get("acceptance_contract"),
        "/acceptance_contract",
    ) != acceptance_contract_from_lock(lock):
        raise SourceAcceptanceError(
            "/acceptance_contract: differs from oracle lock"
        )

    probe = require_object(receipt.get("probe"), "/probe")
    require_exact_keys(
        probe,
        (
            "raw_receipt_sha256",
            "raw_receipt_size_bytes",
            "probe_sha256",
            "helper_sha256",
            "case_definition_sha256",
        ),
        "/probe",
    )
    for key in (
        "raw_receipt_sha256",
        "probe_sha256",
        "helper_sha256",
        "case_definition_sha256",
    ):
        require_sha256(probe.get(key), f"/probe/{key}")
    if require_integer(
        probe.get("raw_receipt_size_bytes"),
        "/probe/raw_receipt_size_bytes",
    ) <= 0:
        raise SourceAcceptanceError(
            "/probe/raw_receipt_size_bytes: expected a positive integer"
        )

    runtime = require_object(
        receipt.get("runtime_environment"),
        "/runtime_environment",
    )
    require_exact_keys(
        runtime,
        (
            "torch_version",
            "torch_cuda_build",
            "cudnn_version",
            "cuda_driver_version",
            "gpu_name",
            "gpu_compute_capability",
        ),
        "/runtime_environment",
    )
    for key in ("torch_version", "torch_cuda_build", "cuda_driver_version", "gpu_name"):
        require_string(runtime.get(key), f"/runtime_environment/{key}")
    if require_integer(
        runtime.get("cudnn_version"),
        "/runtime_environment/cudnn_version",
    ) <= 0:
        raise SourceAcceptanceError(
            "/runtime_environment/cudnn_version: expected a positive integer"
        )
    capability = require_array(
        runtime.get("gpu_compute_capability"),
        "/runtime_environment/gpu_compute_capability",
    )
    if capability != [11, 0]:
        raise SourceAcceptanceError(
            "/runtime_environment/gpu_compute_capability: expected [11, 0]"
        )

    evidence = require_object(receipt.get("evidence"), "/evidence")
    require_exact_keys(
        evidence,
        ("case_names", "determinism", "aggregate", "gates"),
        "/evidence",
    )
    if require_array(evidence.get("case_names"), "/evidence/case_names") != list(
        EXPECTED_CASE_NAMES
    ):
        raise SourceAcceptanceError(
            f"/evidence/case_names: expected {EXPECTED_CASE_NAMES}"
        )
    validate_determinism(
        evidence.get("determinism"),
        acceptance_contract_from_lock(lock),
        "/evidence/determinism",
    )
    validate_aggregate(evidence.get("aggregate"), "/evidence/aggregate")
    validate_gates(evidence.get("gates"), "/evidence/gates")
    reject_absolute_paths(receipt)


def write_no_replace(path: Path, receipt: JsonObject) -> tuple[str, int]:
    output = path.absolute()
    parent = output.parent.resolve(strict=True)
    if output.parent.absolute() != parent:
        raise SourceAcceptanceError(
            f"output parent must not contain symlinks or '..': {path.parent}"
        )
    payload = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise SourceAcceptanceError("receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return sha256_bytes(payload), len(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a portable source-model acceptance receipt v2",
    )
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock, _ = load_json(args.lock)
    raw, raw_payload = load_json(args.raw_receipt)
    receipt = validate_raw_acceptance(raw, raw_payload, lock)
    receipt_sha256, receipt_size = write_no_replace(args.output, receipt)
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "receipt_sha256": receipt_sha256,
                "receipt_size_bytes": receipt_size,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SourceAcceptanceError) as error:
        print(f"source acceptance finalization failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

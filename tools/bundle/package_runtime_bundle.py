#!/usr/bin/env python3
"""Create one immutable, introspected MagpieTTS-RT runtime bundle.

The package specification contains only non-engine semantics. Tensor names,
dtypes, declared shapes, optimization-profile ranges, KV dimensions, and
NanoCodec state bindings are obtained from deserialized TensorRT plans. The
tool never accepts hand-written engine shapes and never replaces an output.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import errno
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

SCHEMA_VERSION = 1
MANIFEST_NAME = "runtime-bundle-manifest.json"
MANIFEST_DIGEST_NAME = "runtime-bundle-manifest.json.sha256"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_SNAPSHOT_BYTES = 16 * 1024 * 1024 * 1024
PLUGIN_ABI_VERSION = 1
PLUGIN_CREATOR_COUNT = 5
PLUGIN_STATUS_OK = 0
PLUGIN_STATUS_ALREADY_REGISTERED = 1
EXPECTED_PLUGIN_CREATORS = (
    ("MagpieLocalARSampling", "1", "magpie_tts_rt"),
    ("MagpieLocalAREos", "1", "magpie_tts_rt"),
    ("MagpieLayerNorm", "1", "magpie_tts_rt"),
    ("MagpieGeluTanh", "1", "magpie_tts_rt"),
    ("MagpieSoftmax", "1", "magpie_tts_rt"),
)
EXPECTED_PLUGIN_SONAME = "libmagpie_tts_rt_plugins.so.0"
EXPECTED_PLUGIN_FILENAME = "libmagpie_tts_rt_plugins.so.0.1.0"
EXPECTED_PLUGIN_SOURCE_PATHS = (
    "CMakeLists.txt",
    "cmake/magpie_tts_rt_plugins.map",
    "include/magpie_tts_rt/magpie_tts_rt_plugin.h",
    "plugins/local_ar_plugins.cu",
    "plugins/local_ar_plugins.hpp",
)
EXPECTED_PLUGIN_NEEDED = frozenset(
    {
        "libcublas.so.13",
        "libcudart.so.13",
        "libnvinfer.so.10",
        "libstdc++.so.6",
        "libgcc_s.so.1",
        "libc.so.6",
        "ld-linux-aarch64.so.1",
    }
)
FORBIDDEN_PLUGIN_DYNAMIC_TAGS = frozenset(
    {"RPATH", "RUNPATH", "AUDIT", "DEPAUDIT", "FILTER", "AUXILIARY"}
)
ENGINE_ROLES = (
    "text_encoder",
    "main_decoder_prefill",
    "main_decoder_step",
    "local_ar_16",
    "nanocodec_initial_4",
    "nanocodec_steady_8",
    "nanocodec_tail_1_8",
)
LICENSE_ROLES = (
    "project_license",
    "project_notice",
    "pytorch_license",
    "cutlass_license",
    "cuda_eula",
    "cuda_notice",
    "nvidia_open_model_license",
    "nvidia_model_notice",
)
NVIDIA_OPEN_MODEL_LICENSE_NAME = "NVIDIA Open Model License Agreement"
NVIDIA_OPEN_MODEL_LICENSE_VERSION = "2025-10-24"
NVIDIA_OPEN_MODEL_LICENSE_FILE_NAME = (
    "NVIDIA-Open-Model-License-Agreement-2025-10-24.pdf"
)
NVIDIA_OPEN_MODEL_LICENSE_SHA256 = (
    "4d2fb590aa9b30c47f2058bff17291df7fa2aa0c1bd775a20703da9bb267cfab"
)
NVIDIA_MODEL_NOTICE_FILE_NAME = "NOTICE"
NVIDIA_MODEL_NOTICE_SHA256 = (
    "5fc60716a9ba57f3792e71ed776b4065671c64d0c2db716698f4d646ad833eb1"
)
ORACLE_SOURCE_PATHS = (
    "nemo/collections/tts/models/easy_magpietts_inference.py",
    "nemo/collections/tts/models/magpietts.py",
    "nemo/collections/tts/models/magpietts_preference_optimization.py",
    "nemo/collections/tts/modules/ffn_modules.py",
    "nemo/collections/tts/modules/magpietts_fused_sampling.py",
    "nemo/collections/tts/modules/magpietts_inference/inference.py",
    "nemo/collections/tts/modules/magpietts_inference/utils.py",
    "nemo/collections/tts/modules/magpietts_modules.py",
    "nemo/collections/tts/modules/streaming_codec.py",
    "nemo/collections/tts/modules/streaming_synthesis.py",
    "nemo/collections/tts/modules/transformer_2501.py",
)
SOURCE_ACCEPTANCE_PASSED_GATES = (
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
SOURCE_ACCEPTANCE_CASE_NAMES = (
    "short_conversation",
    "punctuation",
    "latin_and_numbers",
    "long_multi_chunk",
)
SOURCE_ACCEPTANCE_AGGREGATE_FIELDS = (
    "case_count",
    "raw_ttfa_ms",
    "gapless_start_ms",
    "max_positive_playback_lateness_ms",
    "generation_rtf",
    "total_rtf",
    "maximum_peak_cuda_allocated_bytes",
    "maximum_peak_cuda_reserved_bytes",
)
LOCAL_AR_CODEBOOK_SIZE = 2016
LOCAL_AR_VOCABULARY_SIZE = 2024
LOCAL_AR_AUDIO_EOS_ID = 2017
LOCAL_AR_STATIC_FORBIDDEN_TOKEN_IDS = (
    2016,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
)
SOFIA_LOCAL_AR_POSITION_TABLE_SHA256 = (
    "1db63ebd4ceffba52e03cf67c9d186f3b7e38bb0c6eb9056a93ceeb55a4a695e"
)
ZERO_FRAME_FINALIZATION = "control_marker_without_codec_invocation"
EXPORT_FORMAT = "magpie_tts_rt_export_v1"
ENGINE_NAMES_BY_ROLE = {
    "text_encoder": "text_encoder",
    "main_decoder_prefill": "main_decoder_prefill",
    "main_decoder_step": "main_decoder_step",
    "local_ar_16": "local_ar",
    "nanocodec_initial_4": "nanocodec_initial",
    "nanocodec_steady_8": "nanocodec_steady",
    "nanocodec_tail_1_8": "nanocodec_tail",
}
PROFILE_NAMES_BY_ROLE = {
    "text_encoder": "text_1_512",
    "main_decoder_prefill": "text_1_512",
    "main_decoder_step": "text_1_512",
    "local_ar_16": "fixed",
    "nanocodec_initial_4": "fixed",
    "nanocodec_steady_8": "fixed",
    "nanocodec_tail_1_8": "tail_1_8",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PackageError(RuntimeError):
    """Fail-closed package construction error."""


@dataclass(frozen=True)
class RuntimeFingerprint:
    os_name: str
    os_version: str
    architecture: str
    endianness: str
    cuda_version: str
    tensorrt_version: str
    driver_version: str
    gpu_name: str
    gpu_compute_capability: str
    plugin_abi_version: int

    def to_json(self) -> JsonObject:
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "endianness": self.endianness,
            "cuda_version": self.cuda_version,
            "tensorrt_version": self.tensorrt_version,
            "driver_version": self.driver_version,
            "gpu_name": self.gpu_name,
            "gpu_compute_capability": self.gpu_compute_capability,
            "plugin_abi_version": self.plugin_abi_version,
        }


@dataclass(frozen=True)
class SourceModelSpec:
    model_id: str
    version: str
    revision: str
    source_sha256: str


@dataclass(frozen=True)
class ExportSpec:
    format: str
    source_revision: str
    voice_id: str
    baked_context_length: int
    baked_context_sha256: str
    audio_bos_baked: bool
    receipt_artifact_role: str


@dataclass(frozen=True)
class TokenizerSpecialTokensSpec:
    bos_token_id: int
    eos_token_id: int
    japanese_global_pad_token_id: int

    def to_json(self) -> JsonObject:
        return {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "japanese_global_pad_token_id": (
                self.japanese_global_pad_token_id
            ),
        }


@dataclass(frozen=True)
class TokenizerSpec:
    kind: str
    tokenizer_vocabulary_size: int
    text_embedding_rows: int
    special_tokens: TokenizerSpecialTokensSpec
    identity_sha256: str


@dataclass(frozen=True)
class PluginSpec:
    name: str


@dataclass(frozen=True)
class ClassifierFreeGuidanceSpec:
    row_order: str
    conditional_row_index: int
    conditional_condition_source: str
    conditional_mask_source: str
    unconditional_row_index: int
    unconditional_condition_source: str
    unconditional_mask_source: str

    def to_json(self) -> JsonObject:
        return {
            "row_order": self.row_order,
            "conditional_row_index": self.conditional_row_index,
            "conditional_condition_source": self.conditional_condition_source,
            "conditional_mask_source": self.conditional_mask_source,
            "unconditional_row_index": self.unconditional_row_index,
            "unconditional_condition_source": self.unconditional_condition_source,
            "unconditional_mask_source": self.unconditional_mask_source,
        }


@dataclass(frozen=True)
class KvCacheSpec:
    layout: str
    prefix_length: int
    maximum_generated_steps: int
    update_mode: str
    position_semantics: str
    first_step_position: int
    step_position_upper_bound_exclusive: int


@dataclass(frozen=True)
class AlignmentSpec:
    dtype: str
    source_decoder_layers: tuple[int, ...]
    prefill_output_binding: str
    step_prior_input_binding: str
    step_alignment_output_binding: str
    prior_epsilon: float
    initial_attended: int
    ignored_terminal_tokens: int
    short_text_no_prior_max_tokens: int
    lookahead: int
    sink_threshold: int
    source_position_policy: str

    def to_json(self) -> JsonObject:
        return {
            "dtype": self.dtype,
            "source_decoder_layers": list(self.source_decoder_layers),
            "prefill_output_binding": self.prefill_output_binding,
            "step_prior_input_binding": self.step_prior_input_binding,
            "step_alignment_output_binding": self.step_alignment_output_binding,
            "prior_epsilon": self.prior_epsilon,
            "initial_attended": self.initial_attended,
            "ignored_terminal_tokens": self.ignored_terminal_tokens,
            "short_text_no_prior_max_tokens": self.short_text_no_prior_max_tokens,
            "lookahead": self.lookahead,
            "sink_threshold": self.sink_threshold,
            "source_position_policy": self.source_position_policy,
        }


@dataclass(frozen=True)
class RngSpec:
    algorithm: str
    seed_bits: int
    counter_bits: int
    state_location: str
    ownership: str
    deterministic: bool

    def to_json(self) -> JsonObject:
        return {
            "algorithm": self.algorithm,
            "seed_bits": self.seed_bits,
            "counter_bits": self.counter_bits,
            "state_location": self.state_location,
            "ownership": self.ownership,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class SamplingSpec:
    algorithm: str
    top_k: int
    temperature: float
    eos_token_id: int
    forbidden_token_ids: tuple[int, ...]
    invalid_distribution_policy: str
    next_embedding_location: str
    rng: RngSpec

    def to_json(self) -> JsonObject:
        return {
            "algorithm": self.algorithm,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "eos_token_id": self.eos_token_id,
            "forbidden_token_ids": list(self.forbidden_token_ids),
            "invalid_distribution_policy": self.invalid_distribution_policy,
            "next_embedding_location": self.next_embedding_location,
            "rng": self.rng.to_json(),
        }


@dataclass(frozen=True)
class PositionEmbeddingSpec:
    kind: str
    positions: tuple[int, ...]
    source_shape: tuple[int, ...]
    dtype: str
    source_table_sha256: str

    def to_json(self) -> JsonObject:
        return {
            "kind": self.kind,
            "positions": list(self.positions),
            "source_shape": list(self.source_shape),
            "dtype": self.dtype,
            "source_table_sha256": self.source_table_sha256,
        }


@dataclass(frozen=True)
class LocalArSpec:
    engine_name: str
    execution: str
    iterations: int
    positions: tuple[int, ...]
    position_embedding: PositionEmbeddingSpec
    codebooks_per_frame: int
    frames_per_decoder_step: int
    sampling_plugin_name: str
    invalid_rows_encoding: str
    no_eos_frame_index: int

    def to_json(self) -> JsonObject:
        return {
            "engine_name": self.engine_name,
            "execution": self.execution,
            "iterations": self.iterations,
            "positions": list(self.positions),
            "position_embedding": self.position_embedding.to_json(),
            "codebooks_per_frame": self.codebooks_per_frame,
            "frames_per_decoder_step": self.frames_per_decoder_step,
            "sampling_plugin_name": self.sampling_plugin_name,
            "invalid_rows_encoding": self.invalid_rows_encoding,
            "no_eos_frame_index": self.no_eos_frame_index,
        }


@dataclass(frozen=True)
class CodecSpec:
    initial_engine_name: str
    steady_engine_name: str
    tail_engine_name: str
    sample_rate_hz: int
    hop_length_samples: int
    channels: int
    pcm_format: str
    stateful: bool
    initial_frames: int
    steady_frames: int
    tail_min_frames: int
    tail_max_frames: int
    eos_frame_is_audio: bool
    zero_frame_finalization: str


@dataclass(frozen=True)
class LimitsSpec:
    maximum_text_tokens: int
    maximum_decoder_steps: int
    maximum_audio_frames: int
    maximum_sessions: int
    maximum_concurrent_requests: int
    pcm_ring_capacity_frames: int
    maximum_workspace_bytes: int
    maximum_device_memory_bytes: int


@dataclass(frozen=True)
class EngineDestination:
    name: str
    role: str
    path: str


@dataclass(frozen=True)
class LicenseDestination:
    role: str
    path: str


@dataclass(frozen=True)
class Destinations:
    source_model_acceptance_receipt: str
    export_receipt: str
    tokenizer_identity_receipt: str
    plugin_build_receipt: str
    plugin: str
    golden_fixture: str
    golden_receipt: str
    licenses: tuple[LicenseDestination, ...]
    engines: tuple[EngineDestination, ...]


@dataclass(frozen=True)
class PackageSpec:
    schema_version: int
    bundle_id: str
    created_at_utc: str
    source_model: SourceModelSpec
    export: ExportSpec
    tokenizer: TokenizerSpec
    plugin: PluginSpec
    classifier_free_guidance: ClassifierFreeGuidanceSpec
    kv_cache: KvCacheSpec
    alignment: AlignmentSpec
    sampling: SamplingSpec
    local_ar: LocalArSpec
    codec: CodecSpec
    limits: LimitsSpec
    destinations: Destinations


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    location: str = "device"
    shape_inference_io: bool = False

    def to_json(self) -> JsonObject:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "location": self.location,
            "shape_inference_io": self.shape_inference_io,
        }


@dataclass(frozen=True)
class TensorShapeRange:
    tensor_name: str
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def to_json(self) -> JsonObject:
        return {
            "tensor_name": self.tensor_name,
            "min": list(self.minimum),
            "opt": list(self.optimum),
            "max": list(self.maximum),
        }


@dataclass(frozen=True)
class TensorValueRange:
    tensor_name: str
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def to_json(self) -> JsonObject:
        return {
            "tensor_name": self.tensor_name,
            "min": list(self.minimum),
            "opt": list(self.optimum),
            "max": list(self.maximum),
        }


@dataclass(frozen=True)
class OptimizationProfile:
    name: str
    input_shapes: tuple[TensorShapeRange, ...]
    input_values: tuple[TensorValueRange, ...] = ()

    def to_json(self) -> JsonObject:
        return {
            "name": self.name,
            "input_shapes": [shape.to_json() for shape in self.input_shapes],
            "input_values": [values.to_json() for values in self.input_values],
        }


@dataclass(frozen=True)
class InspectedEngine:
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    profiles: tuple[OptimizationProfile, ...]


@dataclass(frozen=True)
class FileArtifact:
    path: str
    sha256: str
    size_bytes: int

    def to_json(self) -> JsonObject:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class CopiedArtifact:
    artifact: FileArtifact
    source_device: int
    source_inode: int


@dataclass(frozen=True)
class BundleInventoryEntry:
    mode: int
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class PluginIdentity:
    abi_version: int
    creators: tuple[tuple[str, str, str], ...]
    library: ctypes.CDLL


@dataclass(frozen=True)
class PluginDynamicContract:
    soname: str
    needed: frozenset[str]


@dataclass(frozen=True)
class GoldenReceipt:
    receipt_version: int
    created_at_utc: str
    normalized_text_sha256: str
    token_ids_sha256: str
    baked_context_sha256: str
    seed: int
    decoder_tokens_sha256: str
    codec_codes_sha256: str
    codec_frame_count: int
    pcm_f32le_sha256: str
    sample_count: int
    initial_frames: int
    steady_frames: int
    tail_min_frames: int
    tail_max_frames: int
    eos_frame_is_audio: bool
    zero_frame_finalization: str

    def manifest_json(self, file: FileArtifact) -> JsonObject:
        return {
            "receipt_version": self.receipt_version,
            "path": file.path,
            "sha256": file.sha256,
            "size_bytes": file.size_bytes,
            "created_at_utc": self.created_at_utc,
            "normalized_text_sha256": self.normalized_text_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "baked_context_sha256": self.baked_context_sha256,
            "seed": self.seed,
            "decoder_tokens_sha256": self.decoder_tokens_sha256,
            "codec_codes_sha256": self.codec_codes_sha256,
            "codec_frame_count": self.codec_frame_count,
            "pcm_f32le_sha256": self.pcm_f32le_sha256,
            "sample_count": self.sample_count,
            "initial_frames": self.initial_frames,
            "steady_frames": self.steady_frames,
            "tail_min_frames": self.tail_min_frames,
            "tail_max_frames": self.tail_max_frames,
            "eos_frame_is_audio": self.eos_frame_is_audio,
            "zero_frame_finalization": self.zero_frame_finalization,
        }


@dataclass(frozen=True)
class GoldenFixtureExpected:
    decoder_tokens_sha256: str
    codec_codes_sha256: str
    codec_frame_count: int
    pcm_f32le_sha256: str
    pcm_sample_count: int


@dataclass(frozen=True)
class GoldenFixture:
    schema_version: int
    fixture_id: str
    prepared_token_ids: tuple[int, ...]
    seed: int
    tokenizer_identity_sha256: str
    oracle_lock_sha256: str
    normalized_text_sha256: str
    token_ids_sha256: str
    baked_context_sha256: str
    expected: GoldenFixtureExpected


@dataclass(frozen=True)
class ExportReceiptEvidence:
    oracle_lock_sha256: str


@dataclass(frozen=True)
class PluginSourceEvidence:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PluginBuildReceiptEvidence:
    receipt_sha256: str
    receipt_size_bytes: int
    plugin_sha256: str
    plugin_size_bytes: int
    sources: tuple[PluginSourceEvidence, ...]


@dataclass(frozen=True)
class InputPaths:
    specification: Path
    source_model_acceptance_receipt: Path
    export_receipt: Path
    tokenizer_identity_receipt: Path
    plugin_build_receipt: Path
    plugin: Path
    golden_fixture: Path
    golden_receipt: Path
    manifest_validator: Path
    bundle_validator: Path
    licenses: tuple[tuple[str, Path], ...]
    engines: tuple[tuple[str, Path], ...]


class PluginCreator(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("name", ctypes.c_char * 64),
        ("version", ctypes.c_char * 16),
        ("plugin_namespace", ctypes.c_char * 64),
    ]


PluginRegisterFunction = ctypes.CFUNCTYPE(ctypes.c_int32)


class PluginApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("creator_count", ctypes.c_uint32),
        ("reserved_0", ctypes.c_uint32),
        ("creators", PluginCreator * PLUGIN_CREATOR_COUNT),
        ("register_plugins", PluginRegisterFunction),
        ("reserved", ctypes.c_uint64 * 4),
    ]


def canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: JsonValue) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_json(value: JsonValue, path: str = "/") -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PackageError(f"{path}: non-finite JSON number")
        return value
    if isinstance(value, list):
        return [
            normalize_json(item, f"{path}/{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PackageError(f"{path}: JSON object key is not a string")
            normalized[key] = normalize_json(item, f"{path}/{key}")
        return normalized
    raise PackageError(f"{path}: unsupported JSON value type")


def reject_duplicate_json_object(
    pairs: list[tuple[str, JsonValue]],
) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_verified_regular_file(
    path: Path,
    label: str,
    maximum_bytes: int,
) -> bytes:
    require_regular_non_symlink(path, label)
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise PackageError(f"{label} changed before reading: {path}")
        if before.st_size > maximum_bytes:
            raise PackageError(
                f"{label} exceeds {maximum_bytes} bytes: {path}"
            )
        chunks: list[bytes] = []
        read_bytes = 0
        while read_bytes < before.st_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, before.st_size - read_bytes),
            )
            if not chunk:
                break
            chunks.append(chunk)
            read_bytes += len(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            before.st_dev == after.st_dev,
            before.st_ino == after.st_ino,
            before.st_size == after.st_size,
            before.st_mtime_ns == after.st_mtime_ns,
            before.st_ctime_ns == after.st_ctime_ns,
        )
        if not all(stable_fields) or read_bytes != before.st_size:
            raise PackageError(f"{label} changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def parse_json_payload(payload: bytes, path: Path) -> JsonValue:
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageError(f"{path}: invalid UTF-8 JSON: {error}") from error
    return normalize_json(raw)


def load_json_file(path: Path) -> JsonValue:
    payload = read_verified_regular_file(
        path, "JSON input", MAX_MANIFEST_BYTES
    )
    return parse_json_payload(payload, path)


def require_object(
    value: JsonValue,
    required_keys: tuple[str, ...],
    path: str,
) -> JsonObject:
    if not isinstance(value, dict):
        raise PackageError(f"{path}: expected an object")
    actual = set(value)
    expected = set(required_keys)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise PackageError(
            f"{path}: keys mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def require_string(value: JsonValue, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise PackageError(f"{path}: expected a non-empty string")
    return value


def require_identifier(value: JsonValue, path: str) -> str:
    result = require_string(value, path)
    if IDENTIFIER_PATTERN.fullmatch(result) is None:
        raise PackageError(f"{path}: invalid identifier")
    return result


def require_sha256(value: JsonValue, path: str) -> str:
    result = require_string(value, path)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise PackageError(f"{path}: expected a lowercase SHA-256")
    return result


def require_git_sha1(value: JsonValue, path: str) -> str:
    result = require_string(value, path)
    if GIT_SHA1_PATTERN.fullmatch(result) is None:
        raise PackageError(f"{path}: expected a lowercase Git SHA-1")
    return result


def require_bool(value: JsonValue, path: str) -> bool:
    if not isinstance(value, bool):
        raise PackageError(f"{path}: expected a boolean")
    return value


def require_int(
    value: JsonValue,
    path: str,
    minimum: int = 0,
    maximum: int = (1 << 64) - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PackageError(f"{path}: expected an integer")
    if value < minimum or value > maximum:
        raise PackageError(
            f"{path}: integer outside [{minimum}, {maximum}]"
        )
    return value


def require_float(value: JsonValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageError(f"{path}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PackageError(f"{path}: expected a finite number")
    return result


def require_int_tuple(
    value: JsonValue,
    path: str,
    minimum: int = 0,
    maximum: int = (1 << 32) - 1,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise PackageError(f"{path}: expected an integer array")
    return tuple(
        require_int(item, f"{path}/{index}", minimum, maximum)
        for index, item in enumerate(value)
    )


def require_timestamp(value: JsonValue, path: str) -> str:
    result = require_string(value, path)
    if not result.endswith("Z"):
        raise PackageError(f"{path}: timestamp must end in Z")
    try:
        parsed = datetime.datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as error:
        raise PackageError(f"{path}: invalid UTC timestamp: {error}") from error
    if parsed.utcoffset() != datetime.timedelta(0):
        raise PackageError(f"{path}: timestamp is not UTC")
    return result


def require_relative_path(value: JsonValue, path: str) -> str:
    text = require_string(value, path)
    has_control = any(
        ord(character) <= 31 or ord(character) == 127 for character in text
    )
    if "\\" in text or has_control:
        raise PackageError(f"{path}: path contains a forbidden byte")
    pure = PurePosixPath(text)
    if pure.is_absolute() or text != pure.as_posix():
        raise PackageError(f"{path}: path must be normalized and relative")
    if any(part in ("", ".", "..") for part in pure.parts):
        raise PackageError(f"{path}: '.', '..', and empty components are forbidden")
    return text


def parse_source_model(value: JsonValue) -> SourceModelSpec:
    document = require_object(
        value,
        ("model_id", "version", "revision", "source_sha256"),
        "/source_model",
    )
    return SourceModelSpec(
        model_id=require_string(document["model_id"], "/source_model/model_id"),
        version=require_string(document["version"], "/source_model/version"),
        revision=require_git_sha1(
            document["revision"], "/source_model/revision"
        ),
        source_sha256=require_sha256(
            document["source_sha256"], "/source_model/source_sha256"
        ),
    )


def parse_export(value: JsonValue) -> ExportSpec:
    document = require_object(
        value,
        (
            "format",
            "source_revision",
            "voice_id",
            "baked_context_length",
            "baked_context_sha256",
            "audio_bos_baked",
            "receipt_artifact_role",
        ),
        "/export",
    )
    export_format = require_identifier(document["format"], "/export/format")
    if export_format != EXPORT_FORMAT:
        raise PackageError(
            f"/export/format: expected {EXPORT_FORMAT}, got {export_format}"
        )
    return ExportSpec(
        format=export_format,
        source_revision=require_string(
            document["source_revision"], "/export/source_revision"
        ),
        voice_id=require_identifier(document["voice_id"], "/export/voice_id"),
        baked_context_length=require_int(
            document["baked_context_length"],
            "/export/baked_context_length",
            1,
            (1 << 32) - 1,
        ),
        baked_context_sha256=require_sha256(
            document["baked_context_sha256"],
            "/export/baked_context_sha256",
        ),
        audio_bos_baked=require_bool(
            document["audio_bos_baked"], "/export/audio_bos_baked"
        ),
        receipt_artifact_role=require_identifier(
            document["receipt_artifact_role"],
            "/export/receipt_artifact_role",
        ),
    )


def parse_tokenizer(value: JsonValue) -> TokenizerSpec:
    document = require_object(
        value,
        (
            "kind",
            "tokenizer_vocabulary_size",
            "text_embedding_rows",
            "special_tokens",
            "identity_sha256",
        ),
        "/tokenizer",
    )
    tokenizer_vocabulary_size = require_int(
        document["tokenizer_vocabulary_size"],
        "/tokenizer/tokenizer_vocabulary_size",
        1,
        (1 << 31) - 1,
    )
    text_embedding_rows = require_int(
        document["text_embedding_rows"],
        "/tokenizer/text_embedding_rows",
        tokenizer_vocabulary_size + 1,
        (1 << 31) - 1,
    )
    special_document = require_object(
        document["special_tokens"],
        (
            "bos_token_id",
            "eos_token_id",
            "japanese_global_pad_token_id",
        ),
        "/tokenizer/special_tokens",
    )
    special_tokens = TokenizerSpecialTokensSpec(
        bos_token_id=require_int(
            special_document["bos_token_id"],
            "/tokenizer/special_tokens/bos_token_id",
            tokenizer_vocabulary_size,
            text_embedding_rows - 1,
        ),
        eos_token_id=require_int(
            special_document["eos_token_id"],
            "/tokenizer/special_tokens/eos_token_id",
            tokenizer_vocabulary_size,
            text_embedding_rows - 1,
        ),
        japanese_global_pad_token_id=require_int(
            special_document["japanese_global_pad_token_id"],
            "/tokenizer/special_tokens/japanese_global_pad_token_id",
            0,
            tokenizer_vocabulary_size - 1,
        ),
    )
    if special_tokens.bos_token_id == special_tokens.eos_token_id:
        raise PackageError(
            "/tokenizer/special_tokens: BOS and EOS identifiers must differ"
        )
    return TokenizerSpec(
        kind=require_identifier(document["kind"], "/tokenizer/kind"),
        tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        text_embedding_rows=text_embedding_rows,
        special_tokens=special_tokens,
        identity_sha256=require_sha256(
            document["identity_sha256"], "/tokenizer/identity_sha256"
        ),
    )


def parse_plugin(value: JsonValue) -> PluginSpec:
    document = require_object(value, ("name",), "/plugin")
    return PluginSpec(
        name=require_identifier(document["name"], "/plugin/name")
    )


def parse_classifier_free_guidance(
    value: JsonValue,
) -> ClassifierFreeGuidanceSpec:
    keys = (
        "row_order",
        "conditional_row_index",
        "conditional_condition_source",
        "conditional_mask_source",
        "unconditional_row_index",
        "unconditional_condition_source",
        "unconditional_mask_source",
    )
    document = require_object(value, keys, "/classifier_free_guidance")
    return ClassifierFreeGuidanceSpec(
        row_order=require_identifier(
            document["row_order"], "/classifier_free_guidance/row_order"
        ),
        conditional_row_index=require_int(
            document["conditional_row_index"],
            "/classifier_free_guidance/conditional_row_index",
            0,
            (1 << 32) - 1,
        ),
        conditional_condition_source=require_identifier(
            document["conditional_condition_source"],
            "/classifier_free_guidance/conditional_condition_source",
        ),
        conditional_mask_source=require_identifier(
            document["conditional_mask_source"],
            "/classifier_free_guidance/conditional_mask_source",
        ),
        unconditional_row_index=require_int(
            document["unconditional_row_index"],
            "/classifier_free_guidance/unconditional_row_index",
            0,
            (1 << 32) - 1,
        ),
        unconditional_condition_source=require_identifier(
            document["unconditional_condition_source"],
            "/classifier_free_guidance/unconditional_condition_source",
        ),
        unconditional_mask_source=require_identifier(
            document["unconditional_mask_source"],
            "/classifier_free_guidance/unconditional_mask_source",
        ),
    )


def parse_kv_cache(value: JsonValue) -> KvCacheSpec:
    keys = (
        "layout",
        "prefix_length",
        "maximum_generated_steps",
        "update_mode",
        "position_semantics",
        "first_step_position",
        "step_position_upper_bound_exclusive",
    )
    document = require_object(value, keys, "/kv_cache")
    return KvCacheSpec(
        layout=require_identifier(document["layout"], "/kv_cache/layout"),
        prefix_length=require_int(
            document["prefix_length"], "/kv_cache/prefix_length", 1
        ),
        maximum_generated_steps=require_int(
            document["maximum_generated_steps"],
            "/kv_cache/maximum_generated_steps",
            1,
        ),
        update_mode=require_identifier(
            document["update_mode"], "/kv_cache/update_mode"
        ),
        position_semantics=require_identifier(
            document["position_semantics"], "/kv_cache/position_semantics"
        ),
        first_step_position=require_int(
            document["first_step_position"], "/kv_cache/first_step_position"
        ),
        step_position_upper_bound_exclusive=require_int(
            document["step_position_upper_bound_exclusive"],
            "/kv_cache/step_position_upper_bound_exclusive",
            1,
        ),
    )


def parse_alignment(value: JsonValue) -> AlignmentSpec:
    keys = (
        "dtype",
        "source_decoder_layers",
        "prefill_output_binding",
        "step_prior_input_binding",
        "step_alignment_output_binding",
        "prior_epsilon",
        "initial_attended",
        "ignored_terminal_tokens",
        "short_text_no_prior_max_tokens",
        "lookahead",
        "sink_threshold",
        "source_position_policy",
    )
    document = require_object(value, keys, "/alignment")
    return AlignmentSpec(
        dtype=require_identifier(document["dtype"], "/alignment/dtype"),
        source_decoder_layers=require_int_tuple(
            document["source_decoder_layers"],
            "/alignment/source_decoder_layers",
        ),
        prefill_output_binding=require_identifier(
            document["prefill_output_binding"],
            "/alignment/prefill_output_binding",
        ),
        step_prior_input_binding=require_identifier(
            document["step_prior_input_binding"],
            "/alignment/step_prior_input_binding",
        ),
        step_alignment_output_binding=require_identifier(
            document["step_alignment_output_binding"],
            "/alignment/step_alignment_output_binding",
        ),
        prior_epsilon=require_float(
            document["prior_epsilon"], "/alignment/prior_epsilon"
        ),
        initial_attended=require_int(
            document["initial_attended"], "/alignment/initial_attended"
        ),
        ignored_terminal_tokens=require_int(
            document["ignored_terminal_tokens"],
            "/alignment/ignored_terminal_tokens",
        ),
        short_text_no_prior_max_tokens=require_int(
            document["short_text_no_prior_max_tokens"],
            "/alignment/short_text_no_prior_max_tokens",
        ),
        lookahead=require_int(document["lookahead"], "/alignment/lookahead"),
        sink_threshold=require_int(
            document["sink_threshold"], "/alignment/sink_threshold"
        ),
        source_position_policy=require_identifier(
            document["source_position_policy"],
            "/alignment/source_position_policy",
        ),
    )


def parse_rng(value: JsonValue) -> RngSpec:
    keys = (
        "algorithm",
        "seed_bits",
        "counter_bits",
        "state_location",
        "ownership",
        "deterministic",
    )
    document = require_object(value, keys, "/sampling/rng")
    return RngSpec(
        algorithm=require_identifier(
            document["algorithm"], "/sampling/rng/algorithm"
        ),
        seed_bits=require_int(
            document["seed_bits"], "/sampling/rng/seed_bits", 1, 64
        ),
        counter_bits=require_int(
            document["counter_bits"], "/sampling/rng/counter_bits", 1, 64
        ),
        state_location=require_identifier(
            document["state_location"], "/sampling/rng/state_location"
        ),
        ownership=require_identifier(
            document["ownership"], "/sampling/rng/ownership"
        ),
        deterministic=require_bool(
            document["deterministic"], "/sampling/rng/deterministic"
        ),
    )


def parse_sampling(value: JsonValue) -> SamplingSpec:
    keys = (
        "algorithm",
        "top_k",
        "temperature",
        "eos_token_id",
        "forbidden_token_ids",
        "invalid_distribution_policy",
        "next_embedding_location",
        "rng",
    )
    document = require_object(value, keys, "/sampling")
    eos_token_id = require_int(
        document["eos_token_id"], "/sampling/eos_token_id"
    )
    if eos_token_id != LOCAL_AR_AUDIO_EOS_ID:
        raise PackageError(
            "/sampling/eos_token_id: canonical Sofia v1 sampling requires "
            f"AUDIO_EOS={LOCAL_AR_AUDIO_EOS_ID}"
        )
    forbidden_token_ids = require_int_tuple(
        document["forbidden_token_ids"],
        "/sampling/forbidden_token_ids",
    )
    if forbidden_token_ids != LOCAL_AR_STATIC_FORBIDDEN_TOKEN_IDS:
        raise PackageError(
            "/sampling/forbidden_token_ids: canonical Sofia v1 sampling "
            "must permit codec IDs 0..2015, reserve AUDIO_EOS=2017, and "
            "forbid exactly [2016,2018,2019,2020,2021,2022,2023]"
        )
    return SamplingSpec(
        algorithm=require_identifier(
            document["algorithm"], "/sampling/algorithm"
        ),
        top_k=require_int(document["top_k"], "/sampling/top_k", 1),
        temperature=require_float(
            document["temperature"], "/sampling/temperature"
        ),
        eos_token_id=eos_token_id,
        forbidden_token_ids=forbidden_token_ids,
        invalid_distribution_policy=require_identifier(
            document["invalid_distribution_policy"],
            "/sampling/invalid_distribution_policy",
        ),
        next_embedding_location=require_identifier(
            document["next_embedding_location"],
            "/sampling/next_embedding_location",
        ),
        rng=parse_rng(document["rng"]),
    )


def parse_position_embedding(value: JsonValue) -> PositionEmbeddingSpec:
    keys = (
        "kind",
        "positions",
        "source_shape",
        "dtype",
        "source_table_sha256",
    )
    document = require_object(value, keys, "/local_ar/position_embedding")
    source_table_sha256 = require_sha256(
        document["source_table_sha256"],
        "/local_ar/position_embedding/source_table_sha256",
    )
    if source_table_sha256 != SOFIA_LOCAL_AR_POSITION_TABLE_SHA256:
        raise PackageError(
            "/local_ar/position_embedding/source_table_sha256: canonical "
            "Sofia v1 sampling requires the accepted [18,768] BF16 table "
            f"{SOFIA_LOCAL_AR_POSITION_TABLE_SHA256}"
        )
    return PositionEmbeddingSpec(
        kind=require_identifier(
            document["kind"], "/local_ar/position_embedding/kind"
        ),
        positions=require_int_tuple(
            document["positions"], "/local_ar/position_embedding/positions"
        ),
        source_shape=require_int_tuple(
            document["source_shape"],
            "/local_ar/position_embedding/source_shape",
            1,
        ),
        dtype=require_identifier(
            document["dtype"], "/local_ar/position_embedding/dtype"
        ),
        source_table_sha256=source_table_sha256,
    )


def parse_local_ar(value: JsonValue) -> LocalArSpec:
    keys = (
        "engine_name",
        "execution",
        "iterations",
        "positions",
        "position_embedding",
        "codebooks_per_frame",
        "frames_per_decoder_step",
        "sampling_plugin_name",
        "invalid_rows_encoding",
        "no_eos_frame_index",
    )
    document = require_object(value, keys, "/local_ar")
    no_eos = document["no_eos_frame_index"]
    if isinstance(no_eos, bool) or not isinstance(no_eos, int):
        raise PackageError("/local_ar/no_eos_frame_index: expected an integer")
    return LocalArSpec(
        engine_name=require_identifier(
            document["engine_name"], "/local_ar/engine_name"
        ),
        execution=require_identifier(
            document["execution"], "/local_ar/execution"
        ),
        iterations=require_int(
            document["iterations"], "/local_ar/iterations", 1
        ),
        positions=require_int_tuple(
            document["positions"], "/local_ar/positions"
        ),
        position_embedding=parse_position_embedding(
            document["position_embedding"]
        ),
        codebooks_per_frame=require_int(
            document["codebooks_per_frame"],
            "/local_ar/codebooks_per_frame",
            1,
        ),
        frames_per_decoder_step=require_int(
            document["frames_per_decoder_step"],
            "/local_ar/frames_per_decoder_step",
            1,
        ),
        sampling_plugin_name=require_identifier(
            document["sampling_plugin_name"],
            "/local_ar/sampling_plugin_name",
        ),
        invalid_rows_encoding=require_identifier(
            document["invalid_rows_encoding"],
            "/local_ar/invalid_rows_encoding",
        ),
        no_eos_frame_index=no_eos,
    )


def parse_codec(value: JsonValue) -> CodecSpec:
    keys = (
        "initial_engine_name",
        "steady_engine_name",
        "tail_engine_name",
        "sample_rate_hz",
        "hop_length_samples",
        "channels",
        "pcm_format",
        "stateful",
        "initial_frames",
        "steady_frames",
        "tail_min_frames",
        "tail_max_frames",
        "eos_frame_is_audio",
        "zero_frame_finalization",
    )
    document = require_object(value, keys, "/codec")
    eos_frame_is_audio = require_bool(
        document["eos_frame_is_audio"], "/codec/eos_frame_is_audio"
    )
    if eos_frame_is_audio:
        raise PackageError(
            "/codec/eos_frame_is_audio: AUDIO_EOS is a control token, "
            "not an audio frame"
        )
    zero_frame_finalization = require_identifier(
        document["zero_frame_finalization"],
        "/codec/zero_frame_finalization",
    )
    if zero_frame_finalization != ZERO_FRAME_FINALIZATION:
        raise PackageError(
            "/codec/zero_frame_finalization: residual zero frames must emit "
            "a FINAL control marker without invoking NanoCodec"
        )
    return CodecSpec(
        initial_engine_name=require_identifier(
            document["initial_engine_name"], "/codec/initial_engine_name"
        ),
        steady_engine_name=require_identifier(
            document["steady_engine_name"], "/codec/steady_engine_name"
        ),
        tail_engine_name=require_identifier(
            document["tail_engine_name"], "/codec/tail_engine_name"
        ),
        sample_rate_hz=require_int(
            document["sample_rate_hz"], "/codec/sample_rate_hz", 1
        ),
        hop_length_samples=require_int(
            document["hop_length_samples"], "/codec/hop_length_samples", 1
        ),
        channels=require_int(document["channels"], "/codec/channels", 1),
        pcm_format=require_identifier(
            document["pcm_format"], "/codec/pcm_format"
        ),
        stateful=require_bool(document["stateful"], "/codec/stateful"),
        initial_frames=require_int(
            document["initial_frames"], "/codec/initial_frames", 1
        ),
        steady_frames=require_int(
            document["steady_frames"], "/codec/steady_frames", 1
        ),
        tail_min_frames=require_int(
            document["tail_min_frames"], "/codec/tail_min_frames", 1
        ),
        tail_max_frames=require_int(
            document["tail_max_frames"], "/codec/tail_max_frames", 1
        ),
        eos_frame_is_audio=eos_frame_is_audio,
        zero_frame_finalization=zero_frame_finalization,
    )


def parse_limits(value: JsonValue) -> LimitsSpec:
    keys = (
        "maximum_text_tokens",
        "maximum_decoder_steps",
        "maximum_audio_frames",
        "maximum_sessions",
        "maximum_concurrent_requests",
        "pcm_ring_capacity_frames",
        "maximum_workspace_bytes",
        "maximum_device_memory_bytes",
    )
    document = require_object(value, keys, "/limits")
    return LimitsSpec(
        maximum_text_tokens=require_int(
            document["maximum_text_tokens"], "/limits/maximum_text_tokens", 1
        ),
        maximum_decoder_steps=require_int(
            document["maximum_decoder_steps"],
            "/limits/maximum_decoder_steps",
            1,
        ),
        maximum_audio_frames=require_int(
            document["maximum_audio_frames"],
            "/limits/maximum_audio_frames",
            1,
        ),
        maximum_sessions=require_int(
            document["maximum_sessions"], "/limits/maximum_sessions", 1
        ),
        maximum_concurrent_requests=require_int(
            document["maximum_concurrent_requests"],
            "/limits/maximum_concurrent_requests",
            1,
        ),
        pcm_ring_capacity_frames=require_int(
            document["pcm_ring_capacity_frames"],
            "/limits/pcm_ring_capacity_frames",
            1,
        ),
        maximum_workspace_bytes=require_int(
            document["maximum_workspace_bytes"],
            "/limits/maximum_workspace_bytes",
            1,
        ),
        maximum_device_memory_bytes=require_int(
            document["maximum_device_memory_bytes"],
            "/limits/maximum_device_memory_bytes",
            1,
        ),
    )


def parse_destinations(value: JsonValue) -> Destinations:
    keys = (
        "source_model_acceptance_receipt",
        "export_receipt",
        "tokenizer_identity_receipt",
        "plugin_build_receipt",
        "plugin",
        "golden_fixture",
        "golden_receipt",
        "licenses",
        "engines",
    )
    document = require_object(value, keys, "/destinations")
    licenses_value = document["licenses"]
    if not isinstance(licenses_value, list):
        raise PackageError("/destinations/licenses: expected an array")
    licenses: list[LicenseDestination] = []
    for index, value_item in enumerate(licenses_value):
        path = f"/destinations/licenses/{index}"
        item = require_object(value_item, ("role", "path"), path)
        role = require_identifier(item["role"], path + "/role")
        if role not in LICENSE_ROLES:
            raise PackageError(f"{path}/role: unsupported license role {role}")
        licenses.append(
            LicenseDestination(
                role=role,
                path=require_relative_path(item["path"], path + "/path"),
            )
        )
    license_roles = tuple(license.role for license in licenses)
    if license_roles != LICENSE_ROLES:
        raise PackageError(
            "/destinations/licenses: expected the canonical eight-role order"
        )
    engines_value = document["engines"]
    if not isinstance(engines_value, list):
        raise PackageError("/destinations/engines: expected an array")
    engines: list[EngineDestination] = []
    for index, value_item in enumerate(engines_value):
        path = f"/destinations/engines/{index}"
        item = require_object(value_item, ("name", "role", "path"), path)
        role = require_identifier(item["role"], path + "/role")
        if role not in ENGINE_ROLES:
            raise PackageError(f"{path}/role: unsupported engine role {role}")
        name = require_identifier(item["name"], path + "/name")
        expected_name = ENGINE_NAMES_BY_ROLE[role]
        if name != expected_name:
            raise PackageError(
                f"{path}/name: role {role} requires {expected_name}, got {name}"
            )
        engines.append(
            EngineDestination(
                name=name,
                role=role,
                path=require_relative_path(item["path"], path + "/path"),
            )
        )
    roles = tuple(engine.role for engine in engines)
    if len(engines) != len(ENGINE_ROLES) or set(roles) != set(ENGINE_ROLES):
        raise PackageError(
            "/destinations/engines: exactly one destination per v1 role is required"
        )
    if len({engine.name for engine in engines}) != len(engines):
        raise PackageError("/destinations/engines: engine names must be unique")
    result = Destinations(
        source_model_acceptance_receipt=require_relative_path(
            document["source_model_acceptance_receipt"],
            "/destinations/source_model_acceptance_receipt",
        ),
        export_receipt=require_relative_path(
            document["export_receipt"], "/destinations/export_receipt"
        ),
        tokenizer_identity_receipt=require_relative_path(
            document["tokenizer_identity_receipt"],
            "/destinations/tokenizer_identity_receipt",
        ),
        plugin_build_receipt=require_relative_path(
            document["plugin_build_receipt"],
            "/destinations/plugin_build_receipt",
        ),
        plugin=require_relative_path(
            document["plugin"], "/destinations/plugin"
        ),
        golden_fixture=require_relative_path(
            document["golden_fixture"], "/destinations/golden_fixture"
        ),
        golden_receipt=require_relative_path(
            document["golden_receipt"], "/destinations/golden_receipt"
        ),
        licenses=tuple(licenses),
        engines=tuple(engines),
    )
    all_paths = (
        result.source_model_acceptance_receipt,
        result.export_receipt,
        result.tokenizer_identity_receipt,
        result.plugin_build_receipt,
        result.plugin,
        result.golden_fixture,
        result.golden_receipt,
        *(license.path for license in result.licenses),
        *(engine.path for engine in result.engines),
    )
    reserved = {MANIFEST_NAME, MANIFEST_DIGEST_NAME}
    if len(set(all_paths)) != len(all_paths):
        raise PackageError("/destinations: artifact paths must be unique")
    for candidate in all_paths:
        prefix = PurePosixPath(candidate)
        for other in all_paths:
            if candidate != other and prefix in PurePosixPath(other).parents:
                raise PackageError(
                    "/destinations: an artifact path cannot be another "
                    "artifact's parent directory"
                )
    for candidate in all_paths:
        pure = PurePosixPath(candidate)
        if candidate in reserved or any(
            PurePosixPath(path) in pure.parents for path in reserved
        ):
            raise PackageError("/destinations: manifest paths are reserved")
    return result


def parse_package_spec(value: JsonValue) -> PackageSpec:
    keys = (
        "schema_version",
        "bundle_id",
        "created_at_utc",
        "source_model",
        "export",
        "tokenizer",
        "plugin",
        "classifier_free_guidance",
        "kv_cache",
        "alignment",
        "sampling",
        "local_ar",
        "codec",
        "limits",
        "destinations",
    )
    document = require_object(value, keys, "/")
    schema_version = require_int(
        document["schema_version"], "/schema_version", 1, (1 << 32) - 1
    )
    if schema_version != SCHEMA_VERSION:
        raise PackageError(
            f"/schema_version: expected {SCHEMA_VERSION}, got {schema_version}"
        )
    return PackageSpec(
        schema_version=schema_version,
        bundle_id=require_identifier(document["bundle_id"], "/bundle_id"),
        created_at_utc=require_timestamp(
            document["created_at_utc"], "/created_at_utc"
        ),
        source_model=parse_source_model(document["source_model"]),
        export=parse_export(document["export"]),
        tokenizer=parse_tokenizer(document["tokenizer"]),
        plugin=parse_plugin(document["plugin"]),
        classifier_free_guidance=parse_classifier_free_guidance(
            document["classifier_free_guidance"]
        ),
        kv_cache=parse_kv_cache(document["kv_cache"]),
        alignment=parse_alignment(document["alignment"]),
        sampling=parse_sampling(document["sampling"]),
        local_ar=parse_local_ar(document["local_ar"]),
        codec=parse_codec(document["codec"]),
        limits=parse_limits(document["limits"]),
        destinations=parse_destinations(document["destinations"]),
    )


def require_regular_non_symlink(path: Path, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise PackageError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(status.st_mode):
        raise PackageError(f"{label} must not be a symbolic link: {path}")
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as error:
        raise PackageError(f"{label} is missing: {path}") from error
    if absolute != resolved:
        raise PackageError(
            f"{label} must not contain symbolic links or '..': {path}"
        )
    if not stat.S_ISREG(status.st_mode):
        raise PackageError(f"{label} must be a regular file: {path}")
    if status.st_size <= 0:
        raise PackageError(f"{label} must be non-empty: {path}")
    return status


def inventory_problems(paths: InputPaths) -> tuple[str, ...]:
    entries = (
        ("package specification", paths.specification, False),
        (
            "source-model acceptance receipt",
            paths.source_model_acceptance_receipt,
            False,
        ),
        ("export receipt", paths.export_receipt, False),
        ("tokenizer identity receipt", paths.tokenizer_identity_receipt, False),
        ("plugin build receipt", paths.plugin_build_receipt, False),
        ("plugin", paths.plugin, False),
        ("golden fixture", paths.golden_fixture, False),
        ("golden receipt", paths.golden_receipt, False),
        ("manifest validator", paths.manifest_validator, True),
        ("bundle validator", paths.bundle_validator, True),
        *(
            (f"{role} license artifact", path, False)
            for role, path in paths.licenses
        ),
        *(
            (f"{role} plan", path, False)
            for role, path in paths.engines
        ),
    )
    problems: list[str] = []
    for label, path, require_executable in entries:
        try:
            status = path.lstat()
        except FileNotFoundError:
            problems.append(f"MISSING {label}: {path}")
            continue
        if stat.S_ISLNK(status.st_mode):
            problems.append(f"SYMLINK_FORBIDDEN {label}: {path}")
        elif not stat.S_ISREG(status.st_mode):
            problems.append(f"NOT_REGULAR {label}: {path}")
        elif status.st_size <= 0:
            problems.append(f"EMPTY {label}: {path}")
        elif path.absolute() != path.absolute().resolve(strict=True):
            problems.append(
                f"SYMLINK_OR_DOTDOT_FORBIDDEN {label}: {path}"
            )
        elif require_executable and not os.access(path, os.X_OK):
            problems.append(f"NOT_EXECUTABLE {label}: {path}")
    return tuple(problems)


def require_member(document: JsonObject, key: str, path: str) -> JsonValue:
    if key not in document:
        raise PackageError(f"{path}/{key}: missing required field")
    return document[key]


def validate_source_model_receipt(
    path: Path,
    source_model: SourceModelSpec,
) -> None:
    value = load_json_file(path)
    top = require_object(
        value,
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
    if require_int(top["schema_version"], "/schema_version", 1, 2) != 2:
        raise PackageError("/schema_version: expected source receipt v2")
    if require_identifier(top["artifact_role"], "/artifact_role") != (
        "source_model_acceptance"
    ):
        raise PackageError(
            "/artifact_role: expected source_model_acceptance"
        )
    if require_identifier(top["status"], "/status") != "accepted":
        raise PackageError("/status: source-model receipt is not accepted")
    require_timestamp(top["created_at_utc"], "/created_at_utc")

    model = require_object(
        top["source_model"],
        (
            "model_id",
            "version",
            "revision",
            "file_name",
            "sha256",
            "size_bytes",
            "license",
        ),
        "/source_model",
    )
    if (
        require_string(model["model_id"], "/source_model/model_id")
        != source_model.model_id
    ):
        raise PackageError(
            "/source_model/model_id: receipt targets a different model"
        )
    if (
        require_string(model["version"], "/source_model/version")
        != source_model.version
    ):
        raise PackageError(
            "/source_model/version: receipt targets a different model version"
        )
    if (
        require_git_sha1(model["revision"], "/source_model/revision")
        != source_model.revision
    ):
        raise PackageError(
            "/source_model/revision: receipt targets a different immutable "
            "model revision"
        )
    if require_string(
        model["file_name"], "/source_model/file_name"
    ) != "magpie_tts_multilingual_357m.nemo":
        raise PackageError("/source_model/file_name: unexpected model file")
    if (
        require_sha256(model["sha256"], "/source_model/sha256")
        != source_model.source_sha256
    ):
        raise PackageError(
            "/source_model/sha256: receipt does not match "
            "source_model.source_sha256"
        )
    require_int(model["size_bytes"], "/source_model/size_bytes", 1, (1 << 63) - 1)
    license_document = require_object(
        model["license"],
        (
            "name",
            "version",
            "document_file_name",
            "document_sha256",
            "required_notice_file_name",
            "required_notice_sha256",
        ),
        "/source_model/license",
    )
    expected_license: tuple[tuple[str, str], ...] = (
        ("name", NVIDIA_OPEN_MODEL_LICENSE_NAME),
        ("version", NVIDIA_OPEN_MODEL_LICENSE_VERSION),
        ("document_file_name", NVIDIA_OPEN_MODEL_LICENSE_FILE_NAME),
        ("document_sha256", NVIDIA_OPEN_MODEL_LICENSE_SHA256),
        ("required_notice_file_name", NVIDIA_MODEL_NOTICE_FILE_NAME),
        ("required_notice_sha256", NVIDIA_MODEL_NOTICE_SHA256),
    )
    for field, expected in expected_license:
        actual = require_string(
            license_document[field],
            f"/source_model/license/{field}",
        )
        if field.endswith("sha256"):
            actual = require_sha256(
                license_document[field],
                f"/source_model/license/{field}",
            )
        if actual != expected:
            raise PackageError(
                f"/source_model/license/{field}: unexpected value"
            )

    source = require_object(
        top["oracle_source"],
        (
            "repository",
            "base_revision",
            "optimized_source_bundle_sha256",
            "files",
        ),
        "/oracle_source",
    )
    if (
        require_string(source["repository"], "/oracle_source/repository")
        != "https://github.com/NVIDIA/NeMo.git"
    ):
        raise PackageError("/oracle_source/repository: unexpected repository")
    require_git_sha1(source["base_revision"], "/oracle_source/base_revision")
    source_files = require_object(
        source["files"],
        ORACLE_SOURCE_PATHS,
        "/oracle_source/files",
    )
    source_bundle = hashlib.sha256()
    for relative in ORACLE_SOURCE_PATHS:
        digest = require_sha256(
            source_files[relative],
            f"/oracle_source/files/{relative}",
        )
        source_bundle.update(relative.encode("utf-8"))
        source_bundle.update(b"\0")
        source_bundle.update(digest.encode("ascii"))
        source_bundle.update(b"\n")
    expected_bundle = require_sha256(
        source["optimized_source_bundle_sha256"],
        "/oracle_source/optimized_source_bundle_sha256",
    )
    if source_bundle.hexdigest() != expected_bundle:
        raise PackageError(
            "/oracle_source/optimized_source_bundle_sha256: "
            "does not authenticate the ordered 11-file source map"
        )

    contract = require_object(
        top["acceptance_contract"],
        (
            "speaker_name",
            "speaker_index",
            "local_ar_seed",
            "sample_rate_hz",
            "samples_per_codec_frame",
            "first_codec_frames",
            "steady_codec_frames",
            "tail_codec_frames_min",
            "tail_codec_frames_max",
        ),
        "/acceptance_contract",
    )
    expected_contract: tuple[tuple[str, JsonValue], ...] = (
        ("speaker_name", "Sofia"),
        ("speaker_index", 4),
        ("sample_rate_hz", 22050),
        ("samples_per_codec_frame", 1024),
        ("first_codec_frames", 4),
        ("steady_codec_frames", 8),
        ("tail_codec_frames_min", 1),
        ("tail_codec_frames_max", 8),
    )
    for field, expected in expected_contract:
        if contract[field] != expected:
            raise PackageError(
                f"/acceptance_contract/{field}: expected {expected}"
            )
    require_int(
        contract["local_ar_seed"],
        "/acceptance_contract/local_ar_seed",
        0,
        (1 << 64) - 1,
    )

    probe = require_object(
        top["probe"],
        (
            "raw_receipt_sha256",
            "raw_receipt_size_bytes",
            "probe_sha256",
            "helper_sha256",
            "case_definition_sha256",
        ),
        "/probe",
    )
    for field in (
        "raw_receipt_sha256",
        "probe_sha256",
        "helper_sha256",
        "case_definition_sha256",
    ):
        require_sha256(probe[field], f"/probe/{field}")
    require_int(
        probe["raw_receipt_size_bytes"],
        "/probe/raw_receipt_size_bytes",
        1,
        MAX_MANIFEST_BYTES,
    )

    runtime = require_object(
        top["runtime_environment"],
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
    for field in (
        "torch_version",
        "torch_cuda_build",
        "cuda_driver_version",
        "gpu_name",
    ):
        require_string(runtime[field], f"/runtime_environment/{field}")
    require_int(
        runtime["cudnn_version"],
        "/runtime_environment/cudnn_version",
        1,
        (1 << 32) - 1,
    )
    if runtime["gpu_compute_capability"] != [11, 0]:
        raise PackageError(
            "/runtime_environment/gpu_compute_capability: expected [11, 0]"
        )

    evidence = require_object(
        top["evidence"],
        ("case_names", "determinism", "aggregate", "gates"),
        "/evidence",
    )
    if evidence["case_names"] != list(SOURCE_ACCEPTANCE_CASE_NAMES):
        raise PackageError("/evidence/case_names: unexpected acceptance cases")
    determinism = evidence["determinism"]
    if not isinstance(determinism, list):
        raise PackageError("/evidence/determinism: expected an array")
    if len(determinism) != len(SOURCE_ACCEPTANCE_CASE_NAMES):
        raise PackageError(
            "/evidence/determinism: expected exactly four records"
        )
    determinism_fields = (
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
    expected_seed = require_int(
        contract["local_ar_seed"],
        "/acceptance_contract/local_ar_seed",
        0,
        (1 << 64) - 1,
    )
    for index, expected_name in enumerate(SOURCE_ACCEPTANCE_CASE_NAMES):
        record_path = f"/evidence/determinism/{index}"
        record = require_object(
            determinism[index],
            determinism_fields,
            record_path,
        )
        if (
            require_string(record["case_name"], f"{record_path}/case_name")
            != expected_name
        ):
            raise PackageError(
                f"{record_path}/case_name: expected {expected_name}"
            )
        if require_int(
            record["local_ar_seed"],
            f"{record_path}/local_ar_seed",
            0,
            (1 << 64) - 1,
        ) != expected_seed:
            raise PackageError(
                f"{record_path}/local_ar_seed: differs from contract"
            )
        first_codes = require_sha256(
            record["first_codes_sha256"],
            f"{record_path}/first_codes_sha256",
        )
        replay_codes = require_sha256(
            record["replay_codes_sha256"],
            f"{record_path}/replay_codes_sha256",
        )
        first_pcm = require_sha256(
            record["first_pcm_f32le_sha256"],
            f"{record_path}/first_pcm_f32le_sha256",
        )
        replay_pcm = require_sha256(
            record["replay_pcm_f32le_sha256"],
            f"{record_path}/replay_pcm_f32le_sha256",
        )
        first_frames = require_int(
            record["first_codec_frame_count"],
            f"{record_path}/first_codec_frame_count",
            1,
            (1 << 64) - 1,
        )
        replay_frames = require_int(
            record["replay_codec_frame_count"],
            f"{record_path}/replay_codec_frame_count",
            1,
            (1 << 64) - 1,
        )
        if (
            require_bool(
                record["codes_exact"],
                f"{record_path}/codes_exact",
            )
            is not True
            or first_codes != replay_codes
            or require_bool(
                record["pcm_exact"],
                f"{record_path}/pcm_exact",
            )
            is not True
            or first_pcm != replay_pcm
            or require_bool(
                record["codec_frame_count_exact"],
                f"{record_path}/codec_frame_count_exact",
            )
            is not True
            or first_frames != replay_frames
            or require_bool(record["passed"], f"{record_path}/passed")
            is not True
        ):
            raise PackageError(
                f"{record_path}: same-seed replay is not exact"
            )
    aggregate = require_object(
        evidence["aggregate"],
        SOURCE_ACCEPTANCE_AGGREGATE_FIELDS,
        "/evidence/aggregate",
    )
    if (
        require_int(
            aggregate["case_count"],
            "/evidence/aggregate/case_count",
            4,
            4,
        )
        != 4
    ):
        raise PackageError("/evidence/aggregate/case_count: expected 4")
    for metric_name in (
        "raw_ttfa_ms",
        "gapless_start_ms",
        "max_positive_playback_lateness_ms",
        "generation_rtf",
        "total_rtf",
    ):
        metric_path = f"/evidence/aggregate/{metric_name}"
        metric = require_object(
            aggregate[metric_name],
            ("median", "p95", "minimum", "maximum"),
            metric_path,
        )
        median = require_float(metric["median"], f"{metric_path}/median")
        p95 = require_float(metric["p95"], f"{metric_path}/p95")
        minimum = require_float(metric["minimum"], f"{metric_path}/minimum")
        maximum = require_float(metric["maximum"], f"{metric_path}/maximum")
        if (
            minimum < 0.0
            or minimum > median
            or median > maximum
            or minimum > p95
            or p95 > maximum
        ):
            raise PackageError(f"{metric_path}: invalid metric ordering")
    for field in (
        "maximum_peak_cuda_allocated_bytes",
        "maximum_peak_cuda_reserved_bytes",
    ):
        require_int(
            aggregate[field],
            f"/evidence/aggregate/{field}",
            1,
            (1 << 64) - 1,
        )
    gates = require_object(
        evidence["gates"],
        (
            *SOURCE_ACCEPTANCE_PASSED_GATES,
            "parakeet_gate_pending",
            "failures",
        ),
        "/evidence/gates",
    )
    for gate in SOURCE_ACCEPTANCE_PASSED_GATES:
        if require_bool(gates[gate], f"/evidence/gates/{gate}") is not True:
            raise PackageError(f"/evidence/gates/{gate}: expected true")
    if require_bool(
        gates["parakeet_gate_pending"],
        "/evidence/gates/parakeet_gate_pending",
    ) is not True:
        raise PackageError(
            "/evidence/gates/parakeet_gate_pending: expected true"
        )
    if gates["failures"] != []:
        raise PackageError("/evidence/gates/failures: expected an empty array")

    def reject_absolute_paths(candidate: JsonValue, pointer: str) -> None:
        if isinstance(candidate, dict):
            for key, child in candidate.items():
                reject_absolute_paths(child, f"{pointer}/{key}")
        elif isinstance(candidate, list):
            for index, child in enumerate(candidate):
                reject_absolute_paths(child, f"{pointer}/{index}")
        elif isinstance(candidate, str) and (
            candidate.startswith("/")
            or candidate.startswith("file:")
            or re.match(r"^[A-Za-z]:[\\/]", candidate) is not None
        ):
            raise PackageError(
                f"{pointer}: source-model receipt contains an absolute path"
            )

    reject_absolute_paths(top, "")


def validate_plugin_build_receipt(
    path: Path,
    plugin: FileArtifact,
) -> PluginBuildReceiptEvidence:
    receipt_payload = read_verified_regular_file(
        path,
        "plugin build receipt",
        MAX_MANIFEST_BYTES,
    )
    value = parse_json_payload(receipt_payload, path)
    document = require_object(
        value,
        (
            "schema_version",
            "status",
            "artifact",
            "source",
            "toolchain",
            "build",
        ),
        "/",
    )
    if (
        require_string(document["schema_version"], "/schema_version")
        != "magpie-tts-rt.plugin-build.v1"
    ):
        raise PackageError("/schema_version: unsupported plugin build receipt")
    if require_identifier(document["status"], "/status") != "recorded":
        raise PackageError("/status: plugin build receipt is not recorded")

    artifact = require_object(
        document["artifact"],
        ("filename", "size_bytes", "sha256", "soname", "needed"),
        "/artifact",
    )
    if (
        require_string(artifact["filename"], "/artifact/filename")
        != EXPECTED_PLUGIN_FILENAME
    ):
        raise PackageError(
            f"/artifact/filename: expected {EXPECTED_PLUGIN_FILENAME}"
        )
    plugin_size_bytes = require_int(
        artifact["size_bytes"],
        "/artifact/size_bytes",
        1,
    )
    plugin_sha256 = require_sha256(
        artifact["sha256"],
        "/artifact/sha256",
    )
    if plugin_size_bytes != plugin.size_bytes or plugin_sha256 != plugin.sha256:
        raise PackageError(
            "/artifact: plugin build receipt does not authenticate the "
            "staged plugin"
        )
    if require_string(artifact["soname"], "/artifact/soname") != (
        EXPECTED_PLUGIN_SONAME
    ):
        raise PackageError("/artifact/soname: unexpected plugin SONAME")
    needed_value = artifact["needed"]
    if not isinstance(needed_value, list) or not needed_value:
        raise PackageError("/artifact/needed: expected a non-empty array")
    needed = tuple(
        require_string(item, f"/artifact/needed/{index}")
        for index, item in enumerate(needed_value)
    )
    if tuple(sorted(set(needed))) != needed:
        raise PackageError(
            "/artifact/needed: dependencies must be sorted and unique"
        )
    if frozenset(needed) != EXPECTED_PLUGIN_NEEDED:
        raise PackageError(
            "/artifact/needed: plugin dependency closure differs from "
            "the fixed allowlist"
        )

    sources_value = document["source"]
    if not isinstance(sources_value, list) or not sources_value:
        raise PackageError("/source: expected a non-empty array")
    sources: list[PluginSourceEvidence] = []
    for index, source_value in enumerate(sources_value):
        source_path = f"/source/{index}"
        source = require_object(
            source_value,
            ("path", "size_bytes", "sha256"),
            source_path,
        )
        sources.append(
            PluginSourceEvidence(
                path=require_relative_path(
                    source["path"],
                    source_path + "/path",
                ),
                size_bytes=require_int(
                    source["size_bytes"],
                    source_path + "/size_bytes",
                    1,
                ),
                sha256=require_sha256(
                    source["sha256"],
                    source_path + "/sha256",
                ),
            )
        )
    source_paths = tuple(source.path for source in sources)
    if source_paths != EXPECTED_PLUGIN_SOURCE_PATHS:
        raise PackageError(
            "/source: expected the exact canonical plugin source inventory"
        )

    toolchain = require_object(
        document["toolchain"],
        (
            "cxx_path",
            "cxx_version",
            "nvcc_path",
            "nvcc_version",
            "linker_path",
            "linker_version",
            "readelf_path",
            "ninja_path",
            "ninja_version",
            "cmake_path",
            "cmake_version",
            "cuda_architecture",
        ),
        "/toolchain",
    )
    for field in (
        "cxx_path",
        "nvcc_path",
        "linker_path",
        "readelf_path",
        "ninja_path",
        "cmake_path",
    ):
        tool_path = require_string(toolchain[field], f"/toolchain/{field}")
        if not PurePosixPath(tool_path).is_absolute():
            raise PackageError(f"/toolchain/{field}: expected an absolute path")
    for field in (
        "cxx_version",
        "nvcc_version",
        "linker_version",
        "ninja_version",
        "cmake_version",
    ):
        require_string(toolchain[field], f"/toolchain/{field}")
    if (
        require_string(
            toolchain["cuda_architecture"],
            "/toolchain/cuda_architecture",
        )
        != "110"
    ):
        raise PackageError(
            "/toolchain/cuda_architecture: expected NVIDIA Thor sm_110"
        )

    build = require_object(
        document["build"],
        (
            "build_type",
            "tf32_policy",
            "cutlass_archive_sha256",
            "compile_command",
            "link_command",
            "cmake_definitions",
        ),
        "/build",
    )
    if require_string(build["build_type"], "/build/build_type") != "Release":
        raise PackageError("/build/build_type: expected Release")
    if require_string(build["tf32_policy"], "/build/tf32_policy") != "disabled":
        raise PackageError("/build/tf32_policy: TF32 must be disabled")
    if require_sha256(
        build["cutlass_archive_sha256"],
        "/build/cutlass_archive_sha256",
    ) != "5288044d2d5e81632ac0c812b6b85c744901a7d3fd11c9119f18f71c3cef5f79":
        raise PackageError("/build/cutlass_archive_sha256: unpinned CUTLASS source")

    compile_command = require_string(
        build["compile_command"],
        "/build/compile_command",
    )
    link_command = require_string(
        build["link_command"],
        "/build/link_command",
    )
    required_compile_fragments = (
        "-c ${SOURCE_ROOT}/plugins/local_ar_plugins.cu",
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "arch=compute_110,code=[compute_110,sm_110]",
        "-Xcompiler=-fPIC",
        "--frandom-seed=magpie_tts_rt_plugins_v1",
        "-Xcompiler=-Wall,-Wextra",
        "-Xcompiler=-Werror",
    )
    required_link_fragments = (
        "-Wl,--no-undefined",
        "-Wl,--strip-all",
        (
            "-Wl,--version-script="
            "${SOURCE_ROOT}/cmake/magpie_tts_rt_plugins.map"
        ),
        "-shared",
        "-Wl,-soname,libmagpie_tts_rt_plugins.so.0",
        f"-o {EXPECTED_PLUGIN_FILENAME}",
    )
    for fragment in required_compile_fragments:
        if fragment not in compile_command:
            raise PackageError(
                "/build/compile_command: missing actual build fragment "
                f"{fragment}"
            )
    for fragment in required_link_fragments:
        if fragment not in link_command:
            raise PackageError(
                "/build/link_command: missing actual build fragment "
                f"{fragment}"
            )
    definitions_value = build["cmake_definitions"]
    expected_definitions = [
            "CMAKE_BUILD_TYPE=Release",
            "CMAKE_CUDA_ARCHITECTURES=110",
            "MAGPIE_TTS_RT_WARNINGS_AS_ERRORS=ON",
    ]
    if definitions_value != expected_definitions:
        raise PackageError(
            "/build/cmake_definitions: unexpected plugin build configuration"
        )

    return PluginBuildReceiptEvidence(
        receipt_sha256=sha256_bytes(receipt_payload),
        receipt_size_bytes=len(receipt_payload),
        plugin_sha256=plugin_sha256,
        plugin_size_bytes=plugin_size_bytes,
        sources=tuple(sources),
    )


def validate_plugin_build_source_tree(
    evidence: PluginBuildReceiptEvidence,
    source_root: Path,
) -> None:
    root_input = source_root.absolute()
    root = root_input.resolve(strict=True)
    if root_input != root or not root.is_dir():
        raise PackageError(
            "plugin source root must be an absolute nonsymlink directory"
        )
    if tuple(source.path for source in evidence.sources) != (
        EXPECTED_PLUGIN_SOURCE_PATHS
    ):
        raise PackageError("plugin source inventory is not canonical")
    for source in evidence.sources:
        path = root.joinpath(*PurePosixPath(source.path).parts)
        payload = read_verified_regular_file(
            path,
            f"plugin source {source.path}",
            MAX_MANIFEST_BYTES,
        )
        if (
            len(payload) != source.size_bytes
            or sha256_bytes(payload) != source.sha256
        ):
            raise PackageError(
                f"plugin source does not match build receipt: {source.path}"
            )


def validate_export_receipt(
    path: Path,
    export: ExportSpec,
    source_model: SourceModelSpec,
) -> ExportReceiptEvidence:
    value = load_json_file(path)
    if not isinstance(value, dict):
        raise PackageError(f"{path}: export receipt must be an object")
    schema_version = require_int(
        require_member(value, "schema_version", ""),
        "/schema_version",
        1,
        (1 << 32) - 1,
    )
    if schema_version != 1:
        raise PackageError(f"{path}: unsupported export receipt schema version")
    artifact_role = require_identifier(
        require_member(value, "artifact_role", ""), "/artifact_role"
    )
    if artifact_role != export.receipt_artifact_role:
        raise PackageError(
            f"{path}: export receipt artifact_role mismatch: {artifact_role}"
        )
    status_value = require_identifier(
        require_member(value, "status", ""), "/status"
    )
    if status_value != "accepted":
        raise PackageError(
            f"{path}: export receipt status is not accepted: {status_value}"
        )
    source_value = require_member(value, "source", "")
    if not isinstance(source_value, dict):
        raise PackageError("/source: expected an object")
    model_sha256 = require_sha256(
        require_member(source_value, "model_sha256", "/source"),
        "/source/model_sha256",
    )
    if model_sha256 != source_model.source_sha256:
        raise PackageError(
            "/source/model_sha256: export receipt targets a different model"
        )
    oracle_lock_sha256 = require_sha256(
        require_member(source_value, "oracle_lock_sha256", "/source"),
        "/source/oracle_lock_sha256",
    )
    return ExportReceiptEvidence(
        oracle_lock_sha256=oracle_lock_sha256
    )


def validate_consolidated_export_receipt(
    path: Path,
    spec: PackageSpec,
    files: dict[str, FileArtifact],
) -> ExportReceiptEvidence:
    evidence = validate_export_receipt(
        path, spec.export, spec.source_model
    )
    value = load_json_file(path)
    top = require_object(
        value,
        (
            "schema_version",
            "artifact_role",
            "status",
            "created_at_utc",
            "source",
            "component_receipts",
            "sequence_receipt_sha256",
            "complete_generation_receipt_sha256",
            "plugin",
            "engines",
            "golden_fixture",
            "golden_receipt",
            "eos_frame_is_audio",
            "zero_frame_finalization",
        ),
        "/",
    )
    if require_bool(
        top["eos_frame_is_audio"], "/eos_frame_is_audio"
    ) != spec.codec.eos_frame_is_audio:
        raise PackageError(
            "/eos_frame_is_audio: consolidated receipt does not match "
            "the package codec contract"
        )
    if require_identifier(
        top["zero_frame_finalization"], "/zero_frame_finalization"
    ) != spec.codec.zero_frame_finalization:
        raise PackageError(
            "/zero_frame_finalization: consolidated receipt does not match "
            "the package codec contract"
        )
    require_timestamp(top["created_at_utc"], "/created_at_utc")
    source = require_object(
        top["source"],
        (
            "model_id",
            "model_version",
            "model_revision",
            "model_sha256",
            "oracle_lock_sha256",
            "canonical_fixture_manifest_sha256",
            "source_model_acceptance_receipt_sha256",
            "tokenizer_identity_sha256",
            "tokenizer_identity_receipt_sha256",
        ),
        "/source",
    )
    source_comparisons = (
        (
            require_string(source["model_id"], "/source/model_id"),
            spec.source_model.model_id,
            "model_id",
        ),
        (
            require_string(
                source["model_version"], "/source/model_version"
            ),
            spec.source_model.version,
            "model_version",
        ),
        (
            require_string(
                source["model_revision"], "/source/model_revision"
            ),
            spec.source_model.revision,
            "model_revision",
        ),
        (
            require_sha256(
                source["model_sha256"], "/source/model_sha256"
            ),
            spec.source_model.source_sha256,
            "model_sha256",
        ),
        (
            require_sha256(
                source["oracle_lock_sha256"],
                "/source/oracle_lock_sha256",
            ),
            evidence.oracle_lock_sha256,
            "oracle_lock_sha256",
        ),
        (
            require_sha256(
                source["source_model_acceptance_receipt_sha256"],
                "/source/source_model_acceptance_receipt_sha256",
            ),
            files["source_model_acceptance_receipt"].sha256,
            "source_model_acceptance_receipt_sha256",
        ),
        (
            require_sha256(
                source["tokenizer_identity_sha256"],
                "/source/tokenizer_identity_sha256",
            ),
            spec.tokenizer.identity_sha256,
            "tokenizer_identity_sha256",
        ),
        (
            require_sha256(
                source["tokenizer_identity_receipt_sha256"],
                "/source/tokenizer_identity_receipt_sha256",
            ),
            files["tokenizer_identity_receipt"].sha256,
            "tokenizer_identity_receipt_sha256",
        ),
    )
    for actual, expected, field in source_comparisons:
        if actual != expected:
            raise PackageError(
                f"/source/{field}: consolidated receipt does not match "
                "the staged bundle input"
            )
    require_sha256(
        source["canonical_fixture_manifest_sha256"],
        "/source/canonical_fixture_manifest_sha256",
    )

    component_value = top["component_receipts"]
    if not isinstance(component_value, list):
        raise PackageError("/component_receipts: expected an array")
    expected_components = (
        ("text_encoder", "text_encoder_plan"),
        ("main_decoder", "main_decoder"),
        ("local_ar", "local_ar_fixed_16"),
        ("nanocodec", "stateful_nanocodec"),
    )
    if len(component_value) != len(expected_components):
        raise PackageError(
            "/component_receipts: expected exactly four canonical entries"
        )
    for index, (value_item, expected) in enumerate(
        zip(component_value, expected_components, strict=True)
    ):
        item_path = f"/component_receipts/{index}"
        item = require_object(
            value_item,
            ("role", "artifact_role", "receipt_sha256"),
            item_path,
        )
        role = require_identifier(item["role"], item_path + "/role")
        artifact_role = require_identifier(
            item["artifact_role"], item_path + "/artifact_role"
        )
        require_sha256(
            item["receipt_sha256"], item_path + "/receipt_sha256"
        )
        if (role, artifact_role) != expected:
            raise PackageError(
                f"{item_path}: expected canonical component {expected}"
            )

    sequence_receipt_sha256 = require_sha256(
        top["sequence_receipt_sha256"],
        "/sequence_receipt_sha256",
    )
    complete_generation_receipt_sha256 = require_sha256(
        top["complete_generation_receipt_sha256"],
        "/complete_generation_receipt_sha256",
    )
    if complete_generation_receipt_sha256 != files["golden_receipt"].sha256:
        raise PackageError(
            "/complete_generation_receipt_sha256: does not match the "
            "staged golden receipt"
        )
    if sequence_receipt_sha256 == complete_generation_receipt_sha256:
        raise PackageError(
            "/sequence_receipt_sha256: sequence and complete-generation "
            "evidence must be distinct receipts"
        )

    def validate_inventory_item(
        value_item: JsonValue,
        path_item: str,
        expected_role: str,
        expected_file: FileArtifact,
    ) -> None:
        item = require_object(
            value_item,
            ("role", "sha256", "size_bytes"),
            path_item,
        )
        role = require_identifier(item["role"], path_item + "/role")
        sha256 = require_sha256(
            item["sha256"], path_item + "/sha256"
        )
        size_bytes = require_int(
            item["size_bytes"], path_item + "/size_bytes", 1
        )
        if role != expected_role:
            raise PackageError(
                f"{path_item}/role: expected {expected_role}, got {role}"
            )
        if (
            sha256 != expected_file.sha256
            or size_bytes != expected_file.size_bytes
        ):
            raise PackageError(
                f"{path_item}: SHA-256/size do not match the staged artifact"
            )

    plugin_record = require_object(
        top["plugin"],
        (
            "role",
            "sha256",
            "size_bytes",
            "build_receipt_sha256",
            "build_receipt_size_bytes",
        ),
        "/plugin",
    )
    plugin_role = require_identifier(plugin_record["role"], "/plugin/role")
    if plugin_role != "runtime_plugin":
        raise PackageError(
            f"/plugin/role: expected runtime_plugin, got {plugin_role}"
        )
    plugin_comparisons = (
        (
            require_sha256(plugin_record["sha256"], "/plugin/sha256"),
            files["plugin"].sha256,
            "sha256",
        ),
        (
            require_int(
                plugin_record["size_bytes"],
                "/plugin/size_bytes",
                1,
            ),
            files["plugin"].size_bytes,
            "size_bytes",
        ),
        (
            require_sha256(
                plugin_record["build_receipt_sha256"],
                "/plugin/build_receipt_sha256",
            ),
            files["plugin_build_receipt"].sha256,
            "build_receipt_sha256",
        ),
        (
            require_int(
                plugin_record["build_receipt_size_bytes"],
                "/plugin/build_receipt_size_bytes",
                1,
            ),
            files["plugin_build_receipt"].size_bytes,
            "build_receipt_size_bytes",
        ),
    )
    for actual, expected, field in plugin_comparisons:
        if actual != expected:
            raise PackageError(
                f"/plugin/{field}: does not match the staged artifact"
            )
    engines_value = top["engines"]
    if not isinstance(engines_value, list):
        raise PackageError("/engines: expected an array")
    if len(engines_value) != len(ENGINE_ROLES):
        raise PackageError("/engines: expected exactly seven entries")
    for index, (item, role) in enumerate(
        zip(engines_value, ENGINE_ROLES, strict=True)
    ):
        validate_inventory_item(
            item, f"/engines/{index}", role, files[role]
        )

    def validate_file_identity(
        value_item: JsonValue,
        path_item: str,
        expected_file: FileArtifact,
    ) -> None:
        item = require_object(
            value_item, ("sha256", "size_bytes"), path_item
        )
        sha256 = require_sha256(
            item["sha256"], path_item + "/sha256"
        )
        size_bytes = require_int(
            item["size_bytes"], path_item + "/size_bytes", 1
        )
        if (
            sha256 != expected_file.sha256
            or size_bytes != expected_file.size_bytes
        ):
            raise PackageError(
                f"{path_item}: SHA-256/size do not match the staged artifact"
            )

    validate_file_identity(
        top["golden_fixture"],
        "/golden_fixture",
        files["golden_fixture"],
    )
    validate_file_identity(
        top["golden_receipt"],
        "/golden_receipt",
        files["golden_receipt"],
    )
    return evidence


def tokenizer_identity_projection(receipt: JsonObject) -> JsonObject:
    frontend_value = require_member(receipt, "frontend_contract", "")
    if not isinstance(frontend_value, dict):
        raise PackageError("/frontend_contract: expected an object")
    japanese_value = require_member(
        frontend_value, "japanese", "/frontend_contract"
    )
    if not isinstance(japanese_value, dict):
        raise PackageError("/frontend_contract/japanese: expected an object")
    token_table_value = require_member(
        japanese_value,
        "token_table",
        "/frontend_contract/japanese",
    )
    if not isinstance(token_table_value, list) or not token_table_value:
        raise PackageError(
            "/frontend_contract/japanese/token_table: expected a non-empty array"
        )
    vocabulary_sha256 = sha256_bytes(canonical_json_bytes(token_table_value))
    declared_vocabulary_sha256 = require_sha256(
        require_member(
            japanese_value,
            "token_table_sha256",
            "/frontend_contract/japanese",
        ),
        "/frontend_contract/japanese/token_table_sha256",
    )
    if vocabulary_sha256 != declared_vocabulary_sha256:
        raise PackageError(
            "/frontend_contract/japanese/token_table_sha256: "
            "receipt vocabulary hash is not canonical"
        )
    kind = require_identifier(
        require_member(
            japanese_value, "tokenizer_name", "/frontend_contract/japanese"
        ),
        "/frontend_contract/japanese/tokenizer_name",
    )
    vocabulary_size = require_int(
        require_member(
            frontend_value, "aggregate_vocabulary_size", "/frontend_contract"
        ),
        "/frontend_contract/aggregate_vocabulary_size",
        1,
        (1 << 32) - 1,
    )
    special_tokens: JsonObject = {
        "bos_token_id": require_int(
            require_member(frontend_value, "bos_token_id", "/frontend_contract"),
            "/frontend_contract/bos_token_id",
        ),
        "eos_token_id": require_int(
            require_member(frontend_value, "eos_token_id", "/frontend_contract"),
            "/frontend_contract/eos_token_id",
        ),
        "japanese_global_pad_token_id": require_int(
            require_member(
                japanese_value,
                "global_pad_token_id",
                "/frontend_contract/japanese",
            ),
            "/frontend_contract/japanese/global_pad_token_id",
        ),
    }
    return {
        "kind": kind,
        "vocabulary_size": vocabulary_size,
        "frontend_contract_sha256": sha256_bytes(
            canonical_json_bytes(frontend_value)
        ),
        "vocabulary_sha256": vocabulary_sha256,
        "special_tokens": special_tokens,
    }


def validate_tokenizer_identity_receipt(
    path: Path,
    tokenizer: TokenizerSpec,
) -> None:
    value = load_json_file(path)
    if not isinstance(value, dict):
        raise PackageError(f"{path}: tokenizer identity receipt must be an object")
    schema_version = require_int(
        require_member(value, "schema_version", ""),
        "/schema_version",
        1,
        (1 << 32) - 1,
    )
    if schema_version != 1:
        raise PackageError(f"{path}: unsupported tokenizer receipt schema")
    projection = tokenizer_identity_projection(value)
    kind = require_string(projection["kind"], "/identity/kind")
    tokenizer_vocabulary_size = require_int(
        projection["vocabulary_size"], "/identity/vocabulary_size", 1
    )
    frontend_contract_value = require_member(value, "frontend_contract", "")
    if not isinstance(frontend_contract_value, dict):
        raise PackageError("/frontend_contract: expected an object")
    frontend_contract = frontend_contract_value
    text_embedding_rows = require_int(
        require_member(
            frontend_contract, "text_embedding_rows", "/frontend_contract"
        ),
        "/frontend_contract/text_embedding_rows",
        1,
        (1 << 31) - 1,
    )
    projected_special_tokens = require_object(
        projection["special_tokens"],
        (
            "bos_token_id",
            "eos_token_id",
            "japanese_global_pad_token_id",
        ),
        "/identity/special_tokens",
    )
    special_tokens = TokenizerSpecialTokensSpec(
        bos_token_id=require_int(
            projected_special_tokens["bos_token_id"],
            "/identity/special_tokens/bos_token_id",
            0,
            (1 << 31) - 1,
        ),
        eos_token_id=require_int(
            projected_special_tokens["eos_token_id"],
            "/identity/special_tokens/eos_token_id",
            0,
            (1 << 31) - 1,
        ),
        japanese_global_pad_token_id=require_int(
            projected_special_tokens["japanese_global_pad_token_id"],
            "/identity/special_tokens/japanese_global_pad_token_id",
            0,
            (1 << 31) - 1,
        ),
    )
    identity_sha256 = sha256_bytes(canonical_json_bytes(projection))
    if kind != tokenizer.kind:
        raise PackageError(
            f"{path}: tokenizer kind mismatch: expected {tokenizer.kind}, got {kind}"
        )
    if tokenizer_vocabulary_size != tokenizer.tokenizer_vocabulary_size:
        raise PackageError(
            f"{path}: tokenizer vocabulary size mismatch: "
            f"expected {tokenizer.tokenizer_vocabulary_size}, "
            f"got {tokenizer_vocabulary_size}"
        )
    if text_embedding_rows != tokenizer.text_embedding_rows:
        raise PackageError(
            f"{path}: text embedding row count mismatch: "
            f"expected {tokenizer.text_embedding_rows}, "
            f"got {text_embedding_rows}"
        )
    if special_tokens != tokenizer.special_tokens:
        raise PackageError(
            f"{path}: authenticated tokenizer special-token IDs do not "
            "match the package specification"
        )
    if identity_sha256 != tokenizer.identity_sha256:
        raise PackageError(
            f"{path}: tokenizer identity mismatch: "
            f"expected {tokenizer.identity_sha256}, got {identity_sha256}"
        )


def parse_golden_receipt(path: Path) -> GoldenReceipt:
    value = load_json_file(path)
    keys = (
        "receipt_version",
        "created_at_utc",
        "normalized_text_sha256",
        "token_ids_sha256",
        "baked_context_sha256",
        "seed",
        "decoder_tokens_sha256",
        "codec_codes_sha256",
        "codec_frame_count",
        "pcm_f32le_sha256",
        "sample_count",
        "initial_frames",
        "steady_frames",
        "tail_min_frames",
        "tail_max_frames",
        "eos_frame_is_audio",
        "zero_frame_finalization",
    )
    document = require_object(value, keys, "/")
    return GoldenReceipt(
        receipt_version=require_int(
            document["receipt_version"], "/receipt_version", 1, 1
        ),
        created_at_utc=require_timestamp(
            document["created_at_utc"], "/created_at_utc"
        ),
        normalized_text_sha256=require_sha256(
            document["normalized_text_sha256"], "/normalized_text_sha256"
        ),
        token_ids_sha256=require_sha256(
            document["token_ids_sha256"], "/token_ids_sha256"
        ),
        baked_context_sha256=require_sha256(
            document["baked_context_sha256"], "/baked_context_sha256"
        ),
        seed=require_int(document["seed"], "/seed"),
        decoder_tokens_sha256=require_sha256(
            document["decoder_tokens_sha256"], "/decoder_tokens_sha256"
        ),
        codec_codes_sha256=require_sha256(
            document["codec_codes_sha256"], "/codec_codes_sha256"
        ),
        codec_frame_count=require_int(
            document["codec_frame_count"], "/codec_frame_count", 1
        ),
        pcm_f32le_sha256=require_sha256(
            document["pcm_f32le_sha256"], "/pcm_f32le_sha256"
        ),
        sample_count=require_int(
            document["sample_count"], "/sample_count", 1
        ),
        initial_frames=require_int(
            document["initial_frames"], "/initial_frames", 1
        ),
        steady_frames=require_int(
            document["steady_frames"], "/steady_frames", 1
        ),
        tail_min_frames=require_int(
            document["tail_min_frames"], "/tail_min_frames", 1
        ),
        tail_max_frames=require_int(
            document["tail_max_frames"], "/tail_max_frames", 1
        ),
        eos_frame_is_audio=require_bool(
            document["eos_frame_is_audio"], "/eos_frame_is_audio"
        ),
        zero_frame_finalization=require_identifier(
            document["zero_frame_finalization"],
            "/zero_frame_finalization",
        ),
    )


def parse_golden_fixture(path: Path) -> GoldenFixture:
    value = load_json_file(path)
    keys = (
        "schema_version",
        "fixture_id",
        "prepared_token_ids",
        "seed",
        "tokenizer_identity_sha256",
        "oracle_lock_sha256",
        "normalized_text_sha256",
        "token_ids_sha256",
        "baked_context_sha256",
        "expected",
    )
    document = require_object(value, keys, "/")
    expected_document = require_object(
        document["expected"],
        (
            "decoder_tokens_sha256",
            "codec_codes_sha256",
            "codec_frame_count",
            "pcm_f32le_sha256",
            "pcm_sample_count",
        ),
        "/expected",
    )
    prepared_value = document["prepared_token_ids"]
    if not isinstance(prepared_value, list) or not prepared_value:
        raise PackageError(
            "/prepared_token_ids: expected a non-empty integer array"
        )
    prepared_token_ids = tuple(
        require_int(
            token,
            f"/prepared_token_ids/{index}",
            0,
            (1 << 31) - 1,
        )
        for index, token in enumerate(prepared_value)
    )
    schema_version = require_int(
        document["schema_version"], "/schema_version", 1, 1
    )
    return GoldenFixture(
        schema_version=schema_version,
        fixture_id=require_identifier(
            document["fixture_id"], "/fixture_id"
        ),
        prepared_token_ids=prepared_token_ids,
        seed=require_int(document["seed"], "/seed", 0, (1 << 32) - 1),
        tokenizer_identity_sha256=require_sha256(
            document["tokenizer_identity_sha256"],
            "/tokenizer_identity_sha256",
        ),
        oracle_lock_sha256=require_sha256(
            document["oracle_lock_sha256"], "/oracle_lock_sha256"
        ),
        normalized_text_sha256=require_sha256(
            document["normalized_text_sha256"], "/normalized_text_sha256"
        ),
        token_ids_sha256=require_sha256(
            document["token_ids_sha256"], "/token_ids_sha256"
        ),
        baked_context_sha256=require_sha256(
            document["baked_context_sha256"], "/baked_context_sha256"
        ),
        expected=GoldenFixtureExpected(
            decoder_tokens_sha256=require_sha256(
                expected_document["decoder_tokens_sha256"],
                "/expected/decoder_tokens_sha256",
            ),
            codec_codes_sha256=require_sha256(
                expected_document["codec_codes_sha256"],
                "/expected/codec_codes_sha256",
            ),
            codec_frame_count=require_int(
                expected_document["codec_frame_count"],
                "/expected/codec_frame_count",
                1,
            ),
            pcm_f32le_sha256=require_sha256(
                expected_document["pcm_f32le_sha256"],
                "/expected/pcm_f32le_sha256",
            ),
            pcm_sample_count=require_int(
                expected_document["pcm_sample_count"],
                "/expected/pcm_sample_count",
                1,
            ),
        ),
    )


def prepared_token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    payload = b"".join(
        token_id.to_bytes(4, byteorder="little", signed=True)
        for token_id in token_ids
    )
    return sha256_bytes(payload)


def validate_golden_fixture(
    fixture: GoldenFixture,
    receipt: GoldenReceipt,
    spec: PackageSpec,
    export_evidence: ExportReceiptEvidence,
) -> None:
    if len(fixture.prepared_token_ids) > spec.limits.maximum_text_tokens:
        raise PackageError(
            "golden fixture token count exceeds limits.maximum_text_tokens"
        )
    computed_token_sha256 = prepared_token_ids_sha256(
        fixture.prepared_token_ids
    )
    if computed_token_sha256 != fixture.token_ids_sha256:
        raise PackageError(
            "golden fixture token_ids_sha256 does not match the exact "
            "little-endian INT32 token bytes"
        )
    if (
        fixture.prepared_token_ids[-1]
        != spec.tokenizer.special_tokens.eos_token_id
    ):
        raise PackageError(
            "/prepared_token_ids: the final prepared token must be the "
            "authenticated EOS identifier"
        )
    for index, token_id in enumerate(fixture.prepared_token_ids[:-1]):
        if token_id >= spec.tokenizer.tokenizer_vocabulary_size:
            raise PackageError(
                f"/prepared_token_ids/{index}: non-final prepared tokens "
                "must be normal tokenizer rows"
            )
    comparisons = (
        (
            fixture.tokenizer_identity_sha256,
            spec.tokenizer.identity_sha256,
            "tokenizer identity",
        ),
        (
            fixture.oracle_lock_sha256,
            export_evidence.oracle_lock_sha256,
            "oracle lock",
        ),
        (
            fixture.normalized_text_sha256,
            receipt.normalized_text_sha256,
            "normalized text",
        ),
        (
            fixture.token_ids_sha256,
            receipt.token_ids_sha256,
            "token IDs",
        ),
        (
            fixture.baked_context_sha256,
            receipt.baked_context_sha256,
            "baked context",
        ),
        (
            fixture.expected.decoder_tokens_sha256,
            receipt.decoder_tokens_sha256,
            "decoder tokens",
        ),
        (
            fixture.expected.codec_codes_sha256,
            receipt.codec_codes_sha256,
            "codec codes",
        ),
        (
            fixture.expected.pcm_f32le_sha256,
            receipt.pcm_f32le_sha256,
            "PCM",
        ),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            raise PackageError(
                f"golden fixture and receipt {label} SHA-256 differ"
            )
    if fixture.seed != receipt.seed:
        raise PackageError("golden fixture and receipt seeds differ")
    if fixture.expected.codec_frame_count != receipt.codec_frame_count:
        raise PackageError(
            "golden fixture and receipt codec frame counts differ"
        )
    if fixture.expected.codec_frame_count > spec.limits.maximum_audio_frames:
        raise PackageError(
            "golden fixture codec frame count exceeds "
            "limits.maximum_audio_frames"
        )
    if (
        receipt.eos_frame_is_audio != spec.codec.eos_frame_is_audio
        or receipt.zero_frame_finalization
        != spec.codec.zero_frame_finalization
    ):
        raise PackageError(
            "golden receipt and codec stream contract differ"
        )
    if fixture.expected.pcm_sample_count != receipt.sample_count:
        raise PackageError(
            "golden fixture and receipt PCM sample counts differ"
        )
    expected_sample_count = (
        fixture.expected.codec_frame_count * spec.codec.hop_length_samples
    )
    if expected_sample_count != fixture.expected.pcm_sample_count:
        raise PackageError(
            "golden fixture PCM sample count does not equal codec frame "
            "count times hop length"
        )


def parse_plugin_dynamic_section(output: str) -> PluginDynamicContract:
    needed: list[str] = []
    sonames: list[str] = []
    forbidden: set[str] = set()
    for line in output.splitlines():
        match = re.search(r"\(([A-Z0-9_]+)\).*?\[([^\]]*)\]", line)
        if match is None:
            continue
        tag, value = match.groups()
        if tag == "NEEDED":
            needed.append(value)
        elif tag == "SONAME":
            sonames.append(value)
        elif tag in FORBIDDEN_PLUGIN_DYNAMIC_TAGS:
            forbidden.add(tag)

    if forbidden:
        raise PackageError(
            "plugin dynamic section contains forbidden tags: "
            + ", ".join(sorted(forbidden))
        )
    if len(sonames) != 1 or sonames[0] != EXPECTED_PLUGIN_SONAME:
        raise PackageError(
            "plugin SONAME must be exactly "
            f"{EXPECTED_PLUGIN_SONAME}, got {sonames}"
        )
    if len(needed) != len(set(needed)):
        raise PackageError("plugin dynamic section contains duplicate NEEDED entries")
    actual_needed = frozenset(needed)
    if actual_needed != EXPECTED_PLUGIN_NEEDED:
        missing = sorted(EXPECTED_PLUGIN_NEEDED - actual_needed)
        extra = sorted(actual_needed - EXPECTED_PLUGIN_NEEDED)
        raise PackageError(
            "plugin dependency closure differs from the fixed allowlist: "
            f"missing={missing}, extra={extra}"
        )
    return PluginDynamicContract(
        soname=sonames[0],
        needed=actual_needed,
    )


def inspect_plugin_dynamic_contract(path: Path) -> PluginDynamicContract:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise PackageError(
            "readelf is required to authenticate the plugin dependency contract"
        )
    completed = subprocess.run(
        [readelf, "-dW", "--", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PackageError(
            f"readelf failed while inspecting plugin dynamic section: {detail}"
        )
    return parse_plugin_dynamic_section(completed.stdout)


def load_and_register_plugin(path: Path) -> PluginIdentity:
    require_regular_non_symlink(path, "plugin")
    inspect_plugin_dynamic_contract(path)
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    get_api = library.mtt_plugin_get_api_v1
    get_api.argtypes = [ctypes.POINTER(PluginApi)]
    get_api.restype = ctypes.c_int32
    api = PluginApi()
    api.struct_size = ctypes.sizeof(PluginApi)
    api.abi_version = PLUGIN_ABI_VERSION
    status = int(get_api(ctypes.byref(api)))
    if status != PLUGIN_STATUS_OK:
        raise PackageError(f"plugin API query failed with status {status}")
    if api.struct_size != ctypes.sizeof(PluginApi):
        raise PackageError("plugin returned an unexpected API structure size")
    if api.abi_version != PLUGIN_ABI_VERSION:
        raise PackageError("plugin returned an unexpected ABI version")
    if api.creator_count != PLUGIN_CREATOR_COUNT:
        raise PackageError("plugin returned an unexpected creator count")
    if api.reserved_0 != 0 or any(value != 0 for value in api.reserved):
        raise PackageError("plugin API reserved fields are nonzero")
    creators: list[tuple[str, str, str]] = []
    for index in range(api.creator_count):
        creator = api.creators[index]
        if creator.struct_size != ctypes.sizeof(PluginCreator):
            raise PackageError(f"plugin creator {index} has a wrong struct_size")
        if creator.abi_version != PLUGIN_ABI_VERSION:
            raise PackageError(f"plugin creator {index} has a wrong ABI version")
        try:
            record = (
                bytes(creator.name).split(b"\0", 1)[0].decode("ascii"),
                bytes(creator.version).split(b"\0", 1)[0].decode("ascii"),
                bytes(creator.plugin_namespace)
                .split(b"\0", 1)[0]
                .decode("ascii"),
            )
        except UnicodeDecodeError as error:
            raise PackageError(
                f"plugin creator {index} identity is not ASCII"
            ) from error
        creators.append(record)
    if tuple(creators) != EXPECTED_PLUGIN_CREATORS:
        raise PackageError(
            f"plugin creator identity mismatch: expected "
            f"{EXPECTED_PLUGIN_CREATORS}, got {tuple(creators)}"
        )
    if not api.register_plugins:
        raise PackageError("plugin register function is null")
    registration_status = int(api.register_plugins())
    if registration_status not in (
        PLUGIN_STATUS_OK,
        PLUGIN_STATUS_ALREADY_REGISTERED,
    ):
        raise PackageError(
            f"plugin registration failed with status {registration_status}"
        )
    return PluginIdentity(
        abi_version=api.abi_version,
        creators=tuple(creators),
        library=library,
    )


def parse_os_release(path: Path = Path("/usr/lib/os-release")) -> tuple[str, str]:
    payload = read_verified_regular_file(path, "OS release", 64 * 1024)
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PackageError(f"{path}: os-release is not UTF-8") from error
    fields: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise PackageError(
                f"{path}:{line_number}: malformed os-release entry"
            )
        key, raw_value = line.split("=", 1)
        if not key or key in fields:
            raise PackageError(
                f"{path}:{line_number}: duplicate or empty os-release key"
            )
        if raw_value.startswith('"'):
            if not raw_value.endswith('"') or len(raw_value) < 2:
                raise PackageError(
                    f"{path}:{line_number}: malformed quoted os-release value"
                )
            value = raw_value[1:-1]
        else:
            value = raw_value
        fields[key] = value
    if fields.get("ID") != "ubuntu" or fields.get("VERSION_ID") != "24.04":
        raise PackageError(
            "P2 v1 packaging requires exact Ubuntu 24.04 runtime identity"
        )
    return "linux", "ubuntu-24.04"


def require_cuda_success(status: int, operation: str) -> None:
    if status != 0:
        raise PackageError(f"{operation} failed with CUDA status {status}")


def cuda_runtime_version() -> str:
    runtime = ctypes.CDLL("libcudart.so.13")
    version = ctypes.c_int()
    require_cuda_success(
        int(runtime.cudaRuntimeGetVersion(ctypes.byref(version))),
        "cudaRuntimeGetVersion",
    )
    if version.value <= 0:
        raise PackageError("CUDA runtime returned an invalid version")
    major = version.value // 1000
    minor = (version.value % 1000) // 10
    return f"{major}.{minor}"


def cuda_device_identity(device_ordinal: int) -> tuple[str, str]:
    driver = ctypes.CDLL("libcuda.so.1")
    require_cuda_success(int(driver.cuInit(0)), "cuInit")
    device = ctypes.c_int()
    require_cuda_success(
        int(driver.cuDeviceGet(ctypes.byref(device), device_ordinal)),
        "cuDeviceGet",
    )
    name_buffer = ctypes.create_string_buffer(256)
    require_cuda_success(
        int(driver.cuDeviceGetName(name_buffer, len(name_buffer), device)),
        "cuDeviceGetName",
    )
    major = ctypes.c_int()
    minor = ctypes.c_int()
    require_cuda_success(
        int(
            driver.cuDeviceComputeCapability(
                ctypes.byref(major), ctypes.byref(minor), device
            )
        ),
        "cuDeviceComputeCapability",
    )
    try:
        name = name_buffer.value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PackageError("CUDA device name is not UTF-8") from error
    if not name:
        raise PackageError("CUDA device name is empty")
    return name, f"{major.value}.{minor.value}"


def nvidia_driver_version() -> str:
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    status = int(nvml.nvmlInit_v2())
    if status != 0:
        raise PackageError(f"nvmlInit_v2 failed with status {status}")
    try:
        buffer = ctypes.create_string_buffer(80)
        status = int(nvml.nvmlSystemGetDriverVersion(buffer, len(buffer)))
        if status != 0:
            raise PackageError(
                f"nvmlSystemGetDriverVersion failed with status {status}"
            )
        try:
            version = buffer.value.decode("ascii")
        except UnicodeDecodeError as error:
            raise PackageError("NVIDIA driver version is not ASCII") from error
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version) is None:
            raise PackageError(f"invalid NVIDIA driver version {version}")
        return version
    finally:
        shutdown_status = int(nvml.nvmlShutdown())
        if shutdown_status != 0 and sys.exc_info()[0] is None:
            raise PackageError(
                f"nvmlShutdown failed with status {shutdown_status}"
            )


def collect_runtime_fingerprint(
    tensorrt_version: str,
    plugin_abi_version: int,
    device_ordinal: int,
) -> RuntimeFingerprint:
    os_name, os_version = parse_os_release()
    architecture = platform.machine()
    if architecture not in ("aarch64", "x86_64"):
        raise PackageError(f"unsupported architecture {architecture}")
    if sys.byteorder not in ("little", "big"):
        raise PackageError(f"unsupported endianness {sys.byteorder}")
    gpu_name, compute_capability = cuda_device_identity(device_ordinal)
    if gpu_name != "NVIDIA Thor" or compute_capability != "11.0":
        raise PackageError(
            "P2 accepted bundle requires NVIDIA Thor sm_110, got "
            f"{gpu_name} sm_{compute_capability.replace('.', '')}"
        )
    return RuntimeFingerprint(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        endianness=sys.byteorder,
        cuda_version=cuda_runtime_version(),
        tensorrt_version=tensorrt_version,
        driver_version=nvidia_driver_version(),
        gpu_name=gpu_name,
        gpu_compute_capability=compute_capability,
        plugin_abi_version=plugin_abi_version,
    )


def tensor_dtype_name(tensorrt_module, dtype) -> str:
    mapping = {
        tensorrt_module.float32: "fp32",
        tensorrt_module.float16: "fp16",
        tensorrt_module.bfloat16: "bf16",
        tensorrt_module.int64: "int64",
        tensorrt_module.int32: "int32",
        tensorrt_module.int8: "int8",
        tensorrt_module.uint8: "uint8",
        tensorrt_module.bool: "bool",
    }
    result = mapping.get(dtype)
    if result is None:
        raise PackageError(f"TensorRT plan uses unsupported dtype {dtype}")
    return result


def inspect_engine(
    tensorrt_module,
    engine,
    role: str,
) -> InspectedEngine:
    if engine.num_optimization_profiles != 1:
        raise PackageError(
            f"{role}: v1 requires exactly one optimization profile, got "
            f"{engine.num_optimization_profiles}"
        )
    inputs: list[TensorSpec] = []
    outputs: list[TensorSpec] = []
    dynamic_input_names: list[str] = []
    shape_input_names: list[str] = []
    names: set[str] = set()
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if not isinstance(name, str) or not name:
            raise PackageError(f"{role}: TensorRT returned an invalid I/O name")
        if name in names:
            raise PackageError(f"{role}: duplicate TensorRT I/O name {name}")
        names.add(name)
        mode = engine.get_tensor_mode(name)
        if mode not in (
            tensorrt_module.TensorIOMode.INPUT,
            tensorrt_module.TensorIOMode.OUTPUT,
        ):
            raise PackageError(
                f"{role}: unsupported I/O mode for {name}: {mode}"
            )
        location = engine.get_tensor_location(name)
        shape_inference_io = bool(engine.is_shape_inference_io(name))
        is_position_shape_input = (
            role == "main_decoder_step"
            and name == "position"
            and mode == tensorrt_module.TensorIOMode.INPUT
        )
        if is_position_shape_input:
            if (
                location != tensorrt_module.TensorLocation.HOST
                or not shape_inference_io
            ):
                raise PackageError(
                    "main_decoder_step: position must be the authenticated "
                    "HOST shape input"
                )
            location_name = "host"
        elif (
            location != tensorrt_module.TensorLocation.DEVICE
            or shape_inference_io
        ):
            raise PackageError(
                f"{role}: only main_decoder_step/position may be a HOST "
                f"shape input: {name}"
            )
        else:
            location_name = "device"
        shape = tuple(int(dimension) for dimension in engine.get_tensor_shape(name))
        if any(dimension == 0 or dimension < -1 for dimension in shape):
            raise PackageError(
                f"{role}: invalid declared shape for {name}: {shape}"
            )
        dtype = tensor_dtype_name(
            tensorrt_module, engine.get_tensor_dtype(name)
        )
        if is_position_shape_input and (dtype != "int64" or shape != ()):
            raise PackageError(
                "main_decoder_step: position must be a scalar int64 "
                "shape input"
            )
        spec = TensorSpec(
            name=name,
            dtype=dtype,
            shape=shape,
            location=location_name,
            shape_inference_io=shape_inference_io,
        )
        if mode == tensorrt_module.TensorIOMode.INPUT:
            inputs.append(spec)
            if -1 in shape:
                dynamic_input_names.append(name)
            if shape_inference_io:
                shape_input_names.append(name)
        elif mode == tensorrt_module.TensorIOMode.OUTPUT:
            outputs.append(spec)
    if not inputs or not outputs:
        raise PackageError(f"{role}: plan must expose inputs and outputs")
    ranges: list[TensorShapeRange] = []
    for name in dynamic_input_names:
        minimum, optimum, maximum = engine.get_tensor_profile_shape(name, 0)
        shape_range = TensorShapeRange(
            tensor_name=name,
            minimum=tuple(int(value) for value in minimum),
            optimum=tuple(int(value) for value in optimum),
            maximum=tuple(int(value) for value in maximum),
        )
        declared = next(item.shape for item in inputs if item.name == name)
        for label, concrete in (
            ("min", shape_range.minimum),
            ("opt", shape_range.optimum),
            ("max", shape_range.maximum),
        ):
            if len(concrete) != len(declared) or any(value <= 0 for value in concrete):
                raise PackageError(
                    f"{role}: invalid {label} profile shape for {name}: {concrete}"
                )
            for axis, (declared_value, actual_value) in enumerate(
                zip(declared, concrete, strict=True)
            ):
                if declared_value != -1 and declared_value != actual_value:
                    raise PackageError(
                        f"{role}: {name} profile changes static axis {axis}"
                    )
        if not all(
            low <= middle <= high
            for low, middle, high in zip(
                shape_range.minimum,
                shape_range.optimum,
                shape_range.maximum,
                strict=True,
            )
        ):
            raise PackageError(f"{role}: non-monotonic profile for {name}")
        ranges.append(shape_range)
    value_ranges: list[TensorValueRange] = []
    for name in shape_input_names:
        raw_values = engine.get_tensor_profile_values(0, name)
        if raw_values is None or len(raw_values) != 3:
            raise PackageError(
                f"{role}: missing shape-input profile values for {name}"
            )
        value_range = TensorValueRange(
            tensor_name=name,
            minimum=tuple(int(value) for value in raw_values[0]),
            optimum=tuple(int(value) for value in raw_values[1]),
            maximum=tuple(int(value) for value in raw_values[2]),
        )
        declared = next(item.shape for item in inputs if item.name == name)
        value_count = math.prod(declared) if declared else 1
        if any(
            len(values) != value_count
            for values in (
                value_range.minimum,
                value_range.optimum,
                value_range.maximum,
            )
        ):
            raise PackageError(
                f"{role}: invalid profile value count for {name}"
            )
        if not all(
            low <= middle <= high
            for low, middle, high in zip(
                value_range.minimum,
                value_range.optimum,
                value_range.maximum,
                strict=True,
            )
        ):
            raise PackageError(
                f"{role}: non-monotonic shape-input profile for {name}"
            )
        if (
            role != "main_decoder_step"
            or name != "position"
            or value_range.minimum != (218,)
            or value_range.optimum != (342,)
            or value_range.maximum != (466,)
        ):
            raise PackageError(
                f"{role}: unauthenticated shape-input profile for {name}"
            )
        value_ranges.append(value_range)
    return InspectedEngine(
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        profiles=(
            OptimizationProfile(
                name=PROFILE_NAMES_BY_ROLE[role],
                input_shapes=tuple(ranges),
                input_values=tuple(value_ranges),
            ),
        ),
    )


def inspect_plan_files(
    plan_paths: tuple[tuple[str, Path], ...],
) -> tuple[str, dict[str, InspectedEngine]]:
    # TensorRT is an execution-only dependency. Pure parser and identity tests
    # intentionally import this module without loading the GPU runtime.
    import tensorrt

    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    runtime = tensorrt.Runtime(logger)
    if hasattr(runtime, "engine_host_code_allowed"):
        runtime.engine_host_code_allowed = False
    inspected: dict[str, InspectedEngine] = {}
    for role, plan_path in plan_paths:
        require_regular_non_symlink(plan_path, f"{role} plan")
        engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if engine is None:
            raise PackageError(
                f"{role}: TensorRT failed to deserialize {plan_path}"
            )
        inspected[role] = inspect_engine(tensorrt, engine, role)
    if set(inspected) != set(ENGINE_ROLES):
        raise PackageError("plan input set does not contain all seven roles")
    return tensorrt.__version__, inspected


def tensor_map(
    tensors: tuple[TensorSpec, ...],
    engine_role: str,
    mode: str,
) -> dict[str, TensorSpec]:
    result = {tensor.name: tensor for tensor in tensors}
    if len(result) != len(tensors):
        raise PackageError(f"{engine_role}: duplicate {mode} tensor name")
    return result


def require_tensor(
    tensors: dict[str, TensorSpec],
    name: str,
    engine_role: str,
) -> TensorSpec:
    tensor = tensors.get(name)
    if tensor is None:
        raise PackageError(f"{engine_role}: required tensor is missing: {name}")
    return tensor


def derive_kv_cache(
    spec: KvCacheSpec,
    engines: dict[str, InspectedEngine],
) -> JsonObject:
    prefill_outputs = tensor_map(
        engines["main_decoder_prefill"].outputs,
        "main_decoder_prefill",
        "output",
    )
    step_inputs = tensor_map(
        engines["main_decoder_step"].inputs,
        "main_decoder_step",
        "input",
    )
    step_outputs = tensor_map(
        engines["main_decoder_step"].outputs,
        "main_decoder_step",
        "output",
    )
    layer_indices = sorted(
        int(match.group(1))
        for name in prefill_outputs
        if (match := re.fullmatch(r"prefill_self_key_([0-9]+)", name))
        is not None
    )
    if not layer_indices or layer_indices != list(range(len(layer_indices))):
        raise PackageError(
            "main_decoder_prefill: KV layer indices are not contiguous from zero"
        )
    first_key = require_tensor(
        prefill_outputs, "prefill_self_key_0", "main_decoder_prefill"
    )
    if len(first_key.shape) != 4:
        raise PackageError("main_decoder_prefill: self key rank must be four")
    batch, capacity, self_heads, self_head_dimension = first_key.shape
    first_cross = require_tensor(
        prefill_outputs, "prefill_cross_key_0", "main_decoder_prefill"
    )
    if len(first_cross.shape) != 4:
        raise PackageError("main_decoder_prefill: cross key rank must be four")
    cross_batch, cross_text, cross_heads, cross_head_dimension = first_cross.shape
    if batch <= 0 or capacity <= 0 or self_heads <= 0 or self_head_dimension <= 0:
        raise PackageError("main_decoder_prefill: invalid fixed self-cache shape")
    if (
        cross_batch != batch
        or cross_text != -1
        or cross_heads <= 0
        or cross_head_dimension <= 0
    ):
        raise PackageError("main_decoder_prefill: invalid cross-cache shape")
    mask_dtype: str | None = None
    layer_bindings: list[JsonValue] = []
    for index in layer_indices:
        names = {
            "prefill_self_key_output": f"prefill_self_key_{index}",
            "prefill_self_value_output": f"prefill_self_value_{index}",
            "prefill_self_mask_output": f"prefill_self_mask_{index}",
            "prefill_cross_key_output": f"prefill_cross_key_{index}",
            "prefill_cross_value_output": f"prefill_cross_value_{index}",
            "step_self_key_input": f"step_self_key_in_{index}",
            "step_self_value_input": f"step_self_value_in_{index}",
            "step_self_mask_input": f"step_self_mask_in_{index}",
            "step_cross_key_input": f"step_cross_key_in_{index}",
            "step_cross_value_input": f"step_cross_value_in_{index}",
            "step_self_key_output": f"step_self_key_out_{index}",
            "step_self_value_output": f"step_self_value_out_{index}",
            "step_self_mask_output": f"step_self_mask_out_{index}",
        }
        self_key = require_tensor(
            prefill_outputs,
            names["prefill_self_key_output"],
            "main_decoder_prefill",
        )
        self_value = require_tensor(
            prefill_outputs,
            names["prefill_self_value_output"],
            "main_decoder_prefill",
        )
        self_mask = require_tensor(
            prefill_outputs,
            names["prefill_self_mask_output"],
            "main_decoder_prefill",
        )
        cross_key = require_tensor(
            prefill_outputs,
            names["prefill_cross_key_output"],
            "main_decoder_prefill",
        )
        cross_value = require_tensor(
            prefill_outputs,
            names["prefill_cross_value_output"],
            "main_decoder_prefill",
        )
        if (
            self_key.dtype != first_key.dtype
            or self_key.shape != first_key.shape
            or self_value.dtype != self_key.dtype
            or self_value.shape != self_key.shape
        ):
            raise PackageError(
                f"main_decoder_prefill: self-cache mismatch at layer {index}"
            )
        if self_mask.shape != (batch, capacity):
            raise PackageError(
                f"main_decoder_prefill: self-mask shape mismatch at layer {index}"
            )
        if mask_dtype is None:
            mask_dtype = self_mask.dtype
        elif mask_dtype != self_mask.dtype:
            raise PackageError(
                f"main_decoder_prefill: self-mask dtype mismatch at layer {index}"
            )
        if (
            cross_key.dtype != first_key.dtype
            or cross_key.shape != first_cross.shape
            or cross_value.dtype != cross_key.dtype
            or cross_value.shape != cross_key.shape
        ):
            raise PackageError(
                f"main_decoder_prefill: cross-cache mismatch at layer {index}"
            )
        for name in (
            names["step_self_key_input"],
            names["step_self_value_input"],
            names["step_self_mask_input"],
            names["step_cross_key_input"],
            names["step_cross_value_input"],
        ):
            require_tensor(step_inputs, name, "main_decoder_step")
        for name in (
            names["step_self_key_output"],
            names["step_self_value_output"],
            names["step_self_mask_output"],
        ):
            require_tensor(step_outputs, name, "main_decoder_step")
        layer: JsonObject = {"layer_index": index}
        layer.update(names)
        layer_bindings.append(layer)
    if mask_dtype is None:
        raise PackageError("main_decoder_prefill: no self-mask dtype")
    return {
        "layout": spec.layout,
        "dtype": first_key.dtype,
        "mask_dtype": mask_dtype,
        "layers": len(layer_indices),
        "batch_size": batch,
        "self_attention_heads": self_heads,
        "self_attention_head_dimension": self_head_dimension,
        "cross_attention_heads": cross_heads,
        "cross_attention_head_dimension": cross_head_dimension,
        "prefix_length": spec.prefix_length,
        "maximum_generated_steps": spec.maximum_generated_steps,
        "self_cache_capacity": capacity,
        "update_mode": spec.update_mode,
        "position_semantics": spec.position_semantics,
        "first_step_position": spec.first_step_position,
        "step_position_upper_bound_exclusive": (
            spec.step_position_upper_bound_exclusive
        ),
        "layer_bindings": layer_bindings,
    }


def derive_codec(
    spec: CodecSpec,
    engines: dict[str, InspectedEngine],
) -> JsonObject:
    initial = engines["nanocodec_initial_4"]
    steady = engines["nanocodec_steady_8"]
    tail = engines["nanocodec_tail_1_8"]
    initial_outputs = tensor_map(
        initial.outputs, "nanocodec_initial_4", "output"
    )
    steady_inputs = tensor_map(
        steady.inputs, "nanocodec_steady_8", "input"
    )
    steady_outputs = tensor_map(
        steady.outputs, "nanocodec_steady_8", "output"
    )
    tail_inputs = tensor_map(tail.inputs, "nanocodec_tail_1_8", "input")
    tail_outputs = tensor_map(tail.outputs, "nanocodec_tail_1_8", "output")
    ordered_logical_names = tuple(
        tensor.name.removeprefix("state_out.")
        for tensor in initial.outputs
        if tensor.name.startswith("state_out.")
    )
    if len(ordered_logical_names) != 97:
        raise PackageError(
            "nanocodec_initial_4: expected exactly 97 persistent state outputs"
        )
    if len(set(ordered_logical_names)) != len(ordered_logical_names):
        raise PackageError("nanocodec_initial_4: duplicate state logical name")
    states: list[JsonValue] = []
    for logical_name in ordered_logical_names:
        output_name = "state_out." + logical_name
        input_name = "state_in." + logical_name
        initial_tensor = require_tensor(
            initial_outputs, output_name, "nanocodec_initial_4"
        )
        candidates = (
            require_tensor(steady_inputs, input_name, "nanocodec_steady_8"),
            require_tensor(steady_outputs, output_name, "nanocodec_steady_8"),
            require_tensor(tail_inputs, input_name, "nanocodec_tail_1_8"),
            require_tensor(tail_outputs, output_name, "nanocodec_tail_1_8"),
        )
        if any(
            candidate.dtype != initial_tensor.dtype
            or candidate.shape != initial_tensor.shape
            for candidate in candidates
        ):
            raise PackageError(
                f"NanoCodec persistent state mismatch for {logical_name}"
            )
        states.append(
            {
                "logical_name": logical_name,
                "dtype": initial_tensor.dtype,
                "shape": list(initial_tensor.shape),
                "initial_output_binding": output_name,
                "steady_input_binding": input_name,
                "steady_output_binding": output_name,
                "tail_input_binding": input_name,
                "tail_output_binding": output_name,
            }
        )
    expected_initial_outputs = {
        "pcm",
        "valid_sample_length",
        *("state_out." + name for name in ordered_logical_names),
    }
    expected_stateful_inputs = {
        "codec_tokens",
        *("state_in." + name for name in ordered_logical_names),
    }
    expected_stateful_outputs = expected_initial_outputs
    if set(initial_outputs) != expected_initial_outputs:
        raise PackageError("nanocodec_initial_4: unexpected output binding set")
    if set(steady_inputs) != expected_stateful_inputs:
        raise PackageError("nanocodec_steady_8: unexpected input binding set")
    if set(steady_outputs) != expected_stateful_outputs:
        raise PackageError("nanocodec_steady_8: unexpected output binding set")
    if set(tail_inputs) != expected_stateful_inputs:
        raise PackageError("nanocodec_tail_1_8: unexpected input binding set")
    if set(tail_outputs) != expected_stateful_outputs:
        raise PackageError("nanocodec_tail_1_8: unexpected output binding set")
    return {
        "initial_engine_name": spec.initial_engine_name,
        "steady_engine_name": spec.steady_engine_name,
        "tail_engine_name": spec.tail_engine_name,
        "sample_rate_hz": spec.sample_rate_hz,
        "hop_length_samples": spec.hop_length_samples,
        "channels": spec.channels,
        "pcm_format": spec.pcm_format,
        "stateful": spec.stateful,
        "initial_frames": spec.initial_frames,
        "steady_frames": spec.steady_frames,
        "tail_min_frames": spec.tail_min_frames,
        "tail_max_frames": spec.tail_max_frames,
        "eos_frame_is_audio": spec.eos_frame_is_audio,
        "zero_frame_finalization": spec.zero_frame_finalization,
        "state_bindings": states,
    }


def engine_manifest_json(
    destination: EngineDestination,
    file: FileArtifact,
    inspected: InspectedEngine,
) -> JsonObject:
    return {
        "name": destination.name,
        "role": destination.role,
        "file": file.to_json(),
        "inputs": [tensor.to_json() for tensor in inspected.inputs],
        "outputs": [tensor.to_json() for tensor in inspected.outputs],
        "profiles": [profile.to_json() for profile in inspected.profiles],
    }


def build_manifest(
    spec: PackageSpec,
    runtime: RuntimeFingerprint,
    files: dict[str, FileArtifact],
    engines: dict[str, InspectedEngine],
    golden_receipt: GoldenReceipt,
    plugin_identity: PluginIdentity,
) -> JsonObject:
    sampling_creator_name = plugin_identity.creators[0][0]
    if spec.local_ar.sampling_plugin_name != sampling_creator_name:
        raise PackageError(
            "local_ar.sampling_plugin_name must select the authenticated "
            "Local AR sampling creator"
        )
    if spec.export.baked_context_sha256 != golden_receipt.baked_context_sha256:
        raise PackageError(
            "golden receipt and export baked-context SHA-256 differ"
        )
    if (
        golden_receipt.initial_frames != spec.codec.initial_frames
        or golden_receipt.steady_frames != spec.codec.steady_frames
        or golden_receipt.tail_min_frames != spec.codec.tail_min_frames
        or golden_receipt.tail_max_frames != spec.codec.tail_max_frames
        or golden_receipt.eos_frame_is_audio
        != spec.codec.eos_frame_is_audio
        or golden_receipt.zero_frame_finalization
        != spec.codec.zero_frame_finalization
    ):
        raise PackageError("golden receipt and codec stream contract differ")
    destination_by_role = {
        destination.role: destination
        for destination in spec.destinations.engines
    }
    engine_documents = [
        engine_manifest_json(
            destination_by_role[role],
            files[role],
            engines[role],
        )
        for role in ENGINE_ROLES
    ]
    license_documents = [
        {
            "role": destination.role,
            "file": files[destination.role].to_json(),
        }
        for destination in spec.destinations.licenses
    ]
    snapshot_bytes = sum(file.size_bytes for file in files.values())
    if snapshot_bytes <= 0 or snapshot_bytes > MAX_BUNDLE_SNAPSHOT_BYTES:
        raise PackageError(
            f"bundle snapshot bytes outside runtime bound: {snapshot_bytes}"
        )
    limits: JsonObject = {
        "maximum_text_tokens": spec.limits.maximum_text_tokens,
        "maximum_decoder_steps": spec.limits.maximum_decoder_steps,
        "maximum_audio_frames": spec.limits.maximum_audio_frames,
        "maximum_sessions": spec.limits.maximum_sessions,
        "maximum_concurrent_requests": (
            spec.limits.maximum_concurrent_requests
        ),
        "pcm_ring_capacity_frames": spec.limits.pcm_ring_capacity_frames,
        "maximum_workspace_bytes": spec.limits.maximum_workspace_bytes,
        "maximum_device_memory_bytes": (
            spec.limits.maximum_device_memory_bytes
        ),
        "maximum_bundle_snapshot_bytes": snapshot_bytes,
    }
    source_model_file = files["source_model_acceptance_receipt"]
    export_file = files["export_receipt"]
    tokenizer_file = files["tokenizer_identity_receipt"]
    plugin_file = files["plugin"]
    manifest: JsonObject = {
        "schema_version": spec.schema_version,
        "bundle_id": spec.bundle_id,
        "created_at_utc": spec.created_at_utc,
        "runtime": runtime.to_json(),
        "artifacts": {
            "source_model": {
                "model_id": spec.source_model.model_id,
                "version": spec.source_model.version,
                "revision": spec.source_model.revision,
                "source_sha256": spec.source_model.source_sha256,
                "acceptance_receipt": source_model_file.to_json(),
            },
            "export": {
                "format": spec.export.format,
                "source_revision": spec.export.source_revision,
                "voice_id": spec.export.voice_id,
                "baked_context_length": spec.export.baked_context_length,
                "baked_context_sha256": spec.export.baked_context_sha256,
                "audio_bos_baked": spec.export.audio_bos_baked,
                "export_receipt": export_file.to_json(),
            },
            "tokenizer": {
                "kind": spec.tokenizer.kind,
                "tokenizer_vocabulary_size": (
                    spec.tokenizer.tokenizer_vocabulary_size
                ),
                "text_embedding_rows": spec.tokenizer.text_embedding_rows,
                "special_tokens": spec.tokenizer.special_tokens.to_json(),
                "identity_sha256": spec.tokenizer.identity_sha256,
                "identity_receipt": tokenizer_file.to_json(),
            },
            "plugin": {
                "name": spec.plugin.name,
                "abi_version": plugin_identity.abi_version,
                "file": plugin_file.to_json(),
                "build_receipt": files["plugin_build_receipt"].to_json(),
            },
        },
        "classifier_free_guidance": spec.classifier_free_guidance.to_json(),
        "licenses": license_documents,
        "engines": engine_documents,
        "kv_cache": derive_kv_cache(spec.kv_cache, engines),
        "alignment": spec.alignment.to_json(),
        "sampling": spec.sampling.to_json(),
        "local_ar": spec.local_ar.to_json(),
        "codec": derive_codec(spec.codec, engines),
        "limits": limits,
        "golden_fixture": files["golden_fixture"].to_json(),
        "golden_receipt": golden_receipt.manifest_json(
            files["golden_receipt"]
        ),
    }
    return manifest


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise PackageError("artifact destination write made no progress")
        offset += written


def copy_verified_source(
    source: Path,
    staging_root: Path,
    destination: str,
    label: str,
) -> CopiedArtifact:
    require_regular_non_symlink(source, label)
    destination_path = staging_root.joinpath(*PurePosixPath(destination).parts)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise PackageError(f"{label} changed before packaging: {source}")
        destination_descriptor = os.open(
            destination_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o444,
        )
        digest = hashlib.sha256()
        copied = 0
        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                write_all(destination_descriptor, chunk)
                copied += len(chunk)
            os.fchmod(destination_descriptor, 0o444)
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        after = os.fstat(source_descriptor)
        stable_fields = (
            before.st_dev == after.st_dev,
            before.st_ino == after.st_ino,
            before.st_size == after.st_size,
            before.st_mtime_ns == after.st_mtime_ns,
            before.st_ctime_ns == after.st_ctime_ns,
        )
        if not all(stable_fields) or copied != before.st_size:
            raise PackageError(f"{label} changed while being packaged: {source}")
        return CopiedArtifact(
            artifact=FileArtifact(
                path=destination,
                sha256=digest.hexdigest(),
                size_bytes=copied,
            ),
            source_device=before.st_dev,
            source_inode=before.st_ino,
        )
    finally:
        os.close(source_descriptor)


def bundle_inventory_entry(status: os.stat_result) -> BundleInventoryEntry:
    return BundleInventoryEntry(
        mode=status.st_mode,
        device=status.st_dev,
        inode=status.st_ino,
        size_bytes=status.st_size,
        mtime_ns=status.st_mtime_ns,
        ctime_ns=status.st_ctime_ns,
    )


def scan_bundle_inventory(root: Path) -> dict[str, BundleInventoryEntry]:
    root_status = root.lstat()
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise PackageError("bundle staging root must be a nonsymlink directory")
    inventory = {"": bundle_inventory_entry(root_status)}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            status = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(status.st_mode):
                raise PackageError(
                    f"bundle inventory contains a symbolic link: {relative}"
                )
            if not (
                stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)
            ):
                raise PackageError(
                    f"bundle inventory contains a non-file entry: {relative}"
                )
            inventory[relative] = bundle_inventory_entry(status)
            if stat.S_ISDIR(status.st_mode):
                pending.append(path)
    return inventory


def expected_bundle_directories(relative_files: set[str]) -> set[str]:
    directories = {""}
    for relative in relative_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def verify_file_artifact_again(
    root: Path,
    artifact: FileArtifact,
) -> None:
    path = root.joinpath(*PurePosixPath(artifact.path).parts)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PackageError(
                f"bundle artifact is no longer a regular file: {artifact.path}"
            )
        if before.st_size != artifact.size_bytes:
            raise PackageError(
                f"bundle artifact size changed after validation: {artifact.path}"
            )
        digest = hashlib.sha256()
        read_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            read_bytes += len(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            before.st_dev == after.st_dev,
            before.st_ino == after.st_ino,
            before.st_size == after.st_size,
            before.st_mtime_ns == after.st_mtime_ns,
            before.st_ctime_ns == after.st_ctime_ns,
        )
        if not all(stable_fields) or read_bytes != before.st_size:
            raise PackageError(
                f"bundle artifact changed during post-validator verification: "
                f"{artifact.path}"
            )
        if digest.hexdigest() != artifact.sha256:
            raise PackageError(
                f"bundle artifact digest changed after validation: {artifact.path}"
            )
    finally:
        os.close(descriptor)


def verify_exact_bundle_snapshot(
    root: Path,
    expected_files: tuple[FileArtifact, ...],
) -> None:
    files_by_path = {artifact.path: artifact for artifact in expected_files}
    if len(files_by_path) != len(expected_files):
        raise PackageError("post-validator bundle inventory has duplicate paths")
    expected_paths = set(files_by_path)
    expected_entries = expected_paths | expected_bundle_directories(expected_paths)
    before = scan_bundle_inventory(root)
    actual_entries = set(before)
    if actual_entries != expected_entries:
        raise PackageError(
            "post-validator bundle entry set mismatch: "
            f"missing={sorted(expected_entries - actual_entries)}, "
            f"extra={sorted(actual_entries - expected_entries)}"
        )
    for relative in sorted(files_by_path):
        verify_file_artifact_again(root, files_by_path[relative])
    after = scan_bundle_inventory(root)
    if after != before:
        raise PackageError(
            "bundle inventory changed during post-validator verification"
        )


def run_validator(
    executable: Path,
    arguments: tuple[str, ...],
    label: str,
) -> None:
    require_regular_non_symlink(executable, label)
    if not os.access(executable, os.X_OK):
        raise PackageError(f"{label} is not executable: {executable}")
    result = subprocess.run(
        (str(executable), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if result.returncode != 0:
        output = result.stdout.decode("utf-8", errors="replace")
        raise PackageError(
            f"{label} failed with exit code {result.returncode}:\n{output}"
        )


def rename_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = int(
        renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise PackageError(f"output already exists: {destination}")
        raise PackageError(
            f"renameat2(RENAME_NOREPLACE) failed: {os.strerror(error_number)}"
        )


def package_runtime_bundle(
    spec_path: Path,
    input_paths: InputPaths,
    output: Path,
    device_ordinal: int,
) -> tuple[Path, str]:
    problems = inventory_problems(input_paths)
    if problems:
        raise PackageError(
            "required input inventory is incomplete:\n" + "\n".join(problems)
        )
    spec = parse_package_spec(load_json_file(spec_path))

    output_parent_input = output.parent.absolute()
    output_parent = output_parent_input.resolve(strict=True)
    if output_parent_input != output_parent:
        raise PackageError(
            f"output parent must not contain symlinks or '..': {output.parent}"
        )
    if not output_parent.is_dir():
        raise PackageError(f"output parent is not a directory: {output_parent}")
    final_output = output_parent / output.name
    if final_output.exists() or final_output.is_symlink():
        raise PackageError(f"output already exists: {final_output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output_parent)
    )
    published = False
    try:
        destination_by_role = {
            destination.role: destination
            for destination in spec.destinations.engines
        }
        sources: list[tuple[str, Path, str, str]] = [
            (
                "source_model_acceptance_receipt",
                input_paths.source_model_acceptance_receipt,
                spec.destinations.source_model_acceptance_receipt,
                "source-model acceptance receipt",
            ),
            (
                "export_receipt",
                input_paths.export_receipt,
                spec.destinations.export_receipt,
                "export receipt",
            ),
            (
                "tokenizer_identity_receipt",
                input_paths.tokenizer_identity_receipt,
                spec.destinations.tokenizer_identity_receipt,
                "tokenizer identity receipt",
            ),
            (
                "plugin_build_receipt",
                input_paths.plugin_build_receipt,
                spec.destinations.plugin_build_receipt,
                "plugin build receipt",
            ),
            (
                "plugin",
                input_paths.plugin,
                spec.destinations.plugin,
                "plugin",
            ),
            (
                "golden_receipt",
                input_paths.golden_receipt,
                spec.destinations.golden_receipt,
                "golden receipt",
            ),
            (
                "golden_fixture",
                input_paths.golden_fixture,
                spec.destinations.golden_fixture,
                "golden fixture",
            ),
        ]
        license_source_by_role = dict(input_paths.licenses)
        if tuple(license_source_by_role) != LICENSE_ROLES:
            raise PackageError(
                "license input inventory must use the canonical eight-role order"
            )
        for destination in spec.destinations.licenses:
            sources.append(
                (
                    destination.role,
                    license_source_by_role[destination.role],
                    destination.path,
                    f"{destination.role} license artifact",
                )
            )
        for role, plan_path in input_paths.engines:
            sources.append(
                (
                    role,
                    plan_path,
                    destination_by_role[role].path,
                    f"{role} plan",
                )
            )
        files: dict[str, FileArtifact] = {}
        identities: set[tuple[int, int]] = set()
        for logical_name, source, destination, label in sources:
            copied = copy_verified_source(
                source, staging, destination, label
            )
            identity = (copied.source_device, copied.source_inode)
            if identity in identities:
                raise PackageError(
                    f"two bundle artifacts reference the same source file: {source}"
                )
            identities.add(identity)
            files[logical_name] = copied.artifact
        if (
            files["nvidia_open_model_license"].sha256
            != NVIDIA_OPEN_MODEL_LICENSE_SHA256
        ):
            raise PackageError(
                "nvidia_open_model_license differs from the accepted "
                "model-license document"
            )
        if (
            files["nvidia_model_notice"].sha256
            != NVIDIA_MODEL_NOTICE_SHA256
        ):
            raise PackageError(
                "nvidia_model_notice differs from the accepted model notice"
            )

        staged_source_receipt = staging.joinpath(
            *PurePosixPath(
                spec.destinations.source_model_acceptance_receipt
            ).parts
        )
        staged_export_receipt = staging.joinpath(
            *PurePosixPath(spec.destinations.export_receipt).parts
        )
        staged_tokenizer_receipt = staging.joinpath(
            *PurePosixPath(
                spec.destinations.tokenizer_identity_receipt
            ).parts
        )
        staged_plugin_build_receipt = staging.joinpath(
            *PurePosixPath(spec.destinations.plugin_build_receipt).parts
        )
        staged_golden_receipt = staging.joinpath(
            *PurePosixPath(spec.destinations.golden_receipt).parts
        )
        staged_golden_fixture = staging.joinpath(
            *PurePosixPath(spec.destinations.golden_fixture).parts
        )
        validate_source_model_receipt(
            staged_source_receipt, spec.source_model
        )
        export_evidence = validate_consolidated_export_receipt(
            staged_export_receipt, spec, files
        )
        validate_tokenizer_identity_receipt(
            staged_tokenizer_receipt, spec.tokenizer
        )
        validate_plugin_build_receipt(
            staged_plugin_build_receipt,
            files["plugin"],
        )
        golden_receipt = parse_golden_receipt(staged_golden_receipt)
        golden_fixture = parse_golden_fixture(staged_golden_fixture)
        validate_golden_fixture(
            golden_fixture,
            golden_receipt,
            spec,
            export_evidence,
        )

        staged_plugin = staging.joinpath(
            *PurePosixPath(spec.destinations.plugin).parts
        )
        plugin_identity = load_and_register_plugin(staged_plugin)
        staged_plans = tuple(
            (
                role,
                staging.joinpath(
                    *PurePosixPath(destination_by_role[role].path).parts
                ),
            )
            for role in ENGINE_ROLES
        )
        tensorrt_version, inspected_engines = inspect_plan_files(staged_plans)
        runtime = collect_runtime_fingerprint(
            tensorrt_version,
            plugin_identity.abi_version,
            device_ordinal,
        )
        manifest = build_manifest(
            spec,
            runtime,
            files,
            inspected_engines,
            golden_receipt,
            plugin_identity,
        )
        manifest_payload = pretty_json_bytes(manifest)
        if len(manifest_payload) > MAX_MANIFEST_BYTES:
            raise PackageError(
                f"generated manifest exceeds {MAX_MANIFEST_BYTES} bytes"
            )
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_bytes(manifest_payload)
        manifest_path.chmod(0o444)
        manifest_sha256 = sha256_bytes(manifest_payload)
        digest_path = staging / MANIFEST_DIGEST_NAME
        digest_path.write_text(
            f"{manifest_sha256}  {MANIFEST_NAME}\n",
            encoding="ascii",
        )
        digest_path.chmod(0o444)
        run_validator(
            input_paths.manifest_validator,
            (str(manifest_path),),
            "manifest validator",
        )
        run_validator(
            input_paths.bundle_validator,
            (str(staging), manifest_sha256),
            "bundle validator",
        )
        digest_payload = (
            f"{manifest_sha256}  {MANIFEST_NAME}\n".encode("ascii")
        )
        verify_exact_bundle_snapshot(
            staging,
            (
                *files.values(),
                FileArtifact(
                    path=MANIFEST_NAME,
                    sha256=manifest_sha256,
                    size_bytes=len(manifest_payload),
                ),
                FileArtifact(
                    path=MANIFEST_DIGEST_NAME,
                    sha256=sha256_bytes(digest_payload),
                    size_bytes=len(digest_payload),
                ),
            ),
        )
        rename_directory_no_replace(staging, final_output)
        published = True
        return final_output, manifest_sha256
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an immutable introspected MagpieTTS-RT P2 bundle"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--source-model-acceptance-receipt", type=Path, required=True
    )
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-identity-receipt", type=Path, required=True
    )
    parser.add_argument("--plugin-build-receipt", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--text-encoder-plan", type=Path, required=True)
    parser.add_argument(
        "--main-decoder-prefill-plan", type=Path, required=True
    )
    parser.add_argument(
        "--main-decoder-step-plan", type=Path, required=True
    )
    parser.add_argument("--local-ar-plan", type=Path, required=True)
    parser.add_argument(
        "--nanocodec-initial-4-plan", type=Path, required=True
    )
    parser.add_argument(
        "--nanocodec-steady-8-plan", type=Path, required=True
    )
    parser.add_argument(
        "--nanocodec-tail-1-8-plan", type=Path, required=True
    )
    parser.add_argument("--golden-fixture", type=Path, required=True)
    parser.add_argument("--golden-receipt", type=Path, required=True)
    parser.add_argument("--project-license", type=Path, required=True)
    parser.add_argument("--project-notice", type=Path, required=True)
    parser.add_argument("--pytorch-license", type=Path, required=True)
    parser.add_argument("--cutlass-license", type=Path, required=True)
    parser.add_argument("--cuda-eula", type=Path, required=True)
    parser.add_argument("--cuda-notice", type=Path, required=True)
    parser.add_argument(
        "--nvidia-open-model-license", type=Path, required=True
    )
    parser.add_argument("--nvidia-model-notice", type=Path, required=True)
    parser.add_argument("--manifest-validator", type=Path, required=True)
    parser.add_argument("--bundle-validator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="report the complete missing/invalid input inventory and exit",
    )
    return parser.parse_args()


def input_paths_from_args(args: argparse.Namespace) -> InputPaths:
    return InputPaths(
        specification=args.spec,
        source_model_acceptance_receipt=(
            args.source_model_acceptance_receipt
        ),
        export_receipt=args.export_receipt,
        tokenizer_identity_receipt=args.tokenizer_identity_receipt,
        plugin_build_receipt=args.plugin_build_receipt,
        plugin=args.plugin,
        golden_fixture=args.golden_fixture,
        golden_receipt=args.golden_receipt,
        manifest_validator=args.manifest_validator,
        bundle_validator=args.bundle_validator,
        licenses=(
            ("project_license", args.project_license),
            ("project_notice", args.project_notice),
            ("pytorch_license", args.pytorch_license),
            ("cutlass_license", args.cutlass_license),
            ("cuda_eula", args.cuda_eula),
            ("cuda_notice", args.cuda_notice),
            (
                "nvidia_open_model_license",
                args.nvidia_open_model_license,
            ),
            ("nvidia_model_notice", args.nvidia_model_notice),
        ),
        engines=(
            ("text_encoder", args.text_encoder_plan),
            ("main_decoder_prefill", args.main_decoder_prefill_plan),
            ("main_decoder_step", args.main_decoder_step_plan),
            ("local_ar_16", args.local_ar_plan),
            ("nanocodec_initial_4", args.nanocodec_initial_4_plan),
            ("nanocodec_steady_8", args.nanocodec_steady_8_plan),
            ("nanocodec_tail_1_8", args.nanocodec_tail_1_8_plan),
        ),
    )


def main() -> int:
    args = parse_args()
    if args.device_ordinal < 0:
        print("error: --device-ordinal must be nonnegative", file=sys.stderr)
        return 2
    paths = input_paths_from_args(args)
    if args.check_inputs:
        problems = inventory_problems(paths)
        if problems:
            print("\n".join(problems))
            return 3
        print("all required package inputs are present and regular")
        return 0
    try:
        output, manifest_sha256 = package_runtime_bundle(
            args.spec,
            paths,
            args.output,
            args.device_ordinal,
        )
    except (PackageError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"bundle={output}")
    print(f"manifest_sha256={manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

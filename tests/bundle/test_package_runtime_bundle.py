from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "tools" / "bundle" / "package_runtime_bundle.py"
SPEC = importlib.util.spec_from_file_location("package_runtime_bundle", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_cublas_identity() -> MODULE.CublasRuntimeIdentity:
    return MODULE.parse_cublas_runtime_identity(
        {
            "api_version_integer": 130400,
            "library": {
                "soname": "libcublas.so.13",
                "size_bytes": 67_751_616,
                "sha256": (
                    "826486b8869144621e3a477cddcd28f56733c7c80c6f998b"
                    "898384fc09e10f91"
                ),
            },
            "lt_library": {
                "soname": "libcublasLt.so.13",
                "size_bytes": 606_744_240,
                "sha256": (
                    "b7aa42c190c2e7490abd6ea987883e05678e26222b7f9f1c9"
                    "b96374fcbddbf04"
                ),
            },
        },
        "test.cublas",
    )


def valid_cuda_identity() -> MODULE.CudaRuntimeIdentity:
    return MODULE.parse_cuda_runtime_identity(
        {
            "cuda_driver_api_version_integer": 13020,
            "cuda_runtime_version_integer": 13020,
            "nvidia_driver_version": "595.78",
        },
        "test.cuda",
    )


def valid_package_spec() -> MODULE.JsonObject:
    fixture = json.loads(
        (
            PROJECT_ROOT / "tests" / "manifest" / "fixtures" / "minimal-valid.json"
        ).read_text(encoding="utf-8")
    )
    tokenizer_receipt = json.loads(
        (
            PROJECT_ROOT
            / "tests"
            / "frontend"
            / "fixtures"
            / "japanese-source-spans-v1.json"
        ).read_text(encoding="utf-8")
    )
    tokenizer_identity = MODULE.sha256_bytes(
        MODULE.canonical_json_bytes(
            MODULE.tokenizer_identity_projection(tokenizer_receipt)
        )
    )
    kv_cache_keys = (
        "layout",
        "prefix_length",
        "maximum_generated_steps",
        "update_mode",
        "position_semantics",
        "first_step_position",
        "step_position_upper_bound_exclusive",
    )
    codec_keys = (
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
    limits_keys = (
        "maximum_text_tokens",
        "maximum_decoder_steps",
        "maximum_audio_frames",
        "maximum_sessions",
        "maximum_concurrent_requests",
        "pcm_ring_capacity_frames",
        "maximum_workspace_bytes",
        "maximum_device_memory_bytes",
    )
    return {
        "schema_version": 1,
        "bundle_id": fixture["bundle_id"],
        "created_at_utc": fixture["created_at_utc"],
        "source_model": {
            "model_id": fixture["artifacts"]["source_model"]["model_id"],
            "version": fixture["artifacts"]["source_model"]["version"],
            "revision": fixture["artifacts"]["source_model"]["revision"],
            "source_sha256": fixture["artifacts"]["source_model"][
                "source_sha256"
            ],
        },
        "export": {
            key: value
            for key, value in fixture["artifacts"]["export"].items()
            if key != "export_receipt"
        }
        | {"receipt_artifact_role": "runtime_bundle_export"},
        "tokenizer": {
            "kind": "japanese_phoneme",
            "tokenizer_vocabulary_size": tokenizer_receipt[
                "frontend_contract"
            ]["aggregate_vocabulary_size"],
            "text_embedding_rows": tokenizer_receipt[
                "frontend_contract"
            ]["text_embedding_rows"],
            "special_tokens": {
                "bos_token_id": tokenizer_receipt[
                    "frontend_contract"
                ]["bos_token_id"],
                "eos_token_id": tokenizer_receipt[
                    "frontend_contract"
                ]["eos_token_id"],
                "japanese_global_pad_token_id": tokenizer_receipt[
                    "frontend_contract"
                ]["japanese"]["global_pad_token_id"],
            },
            "identity_sha256": tokenizer_identity,
        },
        "plugin": {"name": fixture["artifacts"]["plugin"]["name"]},
        "classifier_free_guidance": fixture["classifier_free_guidance"],
        "kv_cache": {
            key: fixture["kv_cache"][key] for key in kv_cache_keys
        },
        "alignment": fixture["alignment"],
        "sampling": fixture["sampling"],
        "local_ar": fixture["local_ar"],
        "codec": {key: fixture["codec"][key] for key in codec_keys},
        "limits": {key: fixture["limits"][key] for key in limits_keys},
        "destinations": {
            "source_model_acceptance_receipt": (
                "receipts/source-model-acceptance.json"
            ),
            "export_receipt": "receipts/export.json",
            "tokenizer_identity_receipt": "receipts/tokenizer-identity.json",
            "plugin_build_receipt": "receipts/plugin-build.json",
            "plugin": "lib/libmagpie_tts_rt_plugins.so",
            "golden_fixture": "fixtures/golden-input.json",
            "golden_receipt": "receipts/golden.json",
            "licenses": [
                {
                    "role": license_artifact["role"],
                    "path": license_artifact["file"]["path"],
                }
                for license_artifact in fixture["licenses"]
            ],
            "engines": [
                {
                    "name": engine["name"],
                    "role": engine["role"],
                    "path": engine["file"]["path"],
                }
                for engine in fixture["engines"]
            ],
        },
    }


def valid_source_model_receipt(
    source_model: MODULE.SourceModelSpec,
) -> MODULE.JsonObject:
    files = {
        relative: hashlib.sha256(relative.encode("utf-8")).hexdigest()
        for relative in MODULE.ORACLE_SOURCE_PATHS
    }
    bundle = hashlib.sha256()
    for relative, digest in files.items():
        bundle.update(relative.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(digest.encode("ascii"))
        bundle.update(b"\n")
    metric = {
        "median": 1.0,
        "p95": 1.0,
        "minimum": 1.0,
        "maximum": 1.0,
    }
    gates = {
        gate: True for gate in MODULE.SOURCE_ACCEPTANCE_PASSED_GATES
    }
    gates["parakeet_gate_pending"] = True
    gates["failures"] = []
    determinism = [
        {
            "case_name": case_name,
            "local_ar_seed": 20260729,
            "first_codes_sha256": f"{index + 1:x}" * 64,
            "replay_codes_sha256": f"{index + 1:x}" * 64,
            "codes_exact": True,
            "first_pcm_f32le_sha256": f"{index + 5:x}" * 64,
            "replay_pcm_f32le_sha256": f"{index + 5:x}" * 64,
            "pcm_exact": True,
            "first_codec_frame_count": index + 1,
            "replay_codec_frame_count": index + 1,
            "codec_frame_count_exact": True,
            "passed": True,
        }
        for index, case_name in enumerate(
            MODULE.SOURCE_ACCEPTANCE_CASE_NAMES
        )
    ]
    return {
        "schema_version": 2,
        "artifact_role": "source_model_acceptance",
        "status": "accepted",
        "created_at_utc": "2026-07-30T00:00:00.000000Z",
        "source_model": {
            "model_id": source_model.model_id,
            "version": source_model.version,
            "revision": source_model.revision,
            "file_name": "magpie_tts_multilingual_357m.nemo",
            "sha256": source_model.source_sha256,
            "size_bytes": 1470208000,
            "license": {
                "name": MODULE.NVIDIA_OPEN_MODEL_LICENSE_NAME,
                "version": MODULE.NVIDIA_OPEN_MODEL_LICENSE_VERSION,
                "document_file_name": (
                    MODULE.NVIDIA_OPEN_MODEL_LICENSE_FILE_NAME
                ),
                "document_sha256": (
                    MODULE.NVIDIA_OPEN_MODEL_LICENSE_SHA256
                ),
                "required_notice_file_name": (
                    MODULE.NVIDIA_MODEL_NOTICE_FILE_NAME
                ),
                "required_notice_sha256": (
                    MODULE.NVIDIA_MODEL_NOTICE_SHA256
                ),
            },
        },
        "oracle_source": {
            "repository": "https://github.com/NVIDIA/NeMo.git",
            "base_revision": "1" * 40,
            "optimized_source_bundle_sha256": bundle.hexdigest(),
            "files": files,
        },
        "acceptance_contract": {
            "speaker_name": "Sofia",
            "speaker_index": 4,
            "local_ar_seed": 20260729,
            "sample_rate_hz": 22050,
            "samples_per_codec_frame": 1024,
            "first_codec_frames": 4,
            "steady_codec_frames": 8,
            "tail_codec_frames_min": 1,
            "tail_codec_frames_max": 8,
        },
        "probe": {
            "raw_receipt_sha256": "2" * 64,
            "raw_receipt_size_bytes": 1024,
            "probe_sha256": "3" * 64,
            "helper_sha256": "4" * 64,
            "case_definition_sha256": "5" * 64,
        },
        "runtime_environment": {
            "torch_version": "2.11.0",
            "torch_cuda_build": "13.0",
            "cudnn_version": 92300,
            "cuda_driver_version": "595.78",
            "gpu_name": "NVIDIA Thor",
            "gpu_compute_capability": [11, 0],
        },
        "evidence": {
            "case_names": list(MODULE.SOURCE_ACCEPTANCE_CASE_NAMES),
            "determinism": determinism,
            "aggregate": {
                "case_count": 4,
                "raw_ttfa_ms": dict(metric),
                "gapless_start_ms": dict(metric),
                "max_positive_playback_lateness_ms": dict(metric),
                "generation_rtf": dict(metric),
                "total_rtf": dict(metric),
                "maximum_peak_cuda_allocated_bytes": 1,
                "maximum_peak_cuda_reserved_bytes": 1,
            },
            "gates": gates,
        },
    }


class FakeTensorIOMode:
    INPUT = "input"
    OUTPUT = "output"


class FakeTensorLocation:
    DEVICE = "device"
    HOST = "host"


class FakeTensorRt:
    float32 = "fp32"
    float16 = "fp16"
    bfloat16 = "bf16"
    int64 = "int64"
    int32 = "int32"
    int8 = "int8"
    uint8 = "uint8"
    bool = "bool"
    TensorIOMode = FakeTensorIOMode
    TensorLocation = FakeTensorLocation


class FakeEngine:
    num_optimization_profiles = 1

    def __init__(self) -> None:
        self.records = (
            ("text_token_ids", FakeTensorIOMode.INPUT, "int32", (1, -1)),
            ("text_mask", FakeTensorIOMode.INPUT, "bool", (1, -1)),
            (
                "text_condition",
                FakeTensorIOMode.OUTPUT,
                "bf16",
                (1, -1, 768),
            ),
        )
        self.num_io_tensors = len(self.records)

    def get_tensor_name(self, index: int) -> str:
        return self.records[index][0]

    def get_tensor_mode(self, name: str) -> str:
        return next(record[1] for record in self.records if record[0] == name)

    def get_tensor_dtype(self, name: str) -> str:
        return next(record[2] for record in self.records if record[0] == name)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return next(record[3] for record in self.records if record[0] == name)

    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if profile_index != 0 or name not in ("text_token_ids", "text_mask"):
            raise RuntimeError("unexpected profile query")
        return ((1, 1), (1, 64), (1, 512))

    def get_tensor_location(self, name: str) -> str:
        if name not in {record[0] for record in self.records}:
            raise RuntimeError("unknown tensor")
        return FakeTensorLocation.DEVICE

    def is_shape_inference_io(self, name: str) -> bool:
        if name not in {record[0] for record in self.records}:
            raise RuntimeError("unknown tensor")
        return False


class FakeMainDecoderStepPositionEngine:
    num_optimization_profiles = 1

    def __init__(
        self,
        *,
        position_location: str = FakeTensorLocation.DEVICE,
        position_shape_inference_io: bool = False,
        include_execution_status: bool = True,
        status_dtype: str = "int32",
    ) -> None:
        status_records = (
            (
                "execution_status_in",
                FakeTensorIOMode.INPUT,
                status_dtype,
                (),
            ),
            (
                "execution_status_out",
                FakeTensorIOMode.OUTPUT,
                status_dtype,
                (),
            ),
        ) if include_execution_status else ()
        self.records = (
            ("position", FakeTensorIOMode.INPUT, "int64", ()),
            *status_records,
            ("decoder_hidden", FakeTensorIOMode.OUTPUT, "bf16", (2, 768)),
        )
        self.num_io_tensors = len(self.records)
        self.position_location = position_location
        self.position_shape_inference_io = position_shape_inference_io

    def get_tensor_name(self, index: int) -> str:
        return self.records[index][0]

    def get_tensor_mode(self, name: str) -> str:
        return next(record[1] for record in self.records if record[0] == name)

    def get_tensor_dtype(self, name: str) -> str:
        return next(record[2] for record in self.records if record[0] == name)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return next(record[3] for record in self.records if record[0] == name)

    def get_tensor_location(self, name: str) -> str:
        if name == "position":
            return self.position_location
        if name in {
            "decoder_hidden",
            "execution_status_in",
            "execution_status_out",
        }:
            return FakeTensorLocation.DEVICE
        raise RuntimeError("unknown tensor")

    def is_shape_inference_io(self, name: str) -> bool:
        if name == "position":
            return self.position_shape_inference_io
        if name in {
            "decoder_hidden",
            "execution_status_in",
            "execution_status_out",
        }:
            return False
        raise RuntimeError("unknown tensor")


def fake_codec_engine(
    role: str,
    *,
    include_state_inputs: bool,
    dynamic_frames: bool,
) -> MODULE.InspectedEngine:
    state_inputs = (
        tuple(
            MODULE.TensorSpec(
                name=f"state_in.state_{index}",
                dtype="fp32",
                shape=(1, index + 1, 2),
            )
            for index in range(97)
        )
        if include_state_inputs
        else ()
    )
    state_outputs = tuple(
        MODULE.TensorSpec(
            name=f"state_out.state_{index}",
            dtype="fp32",
            shape=(1, index + 1, 2),
        )
        for index in range(97)
    )
    frames = -1 if dynamic_frames else (4 if role.endswith("initial_4") else 8)
    return MODULE.InspectedEngine(
        inputs=(
            MODULE.TensorSpec("codec_tokens", "int64", (1, 8, frames)),
            *state_inputs,
        ),
        outputs=(
            MODULE.TensorSpec(
                "pcm",
                "fp32",
                (1, -1 if dynamic_frames else frames * 1024),
            ),
            MODULE.TensorSpec("valid_sample_length", "int64", (1,)),
            *state_outputs,
        ),
        profiles=(
            MODULE.OptimizationProfile(
                MODULE.PROFILE_NAMES_BY_ROLE[role],
                (),
            ),
        ),
    )


def fake_main_decoder_engines() -> dict[str, MODULE.InspectedEngine]:
    prefill_outputs: list[MODULE.TensorSpec] = []
    step_inputs: list[MODULE.TensorSpec] = []
    step_outputs: list[MODULE.TensorSpec] = []
    for index in range(12):
        prefill_outputs.extend(
            (
                MODULE.TensorSpec(
                    f"prefill_self_key_{index}",
                    "bf16",
                    (2, 467, 12, 64),
                ),
                MODULE.TensorSpec(
                    f"prefill_self_value_{index}",
                    "bf16",
                    (2, 467, 12, 64),
                ),
                MODULE.TensorSpec(
                    f"prefill_self_mask_{index}", "bool", (2, 467)
                ),
                MODULE.TensorSpec(
                    f"prefill_cross_key_{index}",
                    "bf16",
                    (2, -1, 1, 128),
                ),
                MODULE.TensorSpec(
                    f"prefill_cross_value_{index}",
                    "bf16",
                    (2, -1, 1, 128),
                ),
            )
        )
        step_inputs.extend(
            (
                MODULE.TensorSpec(
                    f"step_self_key_in_{index}",
                    "bf16",
                    (2, 467, 12, 64),
                ),
                MODULE.TensorSpec(
                    f"step_self_value_in_{index}",
                    "bf16",
                    (2, 467, 12, 64),
                ),
                MODULE.TensorSpec(
                    f"step_self_mask_in_{index}", "bool", (2, 467)
                ),
                MODULE.TensorSpec(
                    f"step_cross_key_in_{index}",
                    "bf16",
                    (2, -1, 1, 128),
                ),
                MODULE.TensorSpec(
                    f"step_cross_value_in_{index}",
                    "bf16",
                    (2, -1, 1, 128),
                ),
            )
        )
        step_outputs.extend(
            (
                MODULE.TensorSpec(
                    f"step_self_key_out_{index}",
                    "bf16",
                    (2, 467, 12, 64),
                ),
                MODULE.TensorSpec(
                    f"step_self_value_out_{index}",
                    "bf16",
                    (2, 467, 12, 64),
                ),
                MODULE.TensorSpec(
                    f"step_self_mask_out_{index}", "bool", (2, 467)
                ),
            )
        )
    profile = (MODULE.OptimizationProfile("text_1_512", ()),)
    return {
        "main_decoder_prefill": MODULE.InspectedEngine(
            inputs=(),
            outputs=tuple(prefill_outputs),
            profiles=profile,
        ),
        "main_decoder_step": MODULE.InspectedEngine(
            inputs=tuple(step_inputs),
            outputs=tuple(step_outputs),
            profiles=profile,
        ),
    }


def valid_golden_contract(
    specification: MODULE.PackageSpec,
) -> tuple[
    MODULE.GoldenFixture,
    MODULE.GoldenReceipt,
    MODULE.ExportReceiptEvidence,
]:
    token_ids = (842, 843, 3358)
    token_ids_sha256 = MODULE.prepared_token_ids_sha256(token_ids)
    normalized_text_sha256 = "a" * 64
    decoder_tokens_sha256 = "b" * 64
    codec_codes_sha256 = "c" * 64
    pcm_f32le_sha256 = "d" * 64
    oracle_lock_sha256 = "e" * 64
    codec_frame_count = 12
    pcm_sample_count = (
        codec_frame_count * specification.codec.hop_length_samples
    )
    fixture = MODULE.GoldenFixture(
        schema_version=1,
        fixture_id="startup-ja-v1",
        prepared_token_ids=token_ids,
        seed=20260729,
        tokenizer_identity_sha256=(
            specification.tokenizer.identity_sha256
        ),
        oracle_lock_sha256=oracle_lock_sha256,
        normalized_text_sha256=normalized_text_sha256,
        token_ids_sha256=token_ids_sha256,
        baked_context_sha256=(
            specification.export.baked_context_sha256
        ),
        expected=MODULE.GoldenFixtureExpected(
            decoder_tokens_sha256=decoder_tokens_sha256,
            codec_codes_sha256=codec_codes_sha256,
            codec_frame_count=codec_frame_count,
            pcm_f32le_sha256=pcm_f32le_sha256,
            pcm_sample_count=pcm_sample_count,
        ),
    )
    receipt = MODULE.GoldenReceipt(
        receipt_version=1,
        created_at_utc="2026-07-30T00:00:00Z",
        normalized_text_sha256=normalized_text_sha256,
        token_ids_sha256=token_ids_sha256,
        baked_context_sha256=(
            specification.export.baked_context_sha256
        ),
        seed=20260729,
        decoder_tokens_sha256=decoder_tokens_sha256,
        codec_codes_sha256=codec_codes_sha256,
        codec_frame_count=codec_frame_count,
        pcm_f32le_sha256=pcm_f32le_sha256,
        sample_count=pcm_sample_count,
        initial_frames=4,
        steady_frames=8,
        tail_min_frames=1,
        tail_max_frames=8,
        eos_frame_is_audio=False,
        zero_frame_finalization=MODULE.ZERO_FRAME_FINALIZATION,
    )
    return (
        fixture,
        receipt,
        MODULE.ExportReceiptEvidence(
            oracle_lock_sha256=oracle_lock_sha256,
            cuda_identity=valid_cuda_identity(),
            cublas_identity=valid_cublas_identity(),
            mode8_class_table_sha256="f" * 64,
        ),
    )


def fake_bundle_files() -> dict[str, MODULE.FileArtifact]:
    roles = (
        "source_model_acceptance_receipt",
        "tokenizer_identity_receipt",
        "plugin_build_receipt",
        "plugin",
        "golden_fixture",
        "golden_receipt",
        *MODULE.ENGINE_ROLES,
    )
    return {
        role: MODULE.FileArtifact(
            path=f"artifacts/{role}",
            sha256=f"{index + 1:064x}",
            size_bytes=1000 + index,
        )
        for index, role in enumerate(roles)
    }


def valid_consolidated_receipt(
    specification: MODULE.PackageSpec,
    files: dict[str, MODULE.FileArtifact],
) -> MODULE.JsonObject:
    component_receipts = (
        ("text_encoder", "text_encoder_plan"),
        ("main_decoder", "main_decoder"),
        ("local_ar", "local_ar_fixed_16"),
        ("nanocodec", "stateful_nanocodec"),
    )
    return {
        "schema_version": 1,
        "artifact_role": specification.export.receipt_artifact_role,
        "status": "accepted",
        "created_at_utc": "2026-07-30T00:00:00Z",
        "source": {
            "model_id": specification.source_model.model_id,
            "model_version": specification.source_model.version,
            "model_revision": specification.source_model.revision,
            "model_sha256": specification.source_model.source_sha256,
            "oracle_lock_sha256": "e" * 64,
            "canonical_fixture_manifest_sha256": "d" * 64,
            "source_model_acceptance_receipt_sha256": (
                files["source_model_acceptance_receipt"].sha256
            ),
            "tokenizer_identity_sha256": (
                specification.tokenizer.identity_sha256
            ),
            "tokenizer_identity_receipt_sha256": (
                files["tokenizer_identity_receipt"].sha256
            ),
            "locked_magpie_restore_sha256": "7" * 64,
            "codec_restore": {
                "embedded_codec_model_id": "nvidia/locked-codec",
                "codec_model_sha256": "9" * 64,
                "codec_model_size_bytes": 123,
                "codec_resolution": "authenticated_local_file",
                "use_scl_loss": False,
                "network_resolution": False,
            },
        },
        "runtime_dependencies": {
            "cuda": valid_cuda_identity().to_json(),
            "cublas": valid_cublas_identity().to_json(),
            "mode8_class_table_sha256": "f" * 64,
        },
        "component_receipts": [
            {
                "role": role,
                "artifact_role": artifact_role,
                "receipt_sha256": f"{index + 40:064x}",
            }
            for index, (role, artifact_role) in enumerate(
                component_receipts
            )
        ],
        "sequence_receipt_sha256": "9" * 64,
        "complete_generation_receipt_sha256": (
            files["golden_receipt"].sha256
        ),
        "plugin": {
            "role": "runtime_plugin",
            "sha256": files["plugin"].sha256,
            "size_bytes": files["plugin"].size_bytes,
            "build_receipt_sha256": files["plugin_build_receipt"].sha256,
            "build_receipt_size_bytes": (
                files["plugin_build_receipt"].size_bytes
            ),
        },
        "engines": [
            {
                "role": role,
                "sha256": files[role].sha256,
                "size_bytes": files[role].size_bytes,
            }
            for role in MODULE.ENGINE_ROLES
        ],
        "golden_fixture": {
            "sha256": files["golden_fixture"].sha256,
            "size_bytes": files["golden_fixture"].size_bytes,
        },
        "golden_receipt": {
            "sha256": files["golden_receipt"].sha256,
            "size_bytes": files["golden_receipt"].size_bytes,
        },
        "eos_frame_is_audio": False,
        "zero_frame_finalization": MODULE.ZERO_FRAME_FINALIZATION,
    }


def valid_plugin_build_receipt(
    plugin: MODULE.FileArtifact,
) -> MODULE.JsonObject:
    return {
        "schema_version": "magpie-tts-rt.plugin-build.v1",
        "status": "recorded",
        "artifact": {
            "filename": MODULE.EXPECTED_PLUGIN_FILENAME,
            "size_bytes": plugin.size_bytes,
            "sha256": plugin.sha256,
            "soname": MODULE.EXPECTED_PLUGIN_SONAME,
            "needed": sorted(MODULE.EXPECTED_PLUGIN_NEEDED),
        },
        "source": [
            {
                "path": path,
                "size_bytes": 123 + index,
                "sha256": f"{index + 10:064x}",
            }
            for index, path in enumerate(
                MODULE.EXPECTED_PLUGIN_SOURCE_PATHS
            )
        ],
        "toolchain": {
            "cxx_path": "/usr/bin/c++",
            "cxx_version": "c++ 13.3.0",
            "nvcc_path": "/usr/local/cuda-13.2/bin/nvcc",
            "nvcc_version": "Cuda compilation tools, release 13.2",
            "linker_path": "/usr/bin/ld",
            "linker_version": "GNU ld 2.42",
            "readelf_path": "/usr/bin/readelf",
            "ninja_path": "/usr/bin/ninja",
            "ninja_version": "1.11.1",
            "cmake_path": "/usr/bin/cmake",
            "cmake_version": "cmake version 3.28.3",
            "cuda_architecture": "110",
        },
        "build": {
            "build_type": "Release",
            "tf32_policy": "disabled",
            "cutlass_archive_sha256": (
                "5288044d2d5e81632ac0c812b6b85c744901a7d3fd11c9119f18f71c3cef5f79"
            ),
            "compile_command": (
                "/usr/local/cuda/bin/nvcc -O3 -DNDEBUG -std=c++20 "
                "\"--generate-code=arch=compute_110,"
                "code=[compute_110,sm_110]\" -Xcompiler=-fPIC "
                "--frandom-seed=magpie_tts_rt_plugins_v1 "
                "-Xcompiler=-Wall,-Wextra -Xcompiler=-Werror "
                "-c ${SOURCE_ROOT}/plugins/local_ar_plugins.cu"
            ),
            "link_command": (
                "/usr/bin/g++ -Wl,--no-undefined -Wl,--strip-all "
                "-Wl,--version-script=${SOURCE_ROOT}/cmake/"
                "magpie_tts_rt_plugins.map -shared "
                "-Wl,-soname,libmagpie_tts_rt_plugins.so.0 "
                f"-o {MODULE.EXPECTED_PLUGIN_FILENAME}"
            ),
            "cmake_definitions": [
                "CMAKE_BUILD_TYPE=Release",
                "CMAKE_CUDA_ARCHITECTURES=110",
                "MAGPIE_TTS_RT_WARNINGS_AS_ERRORS=ON",
            ],
        },
        "runtime_dependencies": {
            "cublas": valid_cublas_identity().to_json(),
        },
    }


class PackageSpecificationTests(unittest.TestCase):
    def test_reviewed_sofia_thor_spec_matches_locked_evidence(self) -> None:
        reviewed = MODULE.parse_package_spec(
            MODULE.load_json_file(
                PROJECT_ROOT
                / "reference"
                / "runtime-bundle-package-spec-v1.json"
            )
        )
        oracle_lock = MODULE.load_json_file(
            PROJECT_ROOT / "reference" / "oracle-lock.json"
        )
        self.assertIsInstance(oracle_lock, dict)
        assert isinstance(oracle_lock, dict)
        model = oracle_lock["model"]
        oracle_source = oracle_lock["oracle_source"]
        self.assertIsInstance(model, dict)
        self.assertIsInstance(oracle_source, dict)
        assert isinstance(model, dict)
        assert isinstance(oracle_source, dict)
        self.assertEqual(reviewed.source_model.model_id, model["model_id"])
        self.assertEqual(reviewed.source_model.version, model["version"])
        self.assertEqual(reviewed.source_model.revision, model["revision"])
        self.assertEqual(reviewed.source_model.source_sha256, model["sha256"])
        self.assertEqual(
            reviewed.export.source_revision,
            oracle_source["base_revision"],
        )

        tokenizer_receipt = MODULE.load_json_file(
            PROJECT_ROOT
            / "tests"
            / "frontend"
            / "fixtures"
            / "japanese-source-spans-v1.json"
        )
        self.assertIsInstance(tokenizer_receipt, dict)
        assert isinstance(tokenizer_receipt, dict)
        tokenizer_identity = MODULE.sha256_bytes(
            MODULE.canonical_json_bytes(
                MODULE.tokenizer_identity_projection(tokenizer_receipt)
            )
        )
        self.assertEqual(
            reviewed.tokenizer.identity_sha256,
            tokenizer_identity,
        )
        self.assertEqual(
            reviewed.export.baked_context_sha256,
            "aee360ccdeeb96a145efbce176ffe5d444215bdd5bc1a7bc8af8744df53d7b8d",
        )
        self.assertEqual(
            reviewed.local_ar.position_embedding.source_table_sha256,
            "1db63ebd4ceffba52e03cf67c9d186f3b7e38bb0c6eb9056a93ceeb55a4a695e",
        )
        self.assertFalse(reviewed.codec.eos_frame_is_audio)
        self.assertEqual(
            reviewed.codec.zero_frame_finalization,
            "control_marker_without_codec_invocation",
        )

    def test_valid_spec_parses(self) -> None:
        parsed = MODULE.parse_package_spec(valid_package_spec())
        self.assertEqual(parsed.schema_version, 1)
        self.assertEqual(parsed.export.format, MODULE.EXPORT_FORMAT)
        self.assertEqual(parsed.sampling.eos_token_id, 2017)
        self.assertEqual(
            parsed.sampling.forbidden_token_ids,
            (2016, 2018, 2019, 2020, 2021, 2022, 2023),
        )
        self.assertEqual(
            {engine.role for engine in parsed.destinations.engines},
            set(MODULE.ENGINE_ROLES),
        )

    def test_noncanonical_sampling_metadata_is_rejected(self) -> None:
        wrong_eos = valid_package_spec()
        wrong_eos["sampling"]["eos_token_id"] = 2
        with self.assertRaisesRegex(
            MODULE.PackageError, "canonical Sofia v1 sampling"
        ):
            MODULE.parse_package_spec(wrong_eos)

        wrong_forbidden = valid_package_spec()
        wrong_forbidden["sampling"]["forbidden_token_ids"] = [0, 1]
        with self.assertRaisesRegex(
            MODULE.PackageError, "must permit codec IDs 0..2015"
        ):
            MODULE.parse_package_spec(wrong_forbidden)

    def test_noncanonical_export_format_is_rejected(self) -> None:
        candidate = valid_package_spec()
        candidate["export"]["format"] = "another_export"
        with self.assertRaisesRegex(
            MODULE.PackageError, "magpie_tts_rt_export_v1"
        ):
            MODULE.parse_package_spec(candidate)

    def test_noncanonical_position_table_is_rejected(self) -> None:
        candidate = valid_package_spec()
        candidate["local_ar"]["position_embedding"][
            "source_table_sha256"
        ] = "a" * 64
        with self.assertRaisesRegex(
            MODULE.PackageError, "accepted \\[18,768\\] BF16 table"
        ):
            MODULE.parse_package_spec(candidate)

    def test_eos_frame_cannot_be_declared_as_audio(self) -> None:
        candidate = valid_package_spec()
        candidate["codec"]["eos_frame_is_audio"] = True
        with self.assertRaisesRegex(
            MODULE.PackageError, "AUDIO_EOS is a control token"
        ):
            MODULE.parse_package_spec(candidate)

    def test_zero_frame_finalization_policy_is_exact(self) -> None:
        candidate = valid_package_spec()
        candidate["codec"]["zero_frame_finalization"] = "invoke_tail_with_zero"
        with self.assertRaisesRegex(
            MODULE.PackageError, "without invoking NanoCodec"
        ):
            MODULE.parse_package_spec(candidate)

    def test_path_escape_is_rejected(self) -> None:
        candidate = valid_package_spec()
        candidate["destinations"]["plugin"] = "../plugin.so"
        with self.assertRaisesRegex(MODULE.PackageError, "forbidden"):
            MODULE.parse_package_spec(candidate)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(
                MODULE.PackageError, "duplicate JSON object key"
            ):
                MODULE.load_json_file(path)

    def test_duplicate_destination_is_rejected(self) -> None:
        candidate = valid_package_spec()
        candidate["destinations"]["plugin"] = candidate["destinations"][
            "export_receipt"
        ]
        with self.assertRaisesRegex(MODULE.PackageError, "must be unique"):
            MODULE.parse_package_spec(candidate)

    def test_engine_logical_name_must_match_its_role(self) -> None:
        candidate = valid_package_spec()
        candidate["destinations"]["engines"][0]["name"] = "wrong"
        with self.assertRaisesRegex(
            MODULE.PackageError, "requires text_encoder"
        ):
            MODULE.parse_package_spec(candidate)

    def test_destination_path_prefix_collision_is_rejected(self) -> None:
        candidate = valid_package_spec()
        candidate["destinations"]["plugin"] = "engines"
        with self.assertRaisesRegex(
            MODULE.PackageError, "cannot be another artifact's parent"
        ):
            MODULE.parse_package_spec(candidate)

    def test_manifest_path_cannot_be_used_as_a_directory(self) -> None:
        candidate = valid_package_spec()
        candidate["destinations"]["plugin"] = (
            "runtime-bundle-manifest.json/plugin.so"
        )
        with self.assertRaisesRegex(MODULE.PackageError, "paths are reserved"):
            MODULE.parse_package_spec(candidate)

    def test_license_destination_inventory_is_exact_and_ordered(self) -> None:
        candidate = valid_package_spec()
        candidate["destinations"]["licenses"][0]["role"] = "project_notice"
        with self.assertRaisesRegex(
            MODULE.PackageError,
            "canonical eight-role order",
        ):
            MODULE.parse_package_spec(candidate)

    def test_license_destination_cannot_alias_an_artifact(self) -> None:
        candidate = valid_package_spec()
        candidate["destinations"]["licenses"][0]["path"] = candidate[
            "destinations"
        ]["plugin"]
        with self.assertRaisesRegex(
            MODULE.PackageError,
            "artifact paths must be unique",
        ):
            MODULE.parse_package_spec(candidate)

    def test_source_model_receipt_hash_mismatch_fails_closed(self) -> None:
        candidate = valid_package_spec()["source_model"]
        source_model = MODULE.SourceModelSpec(
            candidate["model_id"],
            candidate["version"],
            candidate["revision"],
            candidate["source_sha256"],
        )
        receipt = valid_source_model_receipt(source_model)
        receipt["source_model"]["sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "source_model.source_sha256"
            ):
                MODULE.validate_source_model_receipt(path, source_model)

    def test_source_model_receipt_status_is_exact(self) -> None:
        candidate = valid_package_spec()["source_model"]
        source_model = MODULE.SourceModelSpec(
            candidate["model_id"],
            candidate["version"],
            candidate["revision"],
            candidate["source_sha256"],
        )
        receipt = valid_source_model_receipt(source_model)
        receipt["status"] = "accepted_revoked"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "status.*not accepted"
            ):
                MODULE.validate_source_model_receipt(path, source_model)

    def test_source_model_receipt_same_seed_replay_must_be_exact(
        self,
    ) -> None:
        candidate = valid_package_spec()["source_model"]
        source_model = MODULE.SourceModelSpec(
            candidate["model_id"],
            candidate["version"],
            candidate["revision"],
            candidate["source_sha256"],
        )
        receipt = valid_source_model_receipt(source_model)
        receipt["evidence"]["determinism"][0][
            "replay_pcm_f32le_sha256"
        ] = "f" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "same-seed replay is not exact",
            ):
                MODULE.validate_source_model_receipt(path, source_model)

    def test_export_receipt_model_mismatch_fails_closed(self) -> None:
        specification = valid_package_spec()
        parsed = MODULE.parse_package_spec(specification)
        export_candidate = specification["export"]
        receipt = {
            "schema_version": 1,
            "artifact_role": export_candidate["receipt_artifact_role"],
            "status": "accepted",
            "source": {
                "model_sha256": "f" * 64,
                "oracle_lock_sha256": "e" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "targets a different model"
            ):
                MODULE.validate_export_receipt(
                    path, parsed.export, parsed.source_model
                )

    def test_measured_export_receipt_is_rejected(self) -> None:
        parsed = MODULE.parse_package_spec(valid_package_spec())
        receipt = {
            "schema_version": 1,
            "artifact_role": parsed.export.receipt_artifact_role,
            "status": "measured-not-accepted",
            "source": {
                "model_sha256": parsed.source_model.source_sha256,
                "oracle_lock_sha256": "e" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "status is not accepted"
            ):
                MODULE.validate_export_receipt(
                    path, parsed.export, parsed.source_model
                )

    def test_consolidated_receipt_binds_every_staged_artifact(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        files = fake_bundle_files()
        receipt = valid_consolidated_receipt(specification, files)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            evidence = MODULE.validate_consolidated_export_receipt(
                path, specification, files
            )
        self.assertEqual(evidence.oracle_lock_sha256, "e" * 64)

    def test_consolidated_receipt_wrong_plan_fails_closed(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        files = fake_bundle_files()
        receipt = valid_consolidated_receipt(specification, files)
        receipt["engines"][3]["sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "staged artifact"
            ):
                MODULE.validate_consolidated_export_receipt(
                    path, specification, files
                )

    def test_consolidated_receipt_rejects_audio_eos_contract(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        files = fake_bundle_files()
        receipt = valid_consolidated_receipt(specification, files)
        receipt["eos_frame_is_audio"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "codec contract"
            ):
                MODULE.validate_consolidated_export_receipt(
                    path, specification, files
                )

    def test_consolidated_receipt_without_cublas_fails_closed(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        files = fake_bundle_files()
        receipt = valid_consolidated_receipt(specification, files)
        del receipt["runtime_dependencies"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "runtime_dependencies",
            ):
                MODULE.validate_consolidated_export_receipt(
                    path, specification, files
                )

    def test_consolidated_receipt_without_cuda_identity_fails_closed(
        self,
    ) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        files = fake_bundle_files()
        receipt = valid_consolidated_receipt(specification, files)
        del receipt["runtime_dependencies"]["cuda"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "keys mismatch.*cuda"
            ):
                MODULE.validate_consolidated_export_receipt(
                    path, specification, files
                )

    def test_consolidated_receipt_without_mode8_digest_fails_closed(
        self,
    ) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        files = fake_bundle_files()
        receipt = valid_consolidated_receipt(specification, files)
        del receipt["runtime_dependencies"]["mode8_class_table_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "keys mismatch.*mode8_class_table_sha256",
            ):
                MODULE.validate_consolidated_export_receipt(
                    path, specification, files
                )

    def test_plugin_build_receipt_authenticates_exact_plugin(self) -> None:
        plugin = fake_bundle_files()["plugin"]
        receipt = valid_plugin_build_receipt(plugin)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plugin-build.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            evidence = MODULE.validate_plugin_build_receipt(path, plugin)
        self.assertEqual(evidence.plugin_sha256, plugin.sha256)
        self.assertEqual(evidence.plugin_size_bytes, plugin.size_bytes)

    def test_plugin_build_receipt_rejects_wrong_plugin(self) -> None:
        plugin = fake_bundle_files()["plugin"]
        receipt = valid_plugin_build_receipt(plugin)
        receipt["artifact"]["sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plugin-build.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "does not authenticate",
            ):
                MODULE.validate_plugin_build_receipt(path, plugin)

    def test_plugin_build_receipt_rejects_mutated_cublas_identity(
        self,
    ) -> None:
        plugin = fake_bundle_files()["plugin"]
        receipt = valid_plugin_build_receipt(plugin)
        receipt["runtime_dependencies"]["cublas"]["lt_library"][
            "soname"
        ] = "libcublasLt.so.12"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plugin-build.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "libcublasLt.so.13",
            ):
                MODULE.validate_plugin_build_receipt(path, plugin)

    def test_plugin_build_receipt_requires_reproducible_seed(self) -> None:
        plugin = fake_bundle_files()["plugin"]
        receipt = valid_plugin_build_receipt(plugin)
        receipt["build"]["compile_command"] = receipt["build"][
            "compile_command"
        ].replace(
            "--frandom-seed=magpie_tts_rt_plugins_v1",
            "",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plugin-build.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "actual build fragment",
            ):
                MODULE.validate_plugin_build_receipt(path, plugin)

    def test_plugin_build_receipt_requires_canonical_sources_and_filename(
        self,
    ) -> None:
        plugin = fake_bundle_files()["plugin"]
        for mutation in ("source", "filename"):
            receipt = valid_plugin_build_receipt(plugin)
            if mutation == "source":
                receipt["source"][0]["path"] = "README.md"
                expected = "canonical plugin source inventory"
            else:
                receipt["artifact"]["filename"] = "plugin.so"
                expected = "expected libmagpie_tts_rt_plugins"
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "plugin-build.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(MODULE.PackageError, expected):
                    MODULE.validate_plugin_build_receipt(path, plugin)

    def test_plugin_source_tree_mutation_fails_closed(self) -> None:
        plugin = fake_bundle_files()["plugin"]
        receipt = valid_plugin_build_receipt(plugin)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, source_path in enumerate(
                MODULE.EXPECTED_PLUGIN_SOURCE_PATHS
            ):
                path = root / source_path
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"source-{index}".encode()
                path.write_bytes(payload)
                receipt["source"][index]["size_bytes"] = len(payload)
                receipt["source"][index]["sha256"] = MODULE.sha256_bytes(
                    payload
                )
            receipt_path = root / "plugin-build.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            evidence = MODULE.validate_plugin_build_receipt(
                receipt_path,
                plugin,
            )
            MODULE.validate_plugin_build_source_tree(evidence, root)
            mutated = root / MODULE.EXPECTED_PLUGIN_SOURCE_PATHS[3]
            mutated.write_bytes(b"x" * mutated.stat().st_size)
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "does not match build receipt",
            ):
                MODULE.validate_plugin_build_source_tree(evidence, root)

    def test_tokenizer_identity_matches_frontend_receipt(self) -> None:
        receipt_path = (
            PROJECT_ROOT
            / "tests"
            / "frontend"
            / "fixtures"
            / "japanese-source-spans-v1.json"
        )
        candidate = valid_package_spec()["tokenizer"]
        tokenizer = MODULE.TokenizerSpec(
            kind=candidate["kind"],
            tokenizer_vocabulary_size=(
                candidate["tokenizer_vocabulary_size"]
            ),
            text_embedding_rows=candidate["text_embedding_rows"],
            special_tokens=MODULE.TokenizerSpecialTokensSpec(
                **candidate["special_tokens"]
            ),
            identity_sha256=candidate["identity_sha256"],
        )
        MODULE.validate_tokenizer_identity_receipt(receipt_path, tokenizer)

    def test_tokenizer_mutation_fails_closed(self) -> None:
        original = json.loads(
            (
                PROJECT_ROOT
                / "tests"
                / "frontend"
                / "fixtures"
                / "japanese-source-spans-v1.json"
            ).read_text(encoding="utf-8")
        )
        candidate = valid_package_spec()["tokenizer"]
        tokenizer = MODULE.TokenizerSpec(
            kind=candidate["kind"],
            tokenizer_vocabulary_size=(
                candidate["tokenizer_vocabulary_size"]
            ),
            text_embedding_rows=candidate["text_embedding_rows"],
            special_tokens=MODULE.TokenizerSpecialTokensSpec(
                **candidate["special_tokens"]
            ),
            identity_sha256=candidate["identity_sha256"],
        )
        original["frontend_contract"]["japanese"]["token_table"][0]["token"] = "X"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.PackageError, "vocabulary hash is not canonical"
            ):
                MODULE.validate_tokenizer_identity_receipt(path, tokenizer)

    def test_tokenizer_row_contract_mutation_fails_closed(self) -> None:
        receipt_path = (
            PROJECT_ROOT
            / "tests"
            / "frontend"
            / "fixtures"
            / "japanese-source-spans-v1.json"
        )
        original = json.loads(receipt_path.read_text(encoding="utf-8"))
        candidate = valid_package_spec()["tokenizer"]
        tokenizer = MODULE.TokenizerSpec(
            kind=candidate["kind"],
            tokenizer_vocabulary_size=(
                candidate["tokenizer_vocabulary_size"]
            ),
            text_embedding_rows=candidate["text_embedding_rows"],
            special_tokens=MODULE.TokenizerSpecialTokensSpec(
                **candidate["special_tokens"]
            ),
            identity_sha256=candidate["identity_sha256"],
        )
        for mutate, expected in (
            (
                lambda receipt: receipt["frontend_contract"].__setitem__(
                    "text_embedding_rows", 3360
                ),
                "text embedding row count mismatch",
            ),
            (
                lambda receipt: receipt["frontend_contract"].__setitem__(
                    "eos_token_id", 3357
                ),
                "special-token IDs",
            ),
        ):
            mutated = json.loads(json.dumps(original))
            mutate(mutated)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "receipt.json"
                path.write_text(json.dumps(mutated), encoding="utf-8")
                with self.assertRaisesRegex(MODULE.PackageError, expected):
                    MODULE.validate_tokenizer_identity_receipt(
                        path, tokenizer
                    )

    def test_golden_fixture_is_bound_to_receipt_and_locks(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        fixture, receipt, evidence = valid_golden_contract(specification)
        MODULE.validate_golden_fixture(
            fixture, receipt, specification, evidence
        )

    def test_golden_receipt_rejects_noncanonical_finalization(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        fixture, receipt, evidence = valid_golden_contract(specification)
        mutated = replace(
            receipt,
            zero_frame_finalization="invoke_tail_with_zero",
        )
        with self.assertRaisesRegex(
            MODULE.PackageError, "codec stream contract"
        ):
            MODULE.validate_golden_fixture(
                fixture, mutated, specification, evidence
            )

    def test_golden_fixture_token_bytes_fail_closed(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        fixture, receipt, evidence = valid_golden_contract(specification)
        mutated = MODULE.GoldenFixture(
            schema_version=fixture.schema_version,
            fixture_id=fixture.fixture_id,
            prepared_token_ids=(
                *fixture.prepared_token_ids[:-1],
                fixture.prepared_token_ids[-1] - 1,
            ),
            seed=fixture.seed,
            tokenizer_identity_sha256=(
                fixture.tokenizer_identity_sha256
            ),
            oracle_lock_sha256=fixture.oracle_lock_sha256,
            normalized_text_sha256=fixture.normalized_text_sha256,
            token_ids_sha256=fixture.token_ids_sha256,
            baked_context_sha256=fixture.baked_context_sha256,
            expected=fixture.expected,
        )
        with self.assertRaisesRegex(
            MODULE.PackageError, "little-endian INT32"
        ):
            MODULE.validate_golden_fixture(
                mutated, receipt, specification, evidence
            )

    def test_golden_fixture_requires_one_terminal_eos(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        fixture, receipt, evidence = valid_golden_contract(specification)
        for token_ids, expected_error in (
            ((842,), "final prepared token"),
            ((842, 3358, 3358), "normal tokenizer rows"),
            ((842, 3357, 3358), "normal tokenizer rows"),
            ((842, 3359, 3358), "normal tokenizer rows"),
        ):
            token_hash = MODULE.prepared_token_ids_sha256(token_ids)
            mutated_fixture = replace(
                fixture,
                prepared_token_ids=token_ids,
                token_ids_sha256=token_hash,
            )
            mutated_receipt = replace(
                receipt,
                token_ids_sha256=token_hash,
            )
            with self.assertRaisesRegex(
                MODULE.PackageError, expected_error
            ):
                MODULE.validate_golden_fixture(
                    mutated_fixture,
                    mutated_receipt,
                    specification,
                    evidence,
                )

    def test_golden_fixture_must_fit_audio_frame_limit(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        fixture, receipt, evidence = valid_golden_contract(specification)
        codec_frame_count = specification.limits.maximum_audio_frames + 1
        pcm_sample_count = (
            codec_frame_count * specification.codec.hop_length_samples
        )
        mutated_fixture = replace(
            fixture,
            expected=replace(
                fixture.expected,
                codec_frame_count=codec_frame_count,
                pcm_sample_count=pcm_sample_count,
            ),
        )
        mutated_receipt = replace(
            receipt,
            codec_frame_count=codec_frame_count,
            sample_count=pcm_sample_count,
        )
        with self.assertRaisesRegex(
            MODULE.PackageError, "maximum_audio_frames"
        ):
            MODULE.validate_golden_fixture(
                mutated_fixture,
                mutated_receipt,
                specification,
                evidence,
            )

    def test_symlink_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"content")
            symlink = root / "link.bin"
            symlink.symlink_to(source)
            staging = root / "staging"
            staging.mkdir()
            with self.assertRaisesRegex(MODULE.PackageError, "symbolic link"):
                MODULE.copy_verified_source(
                    symlink, staging, "artifact.bin", "artifact"
                )

    def test_copied_artifact_is_exact_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"content")
            staging = root / "staging"
            staging.mkdir()
            copied = MODULE.copy_verified_source(
                source, staging, "nested/artifact.bin", "artifact"
            )
            destination = staging / "nested" / "artifact.bin"
            self.assertEqual(destination.read_bytes(), b"content")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
            self.assertEqual(copied.artifact.size_bytes, 7)
            self.assertEqual(
                copied.artifact.sha256,
                MODULE.sha256_bytes(b"content"),
            )

    def test_symlinked_source_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            source = actual / "source.bin"
            source.write_bytes(b"content")
            link = root / "linked"
            link.symlink_to(actual, target_is_directory=True)
            staging = root / "staging"
            staging.mkdir()
            with self.assertRaisesRegex(
                MODULE.PackageError, "must not contain symbolic links"
            ):
                MODULE.copy_verified_source(
                    link / "source.bin",
                    staging,
                    "artifact.bin",
                    "artifact",
                )

    def test_input_inventory_reports_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            source = actual / "source.bin"
            source.write_bytes(b"content")
            link = root / "linked"
            link.symlink_to(actual, target_is_directory=True)
            path = link / "source.bin"
            paths = MODULE.InputPaths(
                specification=path,
                source_model_acceptance_receipt=path,
                export_receipt=path,
                tokenizer_identity_receipt=path,
                plugin_build_receipt=path,
                plugin=path,
                golden_fixture=path,
                golden_receipt=path,
                manifest_validator=path,
                bundle_validator=path,
                licenses=tuple(
                    (role, path) for role in MODULE.LICENSE_ROLES
                ),
                engines=tuple(
                    (role, path) for role in MODULE.ENGINE_ROLES
                ),
            )
            problems = MODULE.inventory_problems(paths)
            self.assertEqual(len(problems), 25)
            self.assertTrue(
                all(
                    problem.startswith("SYMLINK_OR_DOTDOT_FORBIDDEN ")
                    for problem in problems
                )
            )

    def test_post_validator_snapshot_accepts_only_exact_files_and_parents(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "nested" / "artifact.bin"
            artifact_path.parent.mkdir()
            artifact_path.write_bytes(b"accepted")
            artifact = MODULE.FileArtifact(
                path="nested/artifact.bin",
                sha256=MODULE.sha256_bytes(b"accepted"),
                size_bytes=len(b"accepted"),
            )
            MODULE.verify_exact_bundle_snapshot(root, (artifact,))

    def test_post_validator_snapshot_rejects_same_size_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "artifact.bin"
            artifact_path.write_bytes(b"before")
            artifact = MODULE.FileArtifact(
                path="artifact.bin",
                sha256=MODULE.sha256_bytes(b"before"),
                size_bytes=len(b"before"),
            )
            artifact_path.write_bytes(b"after!")
            with self.assertRaisesRegex(
                MODULE.PackageError,
                "digest changed after validation",
            ):
                MODULE.verify_exact_bundle_snapshot(root, (artifact,))

    def test_post_validator_snapshot_rejects_extra_file_directory_and_symlink(
        self,
    ) -> None:
        for extra_kind in ("file", "directory", "symlink"):
            with self.subTest(extra_kind=extra_kind):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    artifact_path = root / "artifact.bin"
                    artifact_path.write_bytes(b"accepted")
                    artifact = MODULE.FileArtifact(
                        path="artifact.bin",
                        sha256=MODULE.sha256_bytes(b"accepted"),
                        size_bytes=len(b"accepted"),
                    )
                    extra = root / "extra"
                    if extra_kind == "file":
                        extra.write_bytes(b"extra")
                        expected_error = "entry set mismatch"
                    elif extra_kind == "directory":
                        extra.mkdir()
                        expected_error = "entry set mismatch"
                    else:
                        extra.symlink_to(artifact_path)
                        expected_error = "symbolic link"
                    with self.assertRaisesRegex(
                        MODULE.PackageError,
                        expected_error,
                    ):
                        MODULE.verify_exact_bundle_snapshot(root, (artifact,))


class EngineIntrospectionTests(unittest.TestCase):
    def test_dynamic_shapes_come_from_plan_profile(self) -> None:
        inspected = MODULE.inspect_engine(
            FakeTensorRt, FakeEngine(), "text_encoder"
        )
        self.assertEqual(
            inspected.inputs[0].shape,
            (1, -1),
        )
        self.assertEqual(
            inspected.profiles[0].input_shapes[0].maximum,
            (1, 512),
        )

    def test_main_decoder_position_is_device_execution_input(self) -> None:
        inspected = MODULE.inspect_engine(
            FakeTensorRt,
            FakeMainDecoderStepPositionEngine(),
            "main_decoder_step",
        )
        self.assertEqual(inspected.inputs[0].location, "device")
        self.assertFalse(inspected.inputs[0].shape_inference_io)
        self.assertEqual(inspected.profiles[0].input_values, ())

    def test_host_main_decoder_position_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            MODULE.PackageError,
            "position must be the authenticated DEVICE execution input",
        ):
            MODULE.inspect_engine(
                FakeTensorRt,
                FakeMainDecoderStepPositionEngine(
                    position_location=FakeTensorLocation.HOST,
                ),
                "main_decoder_step",
            )

    def test_shape_inference_main_decoder_position_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            MODULE.PackageError,
            "position must be the authenticated DEVICE execution input",
        ):
            MODULE.inspect_engine(
                FakeTensorRt,
                FakeMainDecoderStepPositionEngine(
                    position_shape_inference_io=True,
                ),
                "main_decoder_step",
            )

    def test_main_decoder_without_execution_status_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            MODULE.PackageError,
            "execution_status_in/out are mandatory",
        ):
            MODULE.inspect_engine(
                FakeTensorRt,
                FakeMainDecoderStepPositionEngine(
                    include_execution_status=False,
                ),
                "main_decoder_step",
            )

    def test_main_decoder_execution_status_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            MODULE.PackageError,
            "execution status must be scalar int32",
        ):
            MODULE.inspect_engine(
                FakeTensorRt,
                FakeMainDecoderStepPositionEngine(status_dtype="int64"),
                "main_decoder_step",
            )

    def test_codec_state_registry_is_derived_from_plans(self) -> None:
        engines = {
            "nanocodec_initial_4": fake_codec_engine(
                "nanocodec_initial_4",
                include_state_inputs=False,
                dynamic_frames=False,
            ),
            "nanocodec_steady_8": fake_codec_engine(
                "nanocodec_steady_8",
                include_state_inputs=True,
                dynamic_frames=False,
            ),
            "nanocodec_tail_1_8": fake_codec_engine(
                "nanocodec_tail_1_8",
                include_state_inputs=True,
                dynamic_frames=True,
            ),
        }
        codec = MODULE.CodecSpec(
            initial_engine_name="nanocodec_initial",
            steady_engine_name="nanocodec_steady",
            tail_engine_name="nanocodec_tail",
            sample_rate_hz=22050,
            hop_length_samples=1024,
            channels=1,
            pcm_format="f32le",
            stateful=True,
            initial_frames=4,
            steady_frames=8,
            tail_min_frames=1,
            tail_max_frames=8,
            eos_frame_is_audio=False,
            zero_frame_finalization=MODULE.ZERO_FRAME_FINALIZATION,
        )
        manifest_codec = MODULE.derive_codec(codec, engines)
        self.assertEqual(len(manifest_codec["state_bindings"]), 97)
        self.assertEqual(
            manifest_codec["state_bindings"][0]["logical_name"], "state_0"
        )
        self.assertFalse(manifest_codec["eos_frame_is_audio"])
        self.assertEqual(
            manifest_codec["zero_frame_finalization"],
            MODULE.ZERO_FRAME_FINALIZATION,
        )

    def test_kv_bindings_use_the_plan_export_names(self) -> None:
        specification = MODULE.parse_package_spec(valid_package_spec())
        manifest_kv = MODULE.derive_kv_cache(
            specification.kv_cache, fake_main_decoder_engines()
        )
        self.assertEqual(len(manifest_kv["layer_bindings"]), 12)
        first = manifest_kv["layer_bindings"][0]
        self.assertEqual(
            first["step_self_key_input"], "step_self_key_in_0"
        )
        self.assertEqual(
            first["step_self_key_output"], "step_self_key_out_0"
        )


class PluginDynamicContractTests(unittest.TestCase):
    @staticmethod
    def valid_dynamic_section() -> str:
        entries = "\n".join(
            f" 0x0000000000000001 (NEEDED) Shared library: [{dependency}]"
            for dependency in sorted(MODULE.EXPECTED_PLUGIN_NEEDED)
        )
        return (
            f"{entries}\n"
            " 0x000000000000000e (SONAME) Library soname: "
            f"[{MODULE.EXPECTED_PLUGIN_SONAME}]\n"
        )

    def test_exact_dependency_allowlist_is_accepted(self) -> None:
        contract = MODULE.parse_plugin_dynamic_section(
            self.valid_dynamic_section()
        )
        self.assertEqual(contract.soname, MODULE.EXPECTED_PLUGIN_SONAME)
        self.assertEqual(contract.needed, MODULE.EXPECTED_PLUGIN_NEEDED)

    def test_search_path_and_loader_injection_tags_are_rejected(self) -> None:
        for tag in sorted(MODULE.FORBIDDEN_PLUGIN_DYNAMIC_TAGS):
            with self.subTest(tag=tag):
                with self.assertRaises(MODULE.PackageError):
                    MODULE.parse_plugin_dynamic_section(
                        self.valid_dynamic_section()
                        + f" 0x1 ({tag}) Library {tag.lower()}: [/tmp]\n"
                    )

    def test_missing_extra_and_duplicate_dependencies_are_rejected(self) -> None:
        valid = self.valid_dynamic_section()
        with self.assertRaises(MODULE.PackageError):
            MODULE.parse_plugin_dynamic_section(
                valid.replace(
                    " 0x0000000000000001 (NEEDED) Shared library: "
                    "[libcudart.so.13]\n",
                    "",
                )
            )
        with self.assertRaises(MODULE.PackageError):
            MODULE.parse_plugin_dynamic_section(
                valid
                + " 0x0000000000000001 (NEEDED) Shared library: "
                "[libunexpected.so.1]\n"
            )
        first_dependency = sorted(MODULE.EXPECTED_PLUGIN_NEEDED)[0]
        with self.assertRaises(MODULE.PackageError):
            MODULE.parse_plugin_dynamic_section(
                valid
                + " 0x0000000000000001 (NEEDED) Shared library: "
                f"[{first_dependency}]\n"
            )

    def test_wrong_or_multiple_sonames_are_rejected(self) -> None:
        valid = self.valid_dynamic_section()
        with self.assertRaises(MODULE.PackageError):
            MODULE.parse_plugin_dynamic_section(
                valid.replace(
                    MODULE.EXPECTED_PLUGIN_SONAME,
                    "libmagpie_tts_rt_plugins.so.9",
                )
            )
        with self.assertRaises(MODULE.PackageError):
            MODULE.parse_plugin_dynamic_section(
                valid
                + " 0x000000000000000e (SONAME) Library soname: "
                f"[{MODULE.EXPECTED_PLUGIN_SONAME}]\n"
            )


if __name__ == "__main__":
    unittest.main()

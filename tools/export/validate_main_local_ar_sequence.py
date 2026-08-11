#!/usr/bin/env python3
"""Accept complete Text Encoder -> Main Decoder -> Local AR -> EOS sequences.

The validator authenticates accepted Text Encoder and Local AR artifacts plus
the measured Main Decoder plans, then reuses one TensorRT context per plan over
multiple locked Japanese fixtures. Acceptance requires exact generated codes
and EOS length for every fixture; token differences are durable evidence and
do not become an inferred tolerance.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Protocol

import torch

from alignment_controller import SofiaAlignmentController
from cublas_runtime_identity import collect_cublas_runtime_identity
from cuda_runtime_identity import collect_cuda_runtime_identity
from export_main_decoder import (
    DECODER_LAYERS,
    MODEL_WIDTH,
    PREFILL_PLAN,
    STEP_PLAN,
    DecoderFixture,
    canonical_json_bytes,
    import_tensorrt,
    load_fixture,
    publish_directory_no_replace,
    tensor_from_fixture,
)
from main_decoder_wrapper import PREFILL_LENGTH
from sequence_contract import (
    MAX_DECODER_STEPS,
    CodeMismatch,
    SequenceCodeTracker,
)
from validate_main_decoder_plans import (
    AuthenticatedCodecRestore,
    AuthenticatedPlanExport,
    JsonValue,
    authenticate_codec_restore,
    authenticate_plan_export,
    load_fixture_metadata,
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


LOCAL_AR_PLAN = "local-ar.plan"
LOCAL_AR_PLUGIN = "libmagpie_tts_rt_plugins.so"
LOCAL_AR_RECEIPT = "export-receipt.json"
LOCAL_AR_RECEIPT_CHECKSUM = "export-receipt.json.sha256"
TEXT_ENCODER_PLAN = "text_encoder.plan"
TEXT_ENCODER_RECEIPT = "plan-receipt.json"
TEXT_ENCODER_RECEIPT_CHECKSUM = "plan-receipt.json.sha256"
SEQUENCE_RECEIPT = "sequence-receipt.json"
SEQUENCE_RECEIPT_CHECKSUM = "sequence-receipt.json.sha256"
MINIMUM_DISTINCT_FIXTURES = 3


@dataclass(frozen=True)
class AuthenticatedLocalARExport:
    root: Path
    receipt_sha256: str
    oracle_lock_sha256: str
    locked_magpie_restore_sha256: str
    codec_restore: AuthenticatedCodecRestore
    source_fixture_manifest_sha256s: tuple[str, ...]
    plan: Path
    plan_sha256: str
    plugin: Path
    plugin_sha256: str
    tensorrt_version: str
    torch_cuda_build: str
    gpu_name: str
    gpu_compute_capability: tuple[int, int]


@dataclass(frozen=True)
class AuthenticatedTextEncoderExport:
    root: Path
    receipt_sha256: str
    oracle_lock_sha256: str
    locked_magpie_restore_sha256: str
    codec_restore: AuthenticatedCodecRestore
    source_fixture_manifest_sha256: str
    required_plugin_sha256: str
    plan: Path
    plan_sha256: str
    tensorrt_version: str
    torch_cuda_build: str
    gpu_name: str
    gpu_compute_capability: tuple[int, int]


class LocalARTensorsProtocol(Protocol):
    codec_tokens: torch.Tensor
    updated_rng_counter: torch.Tensor
    invalid_rows: torch.Tensor
    end_frame_index: torch.Tensor


class LocalARSessionProtocol(Protocol):
    def execute(
        self,
        decoder_hidden: torch.Tensor,
        seed: int,
        counter: int,
        unfinished: bool,
        finished: bool,
        forbid_eos: bool,
    ) -> LocalARTensorsProtocol: ...


class TensorRTPlanSession:
    """Reuse one TensorRT runtime, engine, and context on the current stream."""

    def __init__(self, *, tensorrt: ModuleType, plan_path: Path) -> None:
        self.tensorrt = tensorrt
        self.plan_path = plan_path.resolve(strict=True)
        self.logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        self.runtime = tensorrt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            self.plan_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(
                f"failed to deserialize TensorRT plan: {self.plan_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                f"failed to create TensorRT context: {self.plan_path}"
            )
        # The Main Decoder mode-8 plugin builds a CUDA Graph bank against the
        # first execution's exact tensor addresses.  A reused TensorRT context
        # therefore requires stable I/O storage; rebinding newly allocated
        # tensors on a later decoder step is a contract violation, not a
        # recoverable allocation detail.
        self._bound_inputs: dict[str, torch.Tensor] = {}
        self._outputs: dict[str, torch.Tensor] = {}

    def execute(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        input_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(
                self.engine.get_tensor_name(index)
            )
            == self.tensorrt.TensorIOMode.INPUT
        }
        if set(inputs) != input_names:
            raise RuntimeError(
                "TensorRT plan input set mismatch: "
                f"missing={sorted(input_names - set(inputs))}, "
                f"extra={sorted(set(inputs) - input_names)}"
            )
        dtype_map = {
            self.tensorrt.DataType.BF16: torch.bfloat16,
            self.tensorrt.DataType.BOOL: torch.bool,
            self.tensorrt.DataType.INT32: torch.int32,
            self.tensorrt.DataType.INT64: torch.int64,
        }
        bound_inputs: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            expected_dtype = dtype_map.get(self.engine.get_tensor_dtype(name))
            if expected_dtype is None or value.dtype != expected_dtype:
                raise RuntimeError(
                    f"TensorRT input {name} dtype mismatch: "
                    f"expected={expected_dtype}, actual={value.dtype}"
                )
            location = self.engine.get_tensor_location(name)
            if location == self.tensorrt.TensorLocation.HOST:
                source = value.detach().cpu().contiguous()
                bound = self._bound_inputs.get(name)
                if (
                    bound is None
                    or bound.device.type != "cpu"
                    or not bound.is_pinned()
                    or bound.dtype != source.dtype
                    or tuple(bound.shape) != tuple(source.shape)
                ):
                    bound = torch.empty_like(
                        source,
                        device="cpu",
                        pin_memory=True,
                    )
                bound.copy_(source)
                if bound.data_ptr() % 256 != 0:
                    raise RuntimeError(
                        f"TensorRT HOST input {name} is not 256-byte aligned"
                    )
            elif location == self.tensorrt.TensorLocation.DEVICE:
                if (
                    value.device.type != "cuda"
                    or value.device.index != torch.cuda.current_device()
                    or not value.is_contiguous()
                ):
                    raise RuntimeError(
                        f"TensorRT input {name} must be contiguous on "
                        "the current CUDA device"
                    )
                bound = self._bound_inputs.get(name)
                if (
                    bound is None
                    or bound.device != value.device
                    or bound.dtype != value.dtype
                    or tuple(bound.shape) != tuple(value.shape)
                ):
                    bound = torch.empty_like(value)
                bound.copy_(value)
            else:
                raise RuntimeError(
                    f"TensorRT input {name} has unsupported location "
                    f"{location}"
                )
            if -1 in tuple(self.engine.get_tensor_shape(name)):
                if not self.context.set_input_shape(name, tuple(bound.shape)):
                    raise RuntimeError(
                        f"failed to set TensorRT input shape: {name}"
                    )
            elif tuple(bound.shape) != tuple(self.engine.get_tensor_shape(name)):
                raise RuntimeError(
                    f"TensorRT input {name} shape mismatch: expected="
                    f"{tuple(self.engine.get_tensor_shape(name))}, "
                    f"actual={tuple(bound.shape)}"
                )
            if not self.context.set_tensor_address(name, bound.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT input: {name}")
            bound_inputs[name] = bound
        self._bound_inputs = bound_inputs
        missing = self.context.infer_shapes()
        if missing:
            raise RuntimeError(
                f"TensorRT shape inference is incomplete: {missing}"
            )

        outputs: dict[str, torch.Tensor] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if (
                self.engine.get_tensor_mode(name)
                == self.tensorrt.TensorIOMode.INPUT
            ):
                continue
            else:
                dtype = dtype_map.get(self.engine.get_tensor_dtype(name))
                if dtype is None:
                    raise RuntimeError(
                        f"unsupported TensorRT output dtype: {name}"
                    )
                output_shape = tuple(self.context.get_tensor_shape(name))
                value = self._outputs.get(name)
                if (
                    value is None
                    or value.device.type != "cuda"
                    or value.device.index != torch.cuda.current_device()
                    or value.dtype != dtype
                    or tuple(value.shape) != output_shape
                ):
                    value = torch.empty(
                        output_shape,
                        dtype=dtype,
                        device="cuda",
                    )
                outputs[name] = value
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT tensor: {name}")
        if not self.context.execute_async_v3(
            torch.cuda.current_stream().cuda_stream
        ):
            raise RuntimeError(
                f"TensorRT execution failed: {self.plan_path}"
            )
        self._outputs = outputs
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--text-encoder-export", type=Path, required=True)
    parser.add_argument("--main-export", type=Path, required=True)
    parser.add_argument("--local-ar-export", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        required=True,
        help="Repeat for three or more independently captured Japanese fixtures.",
    )
    parser.add_argument("--tensorrt-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def authenticate_text_encoder_export(
    path: Path,
) -> AuthenticatedTextEncoderExport:
    if path.is_symlink():
        raise RuntimeError(
            f"Text Encoder export must not be a symbolic link: {path}"
        )
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Text Encoder export is not a directory: {root}")
    receipt_path = root / TEXT_ENCODER_RECEIPT
    checksum_path = root / TEXT_ENCODER_RECEIPT_CHECKSUM
    if receipt_path.is_symlink() or checksum_path.is_symlink():
        raise RuntimeError(
            "Text Encoder receipt files must not be symbolic links"
        )
    receipt, receipt_payload = load_json(receipt_path)
    receipt_sha256 = sha256_bytes(receipt_payload)
    expected_checksum = (
        f"{receipt_sha256}  {TEXT_ENCODER_RECEIPT}\n".encode("ascii")
    )
    if checksum_path.read_bytes() != expected_checksum:
        raise RuntimeError("Text Encoder receipt checksum does not match")
    if (
        require_integer(
            receipt.get("schema_version"),
            "text receipt.schema_version",
        )
        != 1
    ):
        raise RuntimeError("unsupported Text Encoder receipt schema")
    if (
        require_string(
            receipt.get("artifact_role"),
            "text receipt.artifact_role",
        )
        != "text_encoder_plan"
    ):
        raise RuntimeError("Text Encoder export has the wrong artifact role")
    if (
        require_string(receipt.get("status"), "text receipt.status")
        != "measured-not-accepted"
    ):
        raise RuntimeError(
            "Text Encoder plan must be an explicitly unaccepted candidate"
        )

    expected_files = {
        TEXT_ENCODER_RECEIPT,
        TEXT_ENCODER_RECEIPT_CHECKSUM,
    }
    authenticated: dict[str, tuple[Path, str]] = {}
    for index, raw_record in enumerate(
        require_list(
            receipt.get("artifacts"),
            "text receipt.artifacts",
        )
    ):
        label = f"text receipt.artifacts[{index}]"
        record = require_mapping(raw_record, label)
        relative = require_safe_artifact_path(
            record.get("path"),
            f"{label}.path",
        )
        if relative in authenticated:
            raise RuntimeError(
                f"duplicate Text Encoder artifact: {relative}"
            )
        size_bytes = require_nonnegative_integer(
            record.get("size_bytes"),
            f"{label}.size_bytes",
        )
        expected_digest = require_sha256(
            record.get("sha256"),
            f"{label}.sha256",
        )
        unresolved = root / PurePosixPath(relative)
        if unresolved.is_symlink():
            raise RuntimeError(
                f"Text Encoder artifact is a symbolic link: {relative}"
            )
        artifact = unresolved.resolve(strict=True)
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise RuntimeError(
                f"invalid Text Encoder artifact: {relative}"
            )
        if artifact.stat().st_size != size_bytes:
            raise RuntimeError(
                f"Text Encoder artifact size mismatch: {relative}"
            )
        actual_digest = sha256_file(artifact)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Text Encoder artifact digest mismatch: {relative}"
            )
        authenticated[relative] = (artifact, actual_digest)
        expected_files.add(relative)

    actual_files: set[str] = set()
    actual_directories = {""}
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise RuntimeError(
                f"Text Encoder export contains a symbolic link: {relative}"
            )
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(
                f"Text Encoder export contains a non-file entry: {relative}"
            )
    expected_directories = {""}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if (
        actual_files != expected_files
        or actual_directories != expected_directories
    ):
        raise RuntimeError(
            "Text Encoder export entry set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}, "
            "extra_directories="
            f"{sorted(actual_directories - expected_directories)}"
        )
    try:
        plan, plan_sha256 = authenticated[TEXT_ENCODER_PLAN]
    except KeyError as error:
        raise RuntimeError(
            "Text Encoder export is missing its required plan"
        ) from error

    source = require_mapping(
        receipt.get("source"),
        "text receipt.source",
    )
    runtime = require_mapping(
        receipt.get("runtime"),
        "text receipt.runtime",
    )
    capability_values = require_list(
        runtime.get("gpu_compute_capability"),
        "text receipt.runtime.gpu_compute_capability",
    )
    if len(capability_values) != 2:
        raise RuntimeError(
            "Text Encoder GPU capability must have two components"
        )
    capability = tuple(
        require_nonnegative_integer(
            value,
            f"text receipt.runtime.gpu_compute_capability[{index}]",
        )
        for index, value in enumerate(capability_values)
    )
    return AuthenticatedTextEncoderExport(
        root=root,
        receipt_sha256=receipt_sha256,
        oracle_lock_sha256=require_sha256(
            source.get("oracle_lock_sha256"),
            "text receipt.source.oracle_lock_sha256",
        ),
        locked_magpie_restore_sha256=require_sha256(
            source.get("locked_magpie_restore_sha256"),
            "text receipt.source.locked_magpie_restore_sha256",
        ),
        codec_restore=authenticate_codec_restore(
            source,
            "text receipt.source",
        ),
        source_fixture_manifest_sha256=require_sha256(
            source.get("boundary_fixture_manifest_sha256"),
            "text receipt.source.boundary_fixture_manifest_sha256",
        ),
        required_plugin_sha256=require_sha256(
            source.get("plugin_sha256"),
            "text receipt.source.plugin_sha256",
        ),
        plan=plan,
        plan_sha256=plan_sha256,
        tensorrt_version=require_string(
            runtime.get("tensorrt"),
            "text receipt.runtime.tensorrt",
        ),
        torch_cuda_build=require_string(
            runtime.get("torch_cuda_build"),
            "text receipt.runtime.torch_cuda_build",
        ),
        gpu_name=require_string(
            runtime.get("gpu_name"),
            "text receipt.runtime.gpu_name",
        ),
        gpu_compute_capability=(capability[0], capability[1]),
    )


def authenticate_local_ar_export(
    path: Path,
) -> AuthenticatedLocalARExport:
    if path.is_symlink():
        raise RuntimeError(
            f"Local AR export must not be a symbolic link: {path}"
        )
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Local AR export is not a directory: {root}")
    receipt_path = root / LOCAL_AR_RECEIPT
    checksum_path = root / LOCAL_AR_RECEIPT_CHECKSUM
    if receipt_path.is_symlink() or checksum_path.is_symlink():
        raise RuntimeError("Local AR receipt files must not be symbolic links")
    receipt, receipt_payload = load_json(receipt_path)
    receipt_sha256 = sha256_bytes(receipt_payload)
    expected_checksum = (
        f"{receipt_sha256}  {LOCAL_AR_RECEIPT}\n".encode("ascii")
    )
    if checksum_path.read_bytes() != expected_checksum:
        raise RuntimeError("Local AR receipt checksum does not match")
    if require_integer(receipt.get("schema_version"), "receipt.schema_version") != 1:
        raise RuntimeError("unsupported Local AR receipt schema")
    if (
        require_string(receipt.get("artifact_role"), "receipt.artifact_role")
        != "local_ar_fixed_16"
    ):
        raise RuntimeError("Local AR export has the wrong artifact role")
    if (
        require_string(receipt.get("status"), "receipt.status")
        != "measured-not-accepted"
    ):
        raise RuntimeError(
            "Local AR plan must be an explicitly unaccepted candidate"
        )

    expected_files = {LOCAL_AR_RECEIPT, LOCAL_AR_RECEIPT_CHECKSUM}
    authenticated: dict[str, tuple[Path, str]] = {}
    for index, raw_record in enumerate(
        require_list(receipt.get("artifacts"), "receipt.artifacts")
    ):
        label = f"receipt.artifacts[{index}]"
        record = require_mapping(raw_record, label)
        relative = require_safe_artifact_path(
            record.get("path"),
            f"{label}.path",
        )
        if relative in authenticated:
            raise RuntimeError(f"duplicate Local AR artifact: {relative}")
        size_bytes = require_nonnegative_integer(
            record.get("size_bytes"),
            f"{label}.size_bytes",
        )
        expected_digest = require_sha256(
            record.get("sha256"),
            f"{label}.sha256",
        )
        unresolved = root / PurePosixPath(relative)
        if unresolved.is_symlink():
            raise RuntimeError(
                f"Local AR artifact is a symbolic link: {relative}"
            )
        artifact = unresolved.resolve(strict=True)
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise RuntimeError(f"invalid Local AR artifact: {relative}")
        if artifact.stat().st_size != size_bytes:
            raise RuntimeError(
                f"Local AR artifact size mismatch: {relative}"
            )
        actual_digest = sha256_file(artifact)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Local AR artifact digest mismatch: {relative}"
            )
        authenticated[relative] = (artifact, actual_digest)
        expected_files.add(relative)

    actual_files: set[str] = set()
    actual_directories = {""}
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise RuntimeError(
                f"Local AR export contains a symbolic link: {relative}"
            )
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(
                f"Local AR export contains a non-file entry: {relative}"
            )
    expected_directories = {""}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            "Local AR export entry set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}, "
            "extra_directories="
            f"{sorted(actual_directories - expected_directories)}"
        )
    try:
        plan, plan_sha256 = authenticated[LOCAL_AR_PLAN]
        plugin, plugin_sha256 = authenticated[LOCAL_AR_PLUGIN]
    except KeyError as error:
        raise RuntimeError(
            f"Local AR export is missing required artifact: {error.args[0]}"
        ) from error

    source = require_mapping(receipt.get("source"), "receipt.source")
    runtime = require_mapping(receipt.get("runtime"), "receipt.runtime")
    capability_values = require_list(
        runtime.get("gpu_compute_capability"),
        "receipt.runtime.gpu_compute_capability",
    )
    if len(capability_values) != 2:
        raise RuntimeError(
            "Local AR GPU capability must have two components"
        )
    capability = tuple(
        require_nonnegative_integer(
            value,
            f"receipt.runtime.gpu_compute_capability[{index}]",
        )
        for index, value in enumerate(capability_values)
    )
    manifest_values = require_list(
        source.get("boundary_fixture_manifest_sha256s"),
        "receipt.source.boundary_fixture_manifest_sha256s",
    )
    source_manifest_sha256s = tuple(
        require_sha256(
            value,
            f"receipt.source.boundary_fixture_manifest_sha256s[{index}]",
        )
        for index, value in enumerate(manifest_values)
    )
    if (
        len(source_manifest_sha256s) < MINIMUM_DISTINCT_FIXTURES
        or len(set(source_manifest_sha256s)) != len(source_manifest_sha256s)
        or tuple(sorted(source_manifest_sha256s)) != source_manifest_sha256s
    ):
        raise RuntimeError(
            "Local AR source fixture manifests must be a sorted unique "
            f"set of at least {MINIMUM_DISTINCT_FIXTURES} digests"
        )
    return AuthenticatedLocalARExport(
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
        source_fixture_manifest_sha256s=source_manifest_sha256s,
        plan=plan,
        plan_sha256=plan_sha256,
        plugin=plugin,
        plugin_sha256=plugin_sha256,
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


def require_generation_codes(fixture: DecoderFixture) -> torch.Tensor:
    record = fixture.tensors.get("generation.codes")
    if record is None:
        raise RuntimeError("fixture tensor is missing: generation.codes")
    if (
        record.dtype != "int64"
        or len(record.shape) != 3
        or record.shape[0:2] != (1, 8)
        or record.shape[2] < 1
        or record.shape[2] > MAX_DECODER_STEPS * 2
    ):
        raise RuntimeError(
            "generation.codes contract mismatch: "
            f"dtype={record.dtype}, shape={record.shape}"
        )
    return tensor_from_fixture(fixture, "generation.codes").cpu()


def write_int64_artifact(
    *,
    staging: Path,
    relative: Path,
    value: torch.Tensor,
) -> dict[str, JsonValue]:
    if value.dtype != torch.int64:
        raise RuntimeError(
            f"sequence artifact must be INT64: {relative}"
        )
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe sequence artifact path: {relative}")
    payload = (
        value.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<i8", copy=False)
        .tobytes()
    )
    path = staging / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative.as_posix(),
        "dtype": "int64",
        "shape": list(value.shape),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def mismatch_record(
    mismatch: CodeMismatch | None,
) -> JsonValue:
    if mismatch is None:
        return None
    return dataclasses.asdict(mismatch)


def text_condition_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, JsonValue]:
    if (
        actual.dtype != torch.bfloat16
        or expected.dtype != torch.bfloat16
        or tuple(actual.shape) != tuple(expected.shape)
    ):
        raise RuntimeError(
            "Text Encoder output contract mismatch: "
            f"actual={actual.dtype}/{tuple(actual.shape)}, "
            f"expected={expected.dtype}/{tuple(expected.shape)}"
        )
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    bit_mismatches = torch.count_nonzero(
        actual.view(torch.int16) != expected.view(torch.int16)
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.reshape(1, -1),
        expected_float.reshape(1, -1),
    )
    return {
        "value_count": actual.numel(),
        "bit_mismatch_count": int(bit_mismatches.item()),
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "p99_absolute_error": float(
            torch.quantile(difference.reshape(-1), 0.99).item()
        ),
        "cosine_similarity": float(cosine.item()),
    }


def run_fixture_sequence(
    *,
    fixture_path: Path,
    lock_path: Path,
    text_encoder_session: TensorRTPlanSession,
    prefill_session: TensorRTPlanSession,
    step_session: TensorRTPlanSession,
    local_ar_session: LocalARSessionProtocol,
    staging: Path,
    case_index: int,
) -> dict[str, JsonValue]:
    fixture = load_fixture(fixture_path, lock_path)
    metadata = load_fixture_metadata(fixture.root)
    expected_codes = require_generation_codes(fixture)
    for name, dtype, shape in (
        ("input.text_token_ids", "int32", (1, fixture.text_tokens)),
        ("text.mask", "bool", (1, fixture.text_tokens)),
        (
            "text.condition",
            "bf16",
            (1, fixture.text_tokens, MODEL_WIDTH),
        ),
    ):
        record = fixture.tensors.get(name)
        if record is None or record.dtype != dtype or record.shape != shape:
            raise RuntimeError(
                f"fixture tensor {name} mismatch: expected {dtype}/{shape}"
            )
    text_token_ids = tensor_from_fixture(
        fixture,
        "input.text_token_ids",
    ).contiguous()
    text_mask = tensor_from_fixture(fixture, "text.mask").contiguous()
    text_outputs = text_encoder_session.execute(
        {
            "text_token_ids": text_token_ids,
            "text_mask": text_mask,
        }
    )
    if set(text_outputs) != {"text_condition"}:
        raise RuntimeError(
            "Text Encoder output set mismatch: "
            f"{sorted(text_outputs)}"
        )
    conditional_condition = text_outputs["text_condition"].contiguous()
    expected_condition = tensor_from_fixture(
        fixture,
        "text.condition",
    )
    condition_metrics = text_condition_metrics(
        conditional_condition,
        expected_condition,
    )
    fixture_conditional = tensor_from_fixture(
        fixture,
        "cfg.conditional_condition",
    )
    fixture_unconditional = tensor_from_fixture(
        fixture,
        "cfg.unconditional_condition",
    )
    fixture_conditional_mask = tensor_from_fixture(
        fixture,
        "cfg.conditional_mask",
    )
    fixture_unconditional_mask = tensor_from_fixture(
        fixture,
        "cfg.unconditional_mask",
    )
    expected_unconditional_mask = torch.zeros_like(text_mask)
    expected_unconditional_mask[:, 0] = True
    if (
        not torch.equal(expected_condition, fixture_conditional)
        or torch.count_nonzero(fixture_unconditional).item() != 0
        or not torch.equal(text_mask, fixture_conditional_mask)
        or not torch.equal(
            expected_unconditional_mask,
            fixture_unconditional_mask,
        )
    ):
        raise RuntimeError(
            "fixture Text Encoder/CFG boundary is not the accepted contract"
        )
    condition = torch.cat(
        [
            conditional_condition,
            torch.zeros_like(conditional_condition),
        ],
        dim=0,
    ).contiguous()
    condition_mask = torch.cat(
        [text_mask, expected_unconditional_mask],
        dim=0,
    ).contiguous()
    prefill_outputs = prefill_session.execute(
        {
            "condition": condition,
            "condition_mask": condition_mask,
        }
    )
    self_keys = [
        prefill_outputs[f"prefill_self_key_{layer}"]
        for layer in range(DECODER_LAYERS)
    ]
    self_values = [
        prefill_outputs[f"prefill_self_value_{layer}"]
        for layer in range(DECODER_LAYERS)
    ]
    self_masks = [
        prefill_outputs[f"prefill_self_mask_{layer}"]
        for layer in range(DECODER_LAYERS)
    ]
    cross_keys = [
        prefill_outputs[f"prefill_cross_key_{layer}"]
        for layer in range(DECODER_LAYERS)
    ]
    cross_values = [
        prefill_outputs[f"prefill_cross_value_{layer}"]
        for layer in range(DECODER_LAYERS)
    ]
    decoder_hidden = prefill_outputs["last_hidden"][:, 0].contiguous()
    alignment = prefill_outputs["alignment"]
    alignment_controller = SofiaAlignmentController(
        text_length=fixture.text_tokens,
        device=alignment.device,
        dtype=alignment.dtype,
    )
    tracker = SequenceCodeTracker()
    attended_trace: list[int] = []
    execution_status = torch.zeros((), dtype=torch.int32, device="cuda")
    execution_status_check_count = 0

    while True:
        alignment_update = alignment_controller.update(alignment)
        attended_trace.append(int(alignment_update.attended.item()))
        local_outputs = local_ar_session.execute(
            decoder_hidden=decoder_hidden,
            seed=metadata.local_ar_seed,
            counter=tracker.rng_counter,
            unfinished=False,
            finished=False,
            forbid_eos=tracker.forbid_eos,
        )
        previous_codec_tokens = local_outputs.codec_tokens
        ended = tracker.accept_step(
            codec_tokens=local_outputs.codec_tokens,
            updated_rng_counter=local_outputs.updated_rng_counter,
            invalid_rows=local_outputs.invalid_rows,
            end_frame_index=local_outputs.end_frame_index,
        )
        if ended:
            break
        absolute_position = PREFILL_LENGTH + tracker.decoder_step - 1
        step_inputs: dict[str, torch.Tensor] = {
            "previous_codec_tokens": previous_codec_tokens,
            "position": torch.tensor(
                absolute_position,
                dtype=torch.int64,
                device="cuda",
            ),
            "execution_status_in": execution_status,
            "alignment_prior": alignment_update.prior,
            "condition_mask": condition_mask,
        }
        for layer in range(DECODER_LAYERS):
            step_inputs[f"step_self_key_in_{layer}"] = self_keys[layer]
            step_inputs[f"step_self_value_in_{layer}"] = self_values[layer]
            step_inputs[f"step_self_mask_in_{layer}"] = self_masks[layer]
            step_inputs[f"step_cross_key_in_{layer}"] = cross_keys[layer]
            step_inputs[f"step_cross_value_in_{layer}"] = cross_values[layer]
        step_outputs = step_session.execute(step_inputs)
        execution_status = step_outputs["execution_status_out"]
        execution_status_check_count += 1
        status_value = int(execution_status.item())
        if status_value != 0:
            raise RuntimeError(
                "Main Decoder execution status failed closed before the "
                f"next Local AR invocation: status={status_value}"
            )
        decoder_hidden = step_outputs["decoder_hidden"]
        alignment = step_outputs["alignment"]
        self_keys = [
            step_outputs[f"step_self_key_out_{layer}"]
            for layer in range(DECODER_LAYERS)
        ]
        self_values = [
            step_outputs[f"step_self_value_out_{layer}"]
            for layer in range(DECODER_LAYERS)
        ]
        self_masks = [
            step_outputs[f"step_self_mask_out_{layer}"]
            for layer in range(DECODER_LAYERS)
        ]

    comparison = tracker.compare(expected_codes)
    generated_codes = tracker.generated_codes()
    monotonic = all(
        left <= right
        for left, right in zip(attended_trace, attended_trace[1:])
    )
    if not monotonic:
        raise RuntimeError(
            f"alignment trace is not monotonic: {attended_trace}"
        )
    case_name = f"case-{case_index:03d}-{fixture.manifest_sha256[:12]}"
    artifacts = [
        write_int64_artifact(
            staging=staging,
            relative=Path("sequences") / case_name / "generated-codes.int64.bin",
            value=generated_codes,
        ),
        write_int64_artifact(
            staging=staging,
            relative=Path("sequences") / case_name / "attended-trace.int64.bin",
            value=torch.tensor(attended_trace, dtype=torch.int64),
        ),
    ]
    return {
        "case": case_name,
        "fixture_id": metadata.fixture_id,
        "fixture_manifest_sha256": fixture.manifest_sha256,
        "text": metadata.text,
        "language": metadata.language,
        "local_ar_seed": metadata.local_ar_seed,
        "text_encoder_metrics": condition_metrics,
        "code_exact": comparison.code_exact,
        "first_code_mismatch": mismatch_record(comparison.first_mismatch),
        "generated_frames": comparison.generated_frames,
        "expected_frames": comparison.expected_frames,
        "generated_codes_sha256": comparison.generated_codes_sha256,
        "expected_codes_sha256": comparison.expected_codes_sha256,
        "terminal_decoder_step": comparison.terminal_decoder_step,
        "terminal_end_frame_index": comparison.terminal_end_frame_index,
        "local_ar_invocations": comparison.local_ar_invocations,
        "final_rng_counter": comparison.final_rng_counter,
        "alignment_monotonic": True,
        "attended_trace_length": len(attended_trace),
        "main_execution_status_check_count": (
            execution_status_check_count
        ),
        "main_execution_status_all_zero": True,
        "artifacts": artifacts,
    }


def main() -> int:
    args = parse_args()
    fixture_paths = [path.resolve(strict=True) for path in args.fixture]
    if len(fixture_paths) < MINIMUM_DISTINCT_FIXTURES:
        raise RuntimeError(
            f"at least {MINIMUM_DISTINCT_FIXTURES} fixtures are required"
        )
    fixture_inputs_seen: set[tuple[str, int]] = set()
    fixture_manifest_sha256s: set[str] = set()
    for fixture_path in fixture_paths:
        metadata = load_fixture_metadata(fixture_path)
        fixture_input = (metadata.text, metadata.local_ar_seed)
        if fixture_input in fixture_inputs_seen:
            raise RuntimeError(
                "fixture text/seed input is duplicated: "
                f"{fixture_input!r}"
            )
        fixture_inputs_seen.add(fixture_input)
        manifest_path = fixture_path / "manifest.json"
        checksum_path = fixture_path / "manifest.json.sha256"
        manifest_sha256 = sha256_file(manifest_path.resolve(strict=True))
        expected_checksum = (
            f"{manifest_sha256}  manifest.json\n".encode("ascii")
        )
        if checksum_path.resolve(strict=True).read_bytes() != expected_checksum:
            raise RuntimeError(
                f"fixture manifest checksum mismatch: {fixture_path}"
            )
        if manifest_sha256 in fixture_manifest_sha256s:
            raise RuntimeError(
                f"duplicate fixture manifest: {manifest_sha256}"
            )
        fixture_manifest_sha256s.add(manifest_sha256)

    lock_path = args.lock.resolve(strict=True)
    lock_sha256 = sha256_file(lock_path)
    lock = require_mapping(json.loads(lock_path.read_text(encoding="utf-8")), "lock")
    text_encoder_export = authenticate_text_encoder_export(
        args.text_encoder_export
    )
    main_export = authenticate_plan_export(args.main_export)
    local_ar_export = authenticate_local_ar_export(args.local_ar_export)
    if (
        text_encoder_export.oracle_lock_sha256 != lock_sha256
        or main_export.oracle_lock_sha256 != lock_sha256
        or local_ar_export.oracle_lock_sha256 != lock_sha256
    ):
        raise RuntimeError(
            "plan export oracle lock mismatch: "
            f"current={lock_sha256}, "
            f"text_encoder={text_encoder_export.oracle_lock_sha256}, "
            f"main={main_export.oracle_lock_sha256}, "
            f"local_ar={local_ar_export.oracle_lock_sha256}"
        )
    expected_restore_sha256 = sha256_file(
        (Path(__file__).parent / "locked_magpie_restore.py").resolve(strict=True)
    )
    restore_sha256s = {
        text_encoder_export.locked_magpie_restore_sha256,
        main_export.locked_magpie_restore_sha256,
        local_ar_export.locked_magpie_restore_sha256,
    }
    if restore_sha256s != {expected_restore_sha256}:
        raise RuntimeError(
            "Text/Main/Local use different or non-current locked Magpie "
            f"restore helpers: {sorted(restore_sha256s)}"
        )
    codec_lock = require_mapping(lock.get("codec"), "lock.codec")
    expected_codec_restore = AuthenticatedCodecRestore(
        embedded_codec_model_id=require_string(
            codec_lock.get("model_id"), "lock.codec.model_id"
        ),
        codec_model_sha256=require_sha256(
            codec_lock.get("sha256"), "lock.codec.sha256"
        ),
        codec_model_size_bytes=require_nonnegative_integer(
            codec_lock.get("size_bytes"), "lock.codec.size_bytes"
        ),
    )
    if (
        text_encoder_export.codec_restore != expected_codec_restore
        or main_export.codec_restore != expected_codec_restore
        or local_ar_export.codec_restore != expected_codec_restore
    ):
        raise RuntimeError(
            "Text/Main/Local codec restore identities differ from the lock"
        )
    if (
        text_encoder_export.source_fixture_manifest_sha256
        != main_export.source_fixture_manifest_sha256
    ):
        raise RuntimeError(
            "Text Encoder and Main Decoder candidates use different "
            "canonical fixtures"
        )
    canonical_manifest_sha256 = (
        text_encoder_export.source_fixture_manifest_sha256
    )
    if canonical_manifest_sha256 not in fixture_manifest_sha256s:
        raise RuntimeError(
            "the canonical Text/Main fixture was not provided to sequence "
            "validation"
        )
    if (
        set(local_ar_export.source_fixture_manifest_sha256s)
        != fixture_manifest_sha256s
    ):
        raise RuntimeError(
            "Local AR candidate fixture set differs from sequence fixtures: "
            f"local={list(local_ar_export.source_fixture_manifest_sha256s)}, "
            f"sequence={sorted(fixture_manifest_sha256s)}"
        )
    if (
        text_encoder_export.required_plugin_sha256
        != local_ar_export.plugin_sha256
    ):
        raise RuntimeError(
            "Text Encoder candidate requires a different plugin: "
            f"text={text_encoder_export.required_plugin_sha256}, "
            f"local={local_ar_export.plugin_sha256}"
        )
    if (
        main_export.required_plugin_sha256
        != local_ar_export.plugin_sha256
    ):
        raise RuntimeError(
            "Main Decoder/Local AR plugin digest mismatch: "
            f"main={main_export.required_plugin_sha256}, "
            f"local={local_ar_export.plugin_sha256}"
        )

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        tensorrt = import_tensorrt(args.tensorrt_python_path)
        current_fingerprint = (
            tensorrt.__version__,
            str(torch.version.cuda),
            torch.cuda.get_device_name(0),
            tuple(torch.cuda.get_device_capability(0)),
        )
        main_fingerprint = (
            main_export.tensorrt_version,
            main_export.torch_cuda_build,
            main_export.gpu_name,
            main_export.gpu_compute_capability,
        )
        local_ar_fingerprint = (
            local_ar_export.tensorrt_version,
            local_ar_export.torch_cuda_build,
            local_ar_export.gpu_name,
            local_ar_export.gpu_compute_capability,
        )
        text_encoder_fingerprint = (
            text_encoder_export.tensorrt_version,
            text_encoder_export.torch_cuda_build,
            text_encoder_export.gpu_name,
            text_encoder_export.gpu_compute_capability,
        )
        if (
            current_fingerprint != text_encoder_fingerprint
            or current_fingerprint != main_fingerprint
            or current_fingerprint != local_ar_fingerprint
        ):
            raise RuntimeError(
                "sequence runtime fingerprint mismatch: "
                f"current={current_fingerprint}, "
                f"text_encoder={text_encoder_fingerprint}, "
                f"main={main_fingerprint}, "
                f"local_ar={local_ar_fingerprint}"
            )

        # Local AR imports TensorRT at module load time, so the explicit
        # TensorRT path must be installed in sys.path by import_tensorrt first.
        from run_local_ar_plan import LocalARSession

        # Register the custom creators before deserializing any plan. This
        # avoids a process-global plugin registry depending on construction
        # order.
        local_ar_session = LocalARSession(
            plan_path=local_ar_export.plan,
            plugin_path=local_ar_export.plugin,
        )
        cublas_identity = collect_cublas_runtime_identity(
            local_ar_session.plugin.library
        )
        cuda_identity = collect_cuda_runtime_identity()
        if cuda_identity != main_export.cuda_identity:
            raise RuntimeError(
                "sequence CUDA runtime identity differs from the Main "
                "Decoder export"
            )
        if cublas_identity != main_export.cublas_identity:
            raise RuntimeError(
                "sequence cuBLAS identity differs from the Main Decoder "
                "export"
            )
        text_encoder_session = TensorRTPlanSession(
            tensorrt=tensorrt,
            plan_path=text_encoder_export.plan,
        )
        prefill_session = TensorRTPlanSession(
            tensorrt=tensorrt,
            plan_path=main_export.prefill_plan,
        )
        step_session = TensorRTPlanSession(
            tensorrt=tensorrt,
            plan_path=main_export.step_plan,
        )
        cases: list[dict[str, JsonValue]] = []
        manifest_digests: set[str] = set()
        for index, fixture_path in enumerate(fixture_paths):
            case = run_fixture_sequence(
                fixture_path=fixture_path,
                lock_path=lock_path,
                text_encoder_session=text_encoder_session,
                prefill_session=prefill_session,
                step_session=step_session,
                local_ar_session=local_ar_session,
                staging=staging,
                case_index=index,
            )
            manifest_digest = require_sha256(
                case.get("fixture_manifest_sha256"),
                "case.fixture_manifest_sha256",
            )
            if manifest_digest in manifest_digests:
                raise RuntimeError(
                    f"duplicate fixture manifest: {manifest_digest}"
                )
            manifest_digests.add(manifest_digest)
            cases.append(case)

        artifact_records: list[JsonValue] = []
        for case in cases:
            artifact_records.extend(
                require_list(case.get("artifacts"), "case.artifacts")
            )
        exact_case_count = sum(
            require_mapping(case, "case").get("code_exact") is True
            for case in cases
        )
        all_codes_exact = exact_case_count == len(cases)
        status = "accepted" if all_codes_exact else "measured-not-accepted"
        reason = (
            "all predeclared Japanese fixtures passed through Text Encoder, "
            "Main Decoder, and Local AR with exact generated codes, EOS "
            "boundaries, RNG counters, and monotonic alignment"
            if all_codes_exact
            else (
                "one or more predeclared Japanese fixtures differed in "
                "generated codes or EOS boundary; the mismatch remains "
                "diagnostic evidence and is not converted to a tolerance"
            )
        )
        receipt: dict[str, JsonValue] = {
            "schema_version": 1,
            "artifact_role": "text_main_local_ar_sequence_validation",
            "status": status,
            "reason": reason,
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "source": {
                "validator_sha256": sha256_file(
                    Path(__file__).resolve(strict=True)
                ),
                "sequence_contract_sha256": sha256_file(
                    (
                        Path(__file__).parent / "sequence_contract.py"
                    ).resolve(strict=True)
                ),
                "alignment_controller_sha256": sha256_file(
                    (
                        Path(__file__).parent / "alignment_controller.py"
                    ).resolve(strict=True)
                ),
                "oracle_lock_sha256": lock_sha256,
                "locked_magpie_restore_sha256": expected_restore_sha256,
                "codec_restore": {
                    "embedded_codec_model_id": (
                        expected_codec_restore.embedded_codec_model_id
                    ),
                    "codec_model_sha256": (
                        expected_codec_restore.codec_model_sha256
                    ),
                    "codec_model_size_bytes": (
                        expected_codec_restore.codec_model_size_bytes
                    ),
                    "codec_resolution": "authenticated_local_file",
                    "use_scl_loss": False,
                    "network_resolution": False,
                },
                "text_encoder_export_receipt_sha256": (
                    text_encoder_export.receipt_sha256
                ),
                "text_encoder_plan_sha256": (
                    text_encoder_export.plan_sha256
                ),
                "main_export_receipt_sha256": main_export.receipt_sha256,
                "main_mode8_validation_receipt_sha256": (
                    main_export.mode8_validation_receipt_sha256
                ),
                "main_prefill_plan_sha256": main_export.prefill_plan_sha256,
                "main_step_plan_sha256": main_export.step_plan_sha256,
                "local_ar_export_receipt_sha256": (
                    local_ar_export.receipt_sha256
                ),
                "local_ar_plan_sha256": local_ar_export.plan_sha256,
                "local_ar_plugin_sha256": local_ar_export.plugin_sha256,
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda": cuda_identity.to_json(),
                "tensorrt": tensorrt.__version__,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
                "float32_matmul_precision": (
                    torch.get_float32_matmul_precision()
                ),
                "cuda_matmul_allow_tf32": (
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas": cublas_identity.to_json(),
                "mode8_class_table_sha256": (
                    main_export.mode8_class_table_sha256
                ),
            },
            "session_policy": {
                "ignore_finished_sentence_tracking": True,
                "unfinished": False,
                "finished": False,
                "minimum_generated_frames": 4,
                "frames_per_decoder_step": 2,
                "local_ar_positions_per_step": 16,
                "maximum_decoder_steps": MAX_DECODER_STEPS,
                "text_encoder_plan_and_context_reuse": True,
                "plan_and_context_reuse": True,
                "main_execution_status_recurrence": (
                    "int32-device-scalar-sticky-12-layer"
                ),
                "main_execution_status_checked_before_next_local_ar": True,
            },
            "fixture_count": len(cases),
            "exact_code_case_count": exact_case_count,
            "all_codes_exact": all_codes_exact,
            "fixtures": cases,
            "artifacts": artifact_records,
        }
        receipt_payload = canonical_json_bytes(receipt)
        (staging / SEQUENCE_RECEIPT).write_bytes(receipt_payload)
        (staging / SEQUENCE_RECEIPT_CHECKSUM).write_text(
            f"{sha256_bytes(receipt_payload)}  {SEQUENCE_RECEIPT}\n",
            encoding="ascii",
        )
        publish_directory_no_replace(staging, output)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    sys.exit(main())

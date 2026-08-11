#!/usr/bin/env python3
"""Export, build, inspect, and verify the locked stateful NanoCodec routes.

This command is deliberately an end-to-end gate rather than a thin ONNX
writer.  It authenticates the oracle inputs, reproduces every fixture codec
chunk with explicit state, checks all tail sizes against the locked NeMo
implementation, builds TensorRT plans with TF32 disabled, introspects every
plan binding/profile, executes plan parity, and only then publishes the
artifact directory with an atomic no-replace rename.
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
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_TOOLS = PROJECT_ROOT / "tools" / "oracle"
EXPORT_TOOLS = PROJECT_ROOT / "tools" / "export"
if str(ORACLE_TOOLS) not in sys.path:
    sys.path.insert(0, str(ORACLE_TOOLS))
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))

from nanocodec_contract import CANONICAL_STATE_BINDINGS  # noqa: E402
from validate_boundary_fixture import validate_boundary_fixture  # noqa: E402
from verify_oracle_lock import (  # noqa: E402
    FileExpectation,
    require_file,
    require_model_configs,
    require_source_checkout,
    sha256_file,
)


ONNX_OPSET = 20
EXPECTED_CODEBOOKS = 8
EXPECTED_STATE_TENSORS = 97
INITIAL_FRAMES = 4
STEADY_FRAMES = 8
TAIL_MIN_FRAMES = 1
TAIL_MAX_FRAMES = 8
SAMPLES_PER_FRAME = 1024
SAMPLE_RATE_HZ = 22050

# The export graph replaces the oracle's mutable preallocated work-buffer
# writes with functional Concat. CUDA convolution tactic/reduction order can
# therefore differ even with TF32 disabled. The measured single-fixture
# initial-4 maximum is 2.79e-6. This 1e-5 bound is a provisional structural
# equivalence gate, not the multi-fixture runtime acceptance tolerance.
PYTORCH_FP32_ATOL = 1.0e-5
PYTORCH_FP32_RTOL = 1.0e-5
TENSORRT_FP32_ATOL = 2.0e-4
TENSORRT_FP32_RTOL = 2.0e-4

ONNX_FILES = {
    "initial_4": "nanocodec-initial-4.onnx",
    "steady_8": "nanocodec-steady-8.onnx",
    "tail_1_8": "nanocodec-tail-1-8.onnx",
}
PLAN_FILES = {
    "initial_4": "nanocodec-initial-4.plan",
    "steady_8": "nanocodec-steady-8.plan",
    "tail_1_8": "nanocodec-tail-1-8.plan",
}
ENGINE_ROLES = {
    "initial_4": "nanocodec_initial_4",
    "steady_8": "nanocodec_steady_8",
    "tail_1_8": "nanocodec_tail_1_8",
}
RECEIPT_FILE_NAME = "export-receipt.json"
RECEIPT_DIGEST_FILE_NAME = "export-receipt.json.sha256"
CONTRACT_FILE_NAME = "nanocodec-contract.json"


@dataclass(frozen=True)
class FixtureTensor:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class NanoCodecFixture:
    root: Path
    manifest_sha256: str
    records: dict[str, FixtureTensor]
    frame_schedule: tuple[int, ...]
    valid_codec_frames: int


@dataclass(frozen=True)
class StateContract:
    logical_name: str
    shape: tuple[int, int, int]

    @property
    def input_binding(self) -> str:
        return f"state_in.{self.logical_name}"

    @property
    def output_binding(self) -> str:
        return f"state_out.{self.logical_name}"

    @property
    def fixture_suffix(self) -> str:
        return self.logical_name.replace("_", "-")


@dataclass(frozen=True)
class ParityMetric:
    tensor: str
    shape: tuple[int, ...]
    maximum_absolute_error: float
    mean_absolute_error: float
    p99_absolute_error: float
    maximum_relative_error: float
    signal_to_noise_ratio_db: float | None
    absolute_tolerance: float
    relative_tolerance: float
    exact: bool


@dataclass(frozen=True)
class TailOracleCase:
    frame_count: int
    codec_tokens: object
    pcm: object
    valid_sample_length: object
    states: tuple[object, ...]


@dataclass(frozen=True)
class ScheduleOracleCase:
    sequence: int
    frame_count: int
    codec_tokens: object
    pcm: object
    valid_sample_length: object
    states: tuple[object, ...]


def canonical_json_bytes(document: dict) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_json_document(path: Path, label: str) -> dict:
    try:
        document = json.loads(
            path.resolve(strict=True).read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return document


def verify_locked_inputs(args: argparse.Namespace) -> tuple[dict, str]:
    lock_path = args.lock.resolve(strict=True)
    lock = require_json_document(lock_path, "oracle lock")
    model_lock = lock["model"]
    codec_lock = lock["codec"]
    source_lock = lock["oracle_source"]
    acceptance_lock = lock["acceptance"]
    require_file(
        args.model,
        FileExpectation(model_lock["sha256"], model_lock["size_bytes"]),
        "Magpie model",
    )
    require_model_configs(
        args.model,
        model_lock["config_member_sha256"],
        model_lock["active_config_policy"],
        model_lock["active_config_sha256"],
    )
    require_file(
        args.codec_model,
        FileExpectation(codec_lock["sha256"], codec_lock["size_bytes"]),
        "NanoCodec model",
    )
    require_file(
        args.acceptance_receipt,
        FileExpectation(
            acceptance_lock["receipt_sha256"],
            acceptance_lock["receipt_size_bytes"],
        ),
        "acceptance receipt",
    )
    require_source_checkout(
        args.speech_root,
        source_lock["base_revision"],
        source_lock["files"],
        source_lock["optimized_source_bundle_sha256"],
    )
    trtexec = args.trtexec.resolve(strict=True)
    if not os.access(trtexec, os.X_OK):
        raise RuntimeError(f"trtexec is not executable: {trtexec}")
    return lock, sha256_file(lock_path)


def require_manifest_checksum(fixture_root: Path) -> tuple[dict, str]:
    manifest_path = fixture_root / "manifest.json"
    checksum_path = fixture_root / "manifest.json.sha256"
    manifest_digest = sha256_file(manifest_path.resolve(strict=True))
    checksum_text = checksum_path.resolve(strict=True).read_text(encoding="ascii")
    expected_line = f"{manifest_digest}  manifest.json\n"
    if checksum_text != expected_line:
        raise RuntimeError(
            "boundary fixture manifest checksum is not canonical or does not "
            "match manifest.json"
        )
    return (
        require_json_document(manifest_path, "boundary fixture manifest"),
        manifest_digest,
    )


def parse_fixture_tensor(
    fixture_root: Path,
    record: dict,
) -> FixtureTensor:
    name = record.get("name")
    relative_path = record.get("path")
    dtype = record.get("dtype")
    shape = record.get("shape")
    size_bytes = record.get("size_bytes")
    digest = record.get("sha256")
    if not isinstance(name, str) or not name:
        raise RuntimeError("boundary fixture tensor has an invalid name")
    if relative_path != f"tensors/{name}.bin":
        raise RuntimeError(f"boundary fixture tensor path mismatch: {name}")
    path = (fixture_root / relative_path).resolve(strict=True)
    if not path.is_relative_to(fixture_root):
        raise RuntimeError(f"boundary fixture tensor escapes its root: {name}")
    if dtype not in {"fp32", "int64"}:
        raise RuntimeError(
            f"NanoCodec fixture tensor {name} has unsupported dtype {dtype!r}"
        )
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        )
    ):
        raise RuntimeError(f"boundary fixture tensor {name} has invalid shape")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise RuntimeError(
            f"boundary fixture tensor {name} has invalid byte size"
        )
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(
            f"boundary fixture tensor {name} has invalid SHA-256"
        )
    if path.stat().st_size != size_bytes:
        raise RuntimeError(f"boundary fixture tensor {name} size mismatch")
    actual_digest = sha256_file(path)
    if actual_digest != digest:
        raise RuntimeError(
            f"boundary fixture tensor {name} SHA-256 mismatch: "
            f"expected {digest}, got {actual_digest}"
        )
    return FixtureTensor(
        name=name,
        path=path,
        dtype=dtype,
        shape=tuple(shape),
        size_bytes=size_bytes,
        sha256=digest,
    )


def verify_nanocodec_fixture(
    fixture_path: Path,
    lock: dict,
    lock_sha256: str,
) -> NanoCodecFixture:
    fixture_root = fixture_path.resolve(strict=True)
    if not fixture_root.is_dir():
        raise RuntimeError(
            f"boundary fixture is not a directory: {fixture_root}"
        )
    manifest, manifest_digest = require_manifest_checksum(fixture_root)
    expected_manifest_values = {
        "oracle_lock_sha256": lock_sha256,
        "model_sha256": lock["model"]["sha256"],
        "codec_model_sha256": lock["codec"]["sha256"],
        "acceptance_receipt_sha256": lock["acceptance"]["receipt_sha256"],
        "source_bundle_sha256": lock["oracle_source"][
            "optimized_source_bundle_sha256"
        ],
    }
    for key, expected in expected_manifest_values.items():
        actual = manifest.get(key)
        if actual != expected:
            raise RuntimeError(
                f"boundary fixture {key} mismatch: expected {expected}, "
                f"got {actual}"
            )
    runtime_policy = manifest.get("runtime")
    if not isinstance(runtime_policy, dict):
        raise RuntimeError("boundary fixture has no runtime precision policy")
    expected_runtime_policy = {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    actual_runtime_policy = {
        key: runtime_policy.get(key) for key in expected_runtime_policy
    }
    if actual_runtime_policy != expected_runtime_policy:
        raise RuntimeError(
            "boundary fixture runtime precision policy mismatch: "
            f"{actual_runtime_policy} != {expected_runtime_policy}"
        )

    contract = manifest.get("codec_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("boundary fixture has no codec contract")
    schedule_value = contract.get("frame_schedule")
    if (
        not isinstance(schedule_value, list)
        or len(schedule_value) < 2
        or any(
            not isinstance(frame_count, int)
            or frame_count < TAIL_MIN_FRAMES
            or frame_count > TAIL_MAX_FRAMES
            for frame_count in schedule_value
        )
    ):
        raise RuntimeError("boundary fixture codec schedule is invalid")
    schedule = tuple(schedule_value)
    if schedule[0] != INITIAL_FRAMES:
        raise RuntimeError(
            f"boundary fixture first codec chunk must be {INITIAL_FRAMES}"
        )
    if any(frame_count != STEADY_FRAMES for frame_count in schedule[1:-1]):
        raise RuntimeError(
            "boundary fixture has a non-terminal partial steady codec chunk"
        )
    valid_frames = contract.get("valid_codec_frames")
    if (
        not isinstance(valid_frames, int)
        or valid_frames != sum(schedule)
        or contract.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or contract.get("samples_per_frame") != SAMPLES_PER_FRAME
    ):
        raise RuntimeError("boundary fixture codec scalar contract is invalid")

    raw_records = manifest.get("tensors")
    if not isinstance(raw_records, list):
        raise RuntimeError("boundary fixture tensors must be a JSON array")
    records: dict[str, FixtureTensor] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise RuntimeError("boundary fixture contains an invalid tensor")
        name = raw_record.get("name")
        if not isinstance(name, str):
            raise RuntimeError("boundary fixture tensor name is invalid")
        if not (
            name.startswith("codec.")
            or name in {"generation.codes", "generation.code_lengths"}
        ):
            continue
        if name in records:
            raise RuntimeError(f"boundary fixture duplicates tensor {name}")
        records[name] = parse_fixture_tensor(fixture_root, raw_record)

    for sequence, frame_count in enumerate(schedule):
        prefix = f"codec.chunk_{sequence:03d}"
        expected = {
            f"{prefix}.codes": ("int64", (1, EXPECTED_CODEBOOKS, frame_count)),
            f"{prefix}.pcm": ("fp32", (1, frame_count * SAMPLES_PER_FRAME)),
            f"{prefix}.pcm_lengths": ("int64", (1,)),
        }
        for name, (dtype, shape) in expected.items():
            tensor = records.get(name)
            if tensor is None:
                raise RuntimeError(f"boundary fixture is missing tensor {name}")
            if tensor.dtype != dtype or tensor.shape != shape:
                raise RuntimeError(
                    f"boundary fixture tensor {name} contract mismatch: "
                    f"{tensor.dtype}/{tensor.shape} != {dtype}/{shape}"
                )
    complete_pcm = records.get("codec.complete_pcm")
    if (
        complete_pcm is None
        or complete_pcm.dtype != "fp32"
        or complete_pcm.shape != (1, valid_frames * SAMPLES_PER_FRAME)
    ):
        raise RuntimeError("boundary fixture complete PCM contract is invalid")
    return NanoCodecFixture(
        root=fixture_root,
        manifest_sha256=manifest_digest,
        records=records,
        frame_schedule=schedule,
        valid_codec_frames=valid_frames,
    )


def require_imported_nemo_sources(speech_root: Path, modules: tuple) -> None:
    expected_relative_paths = {
        "nemo.collections.tts.models.audio_codec": (
            "nemo/collections/tts/models/audio_codec.py"
        ),
        "nemo.collections.tts.modules.audio_codec_modules": (
            "nemo/collections/tts/modules/audio_codec_modules.py"
        ),
        "nemo.collections.tts.modules.streaming_codec": (
            "nemo/collections/tts/modules/streaming_codec.py"
        ),
    }
    root = speech_root.resolve(strict=True)
    for module in modules:
        module_name = module.__name__
        expected_relative = expected_relative_paths.get(module_name)
        if expected_relative is None:
            raise RuntimeError(
                f"unexpected NeMo module in source verification: {module_name}"
            )
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"NeMo module has no source file: {module_name}")
        actual = Path(module_file).resolve(strict=True)
        expected = (root / expected_relative).resolve(strict=True)
        if actual != expected:
            raise RuntimeError(
                f"NeMo source mismatch for {module_name}: "
                f"expected {expected}, got {actual}"
            )


def tensor_from_fixture(torch, numpy, tensor: FixtureTensor):
    dtype_by_name = {
        "fp32": "<f4",
        "int64": "<i8",
    }
    array = numpy.fromfile(tensor.path, dtype=dtype_by_name[tensor.dtype]).copy()
    value = torch.from_numpy(array).reshape(tensor.shape)
    return value.to(device="cuda")


def require_state_fixture_contract(
    fixture: NanoCodecFixture,
    state_contracts: tuple[StateContract, ...],
) -> None:
    capture_sequences = {
        0,
        1,
        len(fixture.frame_schedule) - 1,
    }
    for sequence in capture_sequences:
        for state in state_contracts:
            name = (
                f"codec.chunk_{sequence:03d}.state."
                f"{state.fixture_suffix}"
            )
            tensor = fixture.records.get(name)
            if tensor is None:
                raise RuntimeError(f"boundary fixture is missing state {name}")
            if tensor.dtype != "fp32" or tensor.shape != state.shape:
                raise RuntimeError(
                    f"boundary fixture state {name} contract mismatch: "
                    f"{tensor.dtype}/{tensor.shape} != fp32/{state.shape}"
                )
    state_records = tuple(
        name
        for name in fixture.records
        if ".state." in name and not name.endswith(".work-buffer")
    )
    expected_count = len(capture_sequences) * EXPECTED_STATE_TENSORS
    if len(state_records) != expected_count:
        raise RuntimeError(
            "boundary fixture persistent state count mismatch: "
            f"{len(state_records)} != {expected_count}"
        )


def compare_tensor(
    torch,
    actual,
    expected,
    *,
    name: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> ParityMetric:
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(
            f"{name} shape mismatch: {tuple(actual.shape)} != "
            f"{tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise RuntimeError(
            f"{name} dtype mismatch: {actual.dtype} != {expected.dtype}"
        )
    if not actual.is_floating_point():
        if not torch.equal(actual, expected):
            raise RuntimeError(f"{name} integer values differ")
        return ParityMetric(
            tensor=name,
            shape=tuple(actual.shape),
            maximum_absolute_error=0.0,
            mean_absolute_error=0.0,
            p99_absolute_error=0.0,
            maximum_relative_error=0.0,
            signal_to_noise_ratio_db=None,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            exact=True,
        )
    actual_float = actual.float()
    expected_float = expected.float()
    difference = torch.abs(actual_float - expected_float)
    allowed = absolute_tolerance + relative_tolerance * torch.abs(expected_float)
    failed = difference > allowed
    maximum_absolute = float(difference.max().item()) if difference.numel() else 0.0
    mean_absolute = float(difference.mean().item()) if difference.numel() else 0.0
    p99_absolute = (
        float(torch.quantile(difference.flatten(), 0.99).item())
        if difference.numel()
        else 0.0
    )
    denominator = torch.clamp(torch.abs(expected_float), min=1.0e-12)
    relative = difference / denominator
    maximum_relative = float(relative.max().item()) if relative.numel() else 0.0
    noise_power = (
        float(torch.mean(difference * difference).item())
        if difference.numel()
        else 0.0
    )
    signal_power = (
        float(torch.mean(expected_float * expected_float).item())
        if expected_float.numel()
        else 0.0
    )
    signal_to_noise_ratio_db = (
        10.0 * math.log10(signal_power / noise_power)
        if noise_power > 0.0 and signal_power > 0.0
        else None
    )
    if bool(torch.any(failed).item()):
        failed_count = int(torch.count_nonzero(failed).item())
        raise RuntimeError(
            f"{name} parity failed: failed_values={failed_count}, "
            f"max_abs={maximum_absolute}, max_rel={maximum_relative}, "
            f"atol={absolute_tolerance}, rtol={relative_tolerance}"
        )
    return ParityMetric(
        tensor=name,
        shape=tuple(actual.shape),
        maximum_absolute_error=maximum_absolute,
        mean_absolute_error=mean_absolute,
        p99_absolute_error=p99_absolute,
        maximum_relative_error=maximum_relative,
        signal_to_noise_ratio_db=signal_to_noise_ratio_db,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        exact=maximum_absolute == 0.0,
    )


def summarize_metrics(metrics: list[ParityMetric]) -> dict:
    if not metrics:
        raise RuntimeError("parity stage produced no metrics")
    return {
        "tensor_count": len(metrics),
        "maximum_absolute_error": max(
            metric.maximum_absolute_error for metric in metrics
        ),
        "maximum_mean_absolute_error": max(
            metric.mean_absolute_error for metric in metrics
        ),
        "maximum_p99_absolute_error": max(
            metric.p99_absolute_error for metric in metrics
        ),
        "maximum_relative_error": max(
            metric.maximum_relative_error for metric in metrics
        ),
        "minimum_signal_to_noise_ratio_db": min(
            (
                metric.signal_to_noise_ratio_db
                for metric in metrics
                if metric.signal_to_noise_ratio_db is not None
            ),
            default=None,
        ),
        "absolute_tolerance": max(
            metric.absolute_tolerance for metric in metrics
        ),
        "relative_tolerance": max(
            metric.relative_tolerance for metric in metrics
        ),
        "exact_tensor_count": sum(metric.exact for metric in metrics),
        "metrics": [asdict(metric) for metric in metrics],
        "passed": True,
    }


def load_locked_codec(torch, codec_path: Path, speech_root: Path):
    import nemo.collections.tts.models.audio_codec as audio_codec_module
    import nemo.collections.tts.modules.audio_codec_modules as codec_modules
    import nemo.collections.tts.modules.streaming_codec as streaming_codec
    from nemo.collections.tts.models.audio_codec import AudioCodecModel

    require_imported_nemo_sources(
        speech_root,
        (audio_codec_module, codec_modules, streaming_codec),
    )
    codec_config = AudioCodecModel.restore_from(
        str(codec_path.resolve(strict=True)),
        return_config=True,
    )
    if "use_scl_loss" not in codec_config:
        raise RuntimeError(
            "locked NanoCodec config has no explicit use_scl_loss field"
        )
    # The speaker encoder is a training-only loss dependency.  The accepted
    # Magpie restore path disables it before loading the same codec checkpoint.
    codec_config.use_scl_loss = False
    codec = AudioCodecModel.restore_from(
        str(codec_path.resolve(strict=True)),
        strict=False,
        override_config_path=codec_config,
        map_location="cpu",
    )
    codec.eval()
    codec.to("cuda")
    if int(codec.sample_rate) != SAMPLE_RATE_HZ:
        raise RuntimeError(
            f"NanoCodec sample rate changed: {codec.sample_rate}"
        )
    if int(codec.samples_per_frame) != SAMPLES_PER_FRAME:
        raise RuntimeError(
            "NanoCodec samples-per-frame changed: "
            f"{codec.samples_per_frame}"
        )
    if int(codec.num_codebooks) != EXPECTED_CODEBOOKS:
        raise RuntimeError(
            f"NanoCodec codebook count changed: {codec.num_codebooks}"
        )
    floating_dtypes = {
        parameter.dtype
        for parameter in codec.parameters()
        if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise RuntimeError(
            f"NanoCodec must remain FP32, got {floating_dtypes}"
        )
    for parameter in codec.audio_decoder.parameters():
        parameter.requires_grad_(False)
    weight_receipt = (
        streaming_codec.materialize_causal_hifigan_weight_norm_for_inference(
            codec.audio_decoder,
            expected_target_count=97,
        )
    )
    if weight_receipt.target_count != 97:
        raise RuntimeError("NanoCodec weight materialization is incomplete")
    return codec, weight_receipt, streaming_codec


def build_state_contracts(
    wrapper_state_specs: tuple,
) -> tuple[StateContract, ...]:
    result = tuple(
        StateContract(
            logical_name=spec.logical_name,
            shape=tuple(spec.shape),
        )
        for spec in wrapper_state_specs
    )
    if len(result) != EXPECTED_STATE_TENSORS:
        raise RuntimeError(
            f"NanoCodec state count changed: {len(result)}"
        )
    if len({state.logical_name for state in result}) != len(result):
        raise RuntimeError("NanoCodec state names are duplicated")
    expected = tuple(
        StateContract(
            logical_name=binding.logical_name,
            shape=binding.shape,
        )
        for binding in CANONICAL_STATE_BINDINGS
    )
    if result != expected:
        raise RuntimeError(
            "loaded NanoCodec state registry differs from the canonical v1 "
            "registry"
        )
    return result


def fixture_chunk_inputs(
    torch,
    numpy,
    fixture: NanoCodecFixture,
    sequence: int,
):
    return tensor_from_fixture(
        torch,
        numpy,
        fixture.records[f"codec.chunk_{sequence:03d}.codes"],
    )


def fixture_chunk_pcm(
    torch,
    numpy,
    fixture: NanoCodecFixture,
    sequence: int,
):
    return tensor_from_fixture(
        torch,
        numpy,
        fixture.records[f"codec.chunk_{sequence:03d}.pcm"],
    )


def fixture_chunk_length(
    torch,
    numpy,
    fixture: NanoCodecFixture,
    sequence: int,
):
    return tensor_from_fixture(
        torch,
        numpy,
        fixture.records[f"codec.chunk_{sequence:03d}.pcm_lengths"],
    )


def fixture_state(
    torch,
    numpy,
    fixture: NanoCodecFixture,
    state_contracts: tuple[StateContract, ...],
    sequence: int,
) -> tuple:
    return tuple(
        tensor_from_fixture(
            torch,
            numpy,
            fixture.records[
                f"codec.chunk_{sequence:03d}.state."
                f"{state.fixture_suffix}"
            ],
        )
        for state in state_contracts
    )


def source_state_from_explicit(
    torch,
    streaming_codec,
    codec,
    state_contracts: tuple[StateContract, ...],
    explicit_states: tuple,
):
    if len(explicit_states) != len(state_contracts):
        raise RuntimeError("explicit NanoCodec state tuple is incomplete")
    state_device = explicit_states[0].device
    if any(tensor.device != state_device for tensor in explicit_states):
        raise RuntimeError("explicit NanoCodec state spans multiple devices")
    decoder_state = streaming_codec._new_hifigan_state(codec.audio_decoder)
    streaming_codec._materialize_hifigan_state(
        decoder=codec.audio_decoder,
        state=decoder_state,
        batch_size=1,
        input_channels=32,
        input_frames=TAIL_MAX_FRAMES,
        device=state_device,
        dtype=torch.float32,
    )
    named = dict(
        streaming_codec._iter_named_hifigan_state_tensors(decoder_state)
    )
    for contract, tensor in zip(state_contracts, explicit_states):
        target = named.get(contract.logical_name)
        if target is None:
            raise RuntimeError(
                f"NeMo source state is missing {contract.logical_name}"
            )
        if tuple(target.shape) != contract.shape:
            raise RuntimeError(
                f"NeMo source state shape changed for {contract.logical_name}"
            )
        target.copy_(tensor)
    return streaming_codec.CausalCodecStreamingState(
        decoder=decoder_state,
        batch_size=1,
        codebook_count=EXPECTED_CODEBOOKS,
        codes_device=state_device,
        codes_dtype=torch.int64,
    )


def persistent_state_from_source(
    streaming_codec,
    decoder_state,
    state_contracts: tuple[StateContract, ...],
) -> tuple:
    if decoder_state is None:
        raise RuntimeError("NeMo source did not retain NanoCodec state")
    named = dict(
        streaming_codec._iter_named_hifigan_state_tensors(decoder_state)
    )
    result = []
    for contract in state_contracts:
        tensor = named.get(contract.logical_name)
        if tensor is None:
            raise RuntimeError(
                f"NeMo source state is missing {contract.logical_name}"
            )
        result.append(tensor.clone())
    return tuple(result)


def verify_codebook_table(
    torch,
    codec,
    codebook_table,
    codec_tokens,
) -> None:
    frame_count = codec_tokens.shape[-1]
    lengths = torch.full(
        (1,),
        frame_count,
        dtype=torch.int64,
        device="cuda",
    )
    with torch.inference_mode():
        expected = codec.dequantize(
            tokens=codec_tokens,
            tokens_len=lengths,
        )
        actual = torch.cat(
            [
                torch.nn.functional.embedding(
                    codec_tokens[:, codebook_index, :],
                    codebook_table[codebook_index],
                ).permute(0, 2, 1)
                for codebook_index in range(EXPECTED_CODEBOOKS)
            ],
            dim=1,
        )
    if not torch.equal(actual, expected):
        difference = float(torch.max(torch.abs(actual - expected)).item())
        raise RuntimeError(
            "materialized NanoCodec codebook table changed dequantization: "
            f"max_abs={difference}"
        )


def verify_pytorch_routes(
    torch,
    numpy,
    fixture: NanoCodecFixture,
    codec,
    streaming_codec,
    initial_wrapper,
    stateful_wrapper,
    state_contracts: tuple[StateContract, ...],
) -> tuple[
    dict,
    tuple[TailOracleCase, ...],
    tuple[ScheduleOracleCase, ...],
]:
    metrics: list[ParityMetric] = []
    first_codes = fixture_chunk_inputs(torch, numpy, fixture, 0)
    with torch.inference_mode():
        initial_outputs = initial_wrapper(first_codes)
    expected_initial_pcm = fixture_chunk_pcm(torch, numpy, fixture, 0)
    expected_initial_length = fixture_chunk_length(torch, numpy, fixture, 0)
    metrics.append(
        compare_tensor(
            torch,
            initial_outputs[0],
            expected_initial_pcm,
            name="pytorch.initial_4.pcm",
            absolute_tolerance=PYTORCH_FP32_ATOL,
            relative_tolerance=PYTORCH_FP32_RTOL,
        )
    )
    metrics.append(
        compare_tensor(
            torch,
            initial_outputs[1],
            expected_initial_length,
            name="pytorch.initial_4.valid_sample_length",
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )
    )
    expected_initial_state = fixture_state(
        torch,
        numpy,
        fixture,
        state_contracts,
        0,
    )
    for contract, actual, expected in zip(
        state_contracts,
        initial_outputs[2:],
        expected_initial_state,
    ):
        metrics.append(
            compare_tensor(
                torch,
                actual,
                expected,
                name=f"pytorch.initial_4.{contract.output_binding}",
                absolute_tolerance=PYTORCH_FP32_ATOL,
                relative_tolerance=PYTORCH_FP32_RTOL,
            )
        )

    schedule_cases = [
        ScheduleOracleCase(
            sequence=0,
            frame_count=fixture.frame_schedule[0],
            codec_tokens=first_codes,
            pcm=expected_initial_pcm,
            valid_sample_length=expected_initial_length,
            states=tuple(initial_outputs[2:]),
        )
    ]
    chained_pcm = [initial_outputs[0]]
    chained_state = tuple(initial_outputs[2:])
    for sequence in range(1, len(fixture.frame_schedule)):
        codes = fixture_chunk_inputs(torch, numpy, fixture, sequence)
        with torch.inference_mode():
            outputs = stateful_wrapper(codes, *chained_state)
        metrics.append(
            compare_tensor(
                torch,
                outputs[0],
                fixture_chunk_pcm(torch, numpy, fixture, sequence),
                name=f"pytorch.chunk_{sequence:03d}.pcm",
                absolute_tolerance=PYTORCH_FP32_ATOL,
                relative_tolerance=PYTORCH_FP32_RTOL,
            )
        )
        metrics.append(
            compare_tensor(
                torch,
                outputs[1],
                fixture_chunk_length(torch, numpy, fixture, sequence),
                name=f"pytorch.chunk_{sequence:03d}.valid_sample_length",
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            )
        )
        chained_state = tuple(outputs[2:])
        chained_pcm.append(outputs[0])
        schedule_cases.append(
            ScheduleOracleCase(
                sequence=sequence,
                frame_count=fixture.frame_schedule[sequence],
                codec_tokens=codes,
                pcm=fixture_chunk_pcm(torch, numpy, fixture, sequence),
                valid_sample_length=fixture_chunk_length(
                    torch,
                    numpy,
                    fixture,
                    sequence,
                ),
                states=chained_state,
            )
        )
        if sequence in {1, len(fixture.frame_schedule) - 1}:
            expected_state = fixture_state(
                torch,
                numpy,
                fixture,
                state_contracts,
                sequence,
            )
            for contract, actual, expected in zip(
                state_contracts,
                chained_state,
                expected_state,
            ):
                metrics.append(
                    compare_tensor(
                        torch,
                        actual,
                        expected,
                        name=(
                            f"pytorch.chunk_{sequence:03d}."
                            f"{contract.output_binding}"
                        ),
                        absolute_tolerance=PYTORCH_FP32_ATOL,
                        relative_tolerance=PYTORCH_FP32_RTOL,
                    )
                )

    metrics.append(
        compare_tensor(
            torch,
            torch.cat(chained_pcm, dim=-1),
            tensor_from_fixture(
                torch,
                numpy,
                fixture.records["codec.complete_pcm"],
            ),
            name="pytorch.closed_loop.complete_pcm",
            absolute_tolerance=PYTORCH_FP32_ATOL,
            relative_tolerance=PYTORCH_FP32_RTOL,
        )
    )

    base_state = expected_initial_state
    next_codes = fixture_chunk_inputs(torch, numpy, fixture, 1)
    source_decoder = streaming_codec.CausalCodecStreamingDecoder(
        codec_model=codec,
        codec_converter=None,
    )
    length_table = streaming_codec.preallocate_causal_codec_lengths(
        batch_size=1,
        max_codec_frames=TAIL_MAX_FRAMES,
        samples_per_frame=SAMPLES_PER_FRAME,
        device="cuda",
    )
    tail_cases: list[TailOracleCase] = []
    for frame_count in range(TAIL_MIN_FRAMES, TAIL_MAX_FRAMES + 1):
        codec_tokens = next_codes[:, :, :frame_count].contiguous()
        source_state = source_state_from_explicit(
            torch,
            streaming_codec,
            codec,
            state_contracts,
            base_state,
        )
        with torch.inference_mode():
            expected_pcm, expected_length = source_decoder.decode_new(
                codec_tokens,
                source_state,
                lengths=length_table[frame_count - 1],
            )
            expected_states = persistent_state_from_source(
                streaming_codec,
                source_state.decoder,
                state_contracts,
            )
            outputs = stateful_wrapper(codec_tokens, *base_state)
        metrics.append(
            compare_tensor(
                torch,
                outputs[0],
                expected_pcm,
                name=f"pytorch.tail_{frame_count}.pcm",
                absolute_tolerance=PYTORCH_FP32_ATOL,
                relative_tolerance=PYTORCH_FP32_RTOL,
            )
        )
        metrics.append(
            compare_tensor(
                torch,
                outputs[1],
                expected_length,
                name=f"pytorch.tail_{frame_count}.valid_sample_length",
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            )
        )
        for contract, actual, expected in zip(
            state_contracts,
            outputs[2:],
            expected_states,
        ):
            metrics.append(
                compare_tensor(
                    torch,
                    actual,
                    expected,
                    name=(
                        f"pytorch.tail_{frame_count}."
                        f"{contract.output_binding}"
                    ),
                    absolute_tolerance=PYTORCH_FP32_ATOL,
                    relative_tolerance=PYTORCH_FP32_RTOL,
                )
            )
        tail_cases.append(
            TailOracleCase(
                frame_count=frame_count,
                codec_tokens=codec_tokens,
                pcm=expected_pcm,
                valid_sample_length=expected_length,
                states=expected_states,
            )
        )
    return (
        summarize_metrics(metrics),
        tuple(tail_cases),
        tuple(schedule_cases),
    )


def export_onnx_graph(
    torch,
    *,
    wrapper,
    example_inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    output_path: Path,
) -> None:
    if len(set(input_names)) != len(input_names):
        raise RuntimeError("NanoCodec ONNX input names are duplicated")
    if len(set(output_names)) != len(output_names):
        raise RuntimeError("NanoCodec ONNX output names are duplicated")
    wrapper.eval()
    with torch.inference_mode():
        outputs = wrapper(*example_inputs)
    if len(outputs) != len(output_names):
        raise RuntimeError(
            f"NanoCodec wrapper emitted {len(outputs)} outputs for "
            f"{len(output_names)} names"
        )
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(output_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=False,
        external_data=False,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def onnx_dimensions(value_info) -> tuple[int | str, ...]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            raise RuntimeError(
                f"ONNX tensor {value_info.name} has an unknown dimension"
            )
    return tuple(result)


def expected_tensor_contract(
    state_contracts: tuple[StateContract, ...],
    *,
    route: str,
) -> tuple[dict[str, tuple[str, tuple]], dict[str, tuple[str, tuple]]]:
    if route == "initial_4":
        frame_dimension: int | str = INITIAL_FRAMES
        pcm_dimension: int | str = INITIAL_FRAMES * SAMPLES_PER_FRAME
        inputs = {
            "codec_tokens": (
                "int64",
                (1, EXPECTED_CODEBOOKS, frame_dimension),
            )
        }
    elif route == "steady_8":
        frame_dimension = STEADY_FRAMES
        pcm_dimension = STEADY_FRAMES * SAMPLES_PER_FRAME
        inputs = {
            "codec_tokens": (
                "int64",
                (1, EXPECTED_CODEBOOKS, frame_dimension),
            ),
            **{
                state.input_binding: ("fp32", state.shape)
                for state in state_contracts
            },
        }
    elif route == "tail_1_8":
        frame_dimension = "codec_frames"
        pcm_dimension = "pcm_samples"
        inputs = {
            "codec_tokens": (
                "int64",
                (1, EXPECTED_CODEBOOKS, frame_dimension),
            ),
            **{
                state.input_binding: ("fp32", state.shape)
                for state in state_contracts
            },
        }
    else:
        raise ValueError(f"unknown NanoCodec route: {route}")
    outputs = {
        "pcm": ("fp32", (1, pcm_dimension)),
        "valid_sample_length": ("int64", (1,)),
        **{
            state.output_binding: ("fp32", state.shape)
            for state in state_contracts
        },
    }
    return inputs, outputs


def verify_onnx_contract(
    onnx,
    onnx_path: Path,
    *,
    route: str,
    state_contracts: tuple[StateContract, ...],
) -> dict:
    model = onnx.load(str(onnx_path.resolve(strict=True)), load_external_data=True)
    onnx.checker.check_model(model, full_check=False)
    opsets = {
        entry.domain or "ai.onnx": entry.version
        for entry in model.opset_import
    }
    if opsets != {"ai.onnx": ONNX_OPSET}:
        raise RuntimeError(
            f"NanoCodec ONNX opset mismatch: {opsets}"
        )
    if any(node.domain for node in model.graph.node):
        raise RuntimeError("NanoCodec ONNX contains a non-standard operator")
    if any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
    ):
        raise RuntimeError("NanoCodec ONNX external initializer data is forbidden")
    expected_inputs, expected_outputs = expected_tensor_contract(
        state_contracts,
        route=route,
    )
    actual_inputs = {value.name: value for value in model.graph.input}
    actual_outputs = {value.name: value for value in model.graph.output}
    if set(actual_inputs) != set(expected_inputs):
        raise RuntimeError(
            f"NanoCodec {route} ONNX input bindings differ: "
            f"missing={sorted(set(expected_inputs) - set(actual_inputs))}, "
            f"extra={sorted(set(actual_inputs) - set(expected_inputs))}"
        )
    if set(actual_outputs) != set(expected_outputs):
        raise RuntimeError(
            f"NanoCodec {route} ONNX output bindings differ: "
            f"missing={sorted(set(expected_outputs) - set(actual_outputs))}, "
            f"extra={sorted(set(actual_outputs) - set(expected_outputs))}"
        )
    dtype_by_name = {
        "fp32": onnx.TensorProto.FLOAT,
        "int64": onnx.TensorProto.INT64,
    }
    for values, expected in (
        (actual_inputs, expected_inputs),
        (actual_outputs, expected_outputs),
    ):
        for name, (dtype, shape) in expected.items():
            value = values[name]
            actual_dtype = value.type.tensor_type.elem_type
            if actual_dtype != dtype_by_name[dtype]:
                raise RuntimeError(
                    f"NanoCodec ONNX {name} dtype mismatch: "
                    f"{actual_dtype} != {dtype_by_name[dtype]}"
                )
            actual_shape = onnx_dimensions(value)
            if actual_shape != shape:
                raise RuntimeError(
                    f"NanoCodec ONNX {name} shape mismatch: "
                    f"{actual_shape} != {shape}"
                )
    return {
        "graph_nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "input_count": len(actual_inputs),
        "output_count": len(actual_outputs),
    }


def build_plan(
    *,
    trtexec: Path,
    route: str,
    onnx_path: Path,
    plan_path: Path,
    log_path: Path,
    layer_info_path: Path,
) -> dict:
    command = [
        str(trtexec.resolve(strict=True)),
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        "--noTF32",
        "--stronglyTyped",
        "--skipInference",
        "--builderOptimizationLevel=5",
        "--memPoolSize=workspace:4096MiB",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_path}",
        "--verbose",
    ]
    if route == "tail_1_8":
        command.extend(
            (
                "--minShapes=codec_tokens:1x8x1",
                "--optShapes=codec_tokens:1x8x4",
                "--maxShapes=codec_tokens:1x8x8",
            )
        )
    started = time.monotonic()
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed_seconds = time.monotonic() - started
    log_path.write_text(
        "$ "
        + " ".join(command)
        + "\n"
        + f"# elapsed_seconds={elapsed_seconds:.6f}\n"
        + result.stdout,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TensorRT build failed for {route}; inspect {log_path}"
        )
    if not plan_path.is_file() or plan_path.stat().st_size == 0:
        raise RuntimeError(f"TensorRT build produced no plan for {route}")
    if not layer_info_path.is_file() or layer_info_path.stat().st_size == 0:
        raise RuntimeError(
            f"TensorRT build produced no layer introspection for {route}"
        )
    return {
        "elapsed_seconds": elapsed_seconds,
        "command": command,
        "log_sha256": sha256_file(log_path),
        "layer_info_sha256": sha256_file(layer_info_path),
    }


def import_tensorrt():
    system_packages = Path("/usr/lib/python3.12/dist-packages")
    if not system_packages.is_dir():
        raise RuntimeError(
            f"system TensorRT Python path is missing: {system_packages}"
        )
    if str(system_packages) not in sys.path:
        sys.path.append(str(system_packages))
    import tensorrt

    return tensorrt


def trt_dtype_name(trt, dtype) -> str:
    mapping = {
        trt.float32: "fp32",
        trt.int64: "int64",
    }
    result = mapping.get(dtype)
    if result is None:
        raise RuntimeError(f"unsupported TensorRT dtype: {dtype}")
    return result


def inspect_plan(
    trt,
    plan_path: Path,
    *,
    route: str,
    state_contracts: tuple[StateContract, ...],
):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(
        plan_path.resolve(strict=True).read_bytes()
    )
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize {plan_path}")
    tensors: list[dict] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            mode_name = "input"
        elif mode == trt.TensorIOMode.OUTPUT:
            mode_name = "output"
        else:
            raise RuntimeError(f"TensorRT tensor {name} has unknown I/O mode")
        tensors.append(
            {
                "name": name,
                "mode": mode_name,
                "dtype": trt_dtype_name(trt, engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
                "location": str(engine.get_tensor_location(name)),
                "shape_inference_io": bool(
                    engine.is_shape_inference_io(name)
                ),
            }
        )
    profile: dict | None = None
    if route == "tail_1_8":
        profile_shapes = engine.get_tensor_profile_shape(
            "codec_tokens",
            0,
        )
        profile = {
            "codec_tokens": {
                "min": list(profile_shapes[0]),
                "opt": list(profile_shapes[1]),
                "max": list(profile_shapes[2]),
            }
        }
    metadata = {
        "route": route,
        "num_io_tensors": engine.num_io_tensors,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "device_memory_size_bytes": int(engine.device_memory_size_v2),
        "tensors": tensors,
        "profile": profile,
    }
    verify_inspected_plan_contract(
        metadata,
        route=route,
        state_contracts=state_contracts,
    )
    return runtime, engine, metadata


def verify_inspected_plan_contract(
    metadata: dict,
    *,
    route: str,
    state_contracts: tuple[StateContract, ...],
) -> None:
    expected_inputs, expected_outputs = expected_tensor_contract(
        state_contracts,
        route=route,
    )
    expected = {
        **{
            name: {"mode": "input", "dtype": dtype, "shape": shape}
            for name, (dtype, shape) in expected_inputs.items()
        },
        **{
            name: {"mode": "output", "dtype": dtype, "shape": shape}
            for name, (dtype, shape) in expected_outputs.items()
        },
    }
    raw_tensors = metadata.get("tensors")
    if not isinstance(raw_tensors, list):
        raise RuntimeError("TensorRT plan introspection has no tensor list")
    actual = {}
    for tensor in raw_tensors:
        if not isinstance(tensor, dict) or not isinstance(
            tensor.get("name"), str
        ):
            raise RuntimeError("TensorRT plan introspection tensor is invalid")
        name = tensor["name"]
        if name in actual:
            raise RuntimeError(f"TensorRT plan duplicates tensor {name}")
        actual[name] = tensor
    if set(actual) != set(expected):
        raise RuntimeError(
            f"TensorRT {route} bindings differ: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for name, contract in expected.items():
        tensor = actual[name]
        actual_shape = tuple(tensor.get("shape", ()))
        expected_shape = contract["shape"]
        normalized_expected_shape = tuple(
            -1 if isinstance(dimension, str) else dimension
            for dimension in expected_shape
        )
        if (
            tensor.get("mode") != contract["mode"]
            or tensor.get("dtype") != contract["dtype"]
            or actual_shape != normalized_expected_shape
        ):
            raise RuntimeError(
                f"TensorRT {route} tensor {name} contract mismatch: "
                f"{tensor} != {contract}"
            )
    expected_io_count = len(expected)
    if metadata.get("num_io_tensors") != expected_io_count:
        raise RuntimeError(
            f"TensorRT {route} I/O count mismatch: "
            f"{metadata.get('num_io_tensors')} != {expected_io_count}"
        )
    if metadata.get("num_optimization_profiles") != 1:
        raise RuntimeError(
            f"TensorRT {route} must contain exactly one profile"
        )
    expected_profile = (
        {
            "codec_tokens": {
                "min": [1, EXPECTED_CODEBOOKS, TAIL_MIN_FRAMES],
                "opt": [1, EXPECTED_CODEBOOKS, INITIAL_FRAMES],
                "max": [1, EXPECTED_CODEBOOKS, TAIL_MAX_FRAMES],
            }
        }
        if route == "tail_1_8"
        else None
    )
    if metadata.get("profile") != expected_profile:
        raise RuntimeError(
            f"TensorRT {route} profile mismatch: "
            f"{metadata.get('profile')} != {expected_profile}"
        )


def torch_dtype_for_trt(torch, trt, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.int64: torch.int64,
    }
    result = mapping.get(dtype)
    if result is None:
        raise RuntimeError(f"unsupported TensorRT execution dtype: {dtype}")
    return result


def execute_plan(
    torch,
    trt,
    engine,
    inputs: dict[str, object],
) -> tuple[dict[str, object], object]:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT execution context creation failed")
    expected_inputs = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index))
        == trt.TensorIOMode.INPUT
    }
    if set(inputs) != expected_inputs:
        raise RuntimeError(
            "TensorRT execution inputs differ: "
            f"missing={sorted(expected_inputs - set(inputs))}, "
            f"extra={sorted(set(inputs) - expected_inputs)}"
        )
    retained_inputs: dict[str, object] = {}
    for name, value in inputs.items():
        tensor = value.contiguous()
        if tensor.device.type != "cuda":
            raise RuntimeError(f"TensorRT input {name} is not on CUDA")
        expected_dtype = torch_dtype_for_trt(
            torch,
            trt,
            engine.get_tensor_dtype(name),
        )
        if tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"TensorRT input {name} dtype mismatch: "
                f"{tensor.dtype} != {expected_dtype}"
            )
        declared_shape = tuple(engine.get_tensor_shape(name))
        if -1 in declared_shape:
            if not context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(
                    f"TensorRT rejected input shape {name}={tuple(tensor.shape)}"
                )
        elif tuple(tensor.shape) != declared_shape:
            raise RuntimeError(
                f"TensorRT input {name} shape mismatch: "
                f"{tuple(tensor.shape)} != {declared_shape}"
            )
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(
                f"TensorRT rejected input address for {name}"
            )
        retained_inputs[name] = tensor
    missing_shapes = context.infer_shapes()
    if missing_shapes:
        raise RuntimeError(
            f"TensorRT shape inference is incomplete: {list(missing_shapes)}"
        )
    outputs: dict[str, object] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(context.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(
                f"TensorRT output {name} has unresolved shape {shape}"
            )
        dtype = torch_dtype_for_trt(
            torch,
            trt,
            engine.get_tensor_dtype(name),
        )
        location = engine.get_tensor_location(name)
        if location == trt.TensorLocation.DEVICE:
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
        elif location == trt.TensorLocation.HOST:
            tensor = torch.empty(
                shape,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            raise RuntimeError(
                f"TensorRT output {name} has unknown location {location}"
            )
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(
                f"TensorRT rejected output address for {name}"
            )
        outputs[name] = tensor
    stream = torch.cuda.current_stream()
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT execution failed")
    stream.synchronize()
    return outputs, context


def route_inputs(
    state_contracts: tuple[StateContract, ...],
    codec_tokens,
    states: tuple | None,
) -> dict[str, object]:
    result: dict[str, object] = {"codec_tokens": codec_tokens}
    if states is None:
        return result
    if len(states) != len(state_contracts):
        raise RuntimeError("NanoCodec route state tuple is incomplete")
    result.update(
        {
            contract.input_binding: tensor
            for contract, tensor in zip(state_contracts, states)
        }
    )
    return result


def compare_plan_outputs(
    torch,
    outputs: dict[str, object],
    *,
    route_label: str,
    expected_pcm,
    expected_length,
    expected_states: tuple,
    state_contracts: tuple[StateContract, ...],
) -> list[ParityMetric]:
    expected_names = {
        "pcm",
        "valid_sample_length",
        *(state.output_binding for state in state_contracts),
    }
    if set(outputs) != expected_names:
        raise RuntimeError(
            f"TensorRT {route_label} output names differ"
        )
    metrics = [
        compare_tensor(
            torch,
            outputs["pcm"].to(device="cuda"),
            expected_pcm,
            name=f"tensorrt.{route_label}.pcm",
            absolute_tolerance=TENSORRT_FP32_ATOL,
            relative_tolerance=TENSORRT_FP32_RTOL,
        ),
        compare_tensor(
            torch,
            outputs["valid_sample_length"].to(device="cuda"),
            expected_length,
            name=f"tensorrt.{route_label}.valid_sample_length",
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        ),
    ]
    for contract, expected in zip(state_contracts, expected_states):
        metrics.append(
            compare_tensor(
                torch,
                outputs[contract.output_binding].to(device="cuda"),
                expected,
                name=f"tensorrt.{route_label}.{contract.output_binding}",
                absolute_tolerance=TENSORRT_FP32_ATOL,
                relative_tolerance=TENSORRT_FP32_RTOL,
            )
        )
    return metrics


def benchmark_context(
    torch,
    context,
    *,
    warmup_iterations: int = 10,
    measured_iterations: int = 50,
) -> dict:
    if warmup_iterations < 1 or measured_iterations < 2:
        raise ValueError("NanoCodec benchmark iteration counts are invalid")
    stream = torch.cuda.current_stream()
    for _ in range(warmup_iterations):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT benchmark warmup failed")
    stream.synchronize()
    starts = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(measured_iterations)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(measured_iterations)
    ]
    for start, end in zip(starts, ends):
        start.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT benchmark execution failed")
        end.record(stream)
    stream.synchronize()
    milliseconds = [
        float(start.elapsed_time(end))
        for start, end in zip(starts, ends)
    ]
    ordered = sorted(milliseconds)
    p95_index = min(
        len(ordered) - 1,
        math.ceil(0.95 * len(ordered)) - 1,
    )
    return {
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "median_ms": statistics.median(milliseconds),
        "p95_ms": ordered[p95_index],
        "maximum_ms": max(milliseconds),
        "minimum_ms": min(milliseconds),
        "measurement_scope": (
            "non-isolated AGX Thor; existing ROS/GPU processes were not "
            "stopped or changed"
        ),
    }


def verify_tensorrt_plans(
    torch,
    trt,
    *,
    staging: Path,
    fixture: NanoCodecFixture,
    numpy,
    state_contracts: tuple[StateContract, ...],
    tail_cases: tuple[TailOracleCase, ...],
    schedule_cases: tuple[ScheduleOracleCase, ...],
) -> tuple[dict, dict, dict]:
    runtimes = {}
    engines = {}
    introspection = {}
    for route in ("initial_4", "steady_8", "tail_1_8"):
        runtime, engine, metadata = inspect_plan(
            trt,
            staging / PLAN_FILES[route],
            route=route,
            state_contracts=state_contracts,
        )
        runtimes[route] = runtime
        engines[route] = engine
        introspection[route] = metadata

    metrics_by_route: dict[str, dict] = {}
    performance: dict[str, dict] = {}
    first_codes = fixture_chunk_inputs(torch, numpy, fixture, 0)
    first_state = fixture_state(
        torch,
        numpy,
        fixture,
        state_contracts,
        0,
    )
    initial_outputs, initial_context = execute_plan(
        torch,
        trt,
        engines["initial_4"],
        route_inputs(state_contracts, first_codes, None),
    )
    initial_metrics = compare_plan_outputs(
        torch,
        initial_outputs,
        route_label="initial_4",
        expected_pcm=fixture_chunk_pcm(torch, numpy, fixture, 0),
        expected_length=fixture_chunk_length(torch, numpy, fixture, 0),
        expected_states=first_state,
        state_contracts=state_contracts,
    )
    metrics_by_route["initial_4"] = summarize_metrics(initial_metrics)
    performance["initial_4"] = benchmark_context(torch, initial_context)

    steady_codes = fixture_chunk_inputs(torch, numpy, fixture, 1)
    steady_expected_state = fixture_state(
        torch,
        numpy,
        fixture,
        state_contracts,
        1,
    )
    steady_outputs, steady_context = execute_plan(
        torch,
        trt,
        engines["steady_8"],
        route_inputs(state_contracts, steady_codes, first_state),
    )
    steady_metrics = compare_plan_outputs(
        torch,
        steady_outputs,
        route_label="steady_8",
        expected_pcm=fixture_chunk_pcm(torch, numpy, fixture, 1),
        expected_length=fixture_chunk_length(torch, numpy, fixture, 1),
        expected_states=steady_expected_state,
        state_contracts=state_contracts,
    )
    metrics_by_route["steady_8"] = summarize_metrics(steady_metrics)
    performance["steady_8"] = benchmark_context(torch, steady_context)

    tail_metrics: list[ParityMetric] = []
    tail_benchmarks: dict[str, dict] = {}
    expected_tail_counts = set(range(TAIL_MIN_FRAMES, TAIL_MAX_FRAMES + 1))
    if {case.frame_count for case in tail_cases} != expected_tail_counts:
        raise RuntimeError("NanoCodec tail oracle cases are incomplete")
    for case in tail_cases:
        outputs, context = execute_plan(
            torch,
            trt,
            engines["tail_1_8"],
            route_inputs(
                state_contracts,
                case.codec_tokens,
                first_state,
            ),
        )
        tail_metrics.extend(
            compare_plan_outputs(
                torch,
                outputs,
                route_label=f"tail_{case.frame_count}",
                expected_pcm=case.pcm,
                expected_length=case.valid_sample_length,
                expected_states=case.states,
                state_contracts=state_contracts,
            )
        )
        if case.frame_count in {
            TAIL_MIN_FRAMES,
            INITIAL_FRAMES,
            TAIL_MAX_FRAMES,
        }:
            tail_benchmarks[str(case.frame_count)] = benchmark_context(
                torch,
                context,
            )
    metrics_by_route["tail_1_8"] = summarize_metrics(tail_metrics)
    performance["tail_1_8"] = tail_benchmarks

    if len(schedule_cases) != len(fixture.frame_schedule):
        raise RuntimeError("NanoCodec closed-loop oracle schedule is incomplete")
    if any(
        case.sequence != sequence
        or case.frame_count != fixture.frame_schedule[sequence]
        for sequence, case in enumerate(schedule_cases)
    ):
        raise RuntimeError("NanoCodec closed-loop oracle schedule differs")

    closed_loop_metrics = compare_plan_outputs(
        torch,
        initial_outputs,
        route_label="closed_loop.chunk_000",
        expected_pcm=schedule_cases[0].pcm,
        expected_length=schedule_cases[0].valid_sample_length,
        expected_states=schedule_cases[0].states,
        state_contracts=state_contracts,
    )
    closed_loop_pcm = [initial_outputs["pcm"]]
    current_state = tuple(
        initial_outputs[contract.output_binding]
        for contract in state_contracts
    )
    closed_loop_contexts = [initial_context]
    for case in schedule_cases[1:]:
        terminal = case.sequence == len(schedule_cases) - 1
        route = "tail_1_8" if terminal else "steady_8"
        outputs, context = execute_plan(
            torch,
            trt,
            engines[route],
            route_inputs(
                state_contracts,
                case.codec_tokens,
                current_state,
            ),
        )
        closed_loop_contexts.append(context)
        closed_loop_metrics.extend(
            compare_plan_outputs(
                torch,
                outputs,
                route_label=f"closed_loop.chunk_{case.sequence:03d}",
                expected_pcm=case.pcm,
                expected_length=case.valid_sample_length,
                expected_states=case.states,
                state_contracts=state_contracts,
            )
        )
        closed_loop_pcm.append(outputs["pcm"])
        current_state = tuple(
            outputs[contract.output_binding]
            for contract in state_contracts
        )
    closed_loop_metrics.append(
        compare_tensor(
            torch,
            torch.cat(closed_loop_pcm, dim=-1),
            tensor_from_fixture(
                torch,
                numpy,
                fixture.records["codec.complete_pcm"],
            ),
            name="tensorrt.closed_loop.complete_pcm",
            absolute_tolerance=TENSORRT_FP32_ATOL,
            relative_tolerance=TENSORRT_FP32_RTOL,
        )
    )
    metrics_by_route["closed_loop_4_8_tail"] = summarize_metrics(
        closed_loop_metrics
    )

    # Keep runtimes alive until every engine/context operation has completed.
    if len(runtimes) != 3:
        raise AssertionError("TensorRT runtime ownership is incomplete")
    if len(closed_loop_contexts) != len(schedule_cases):
        raise AssertionError("TensorRT closed-loop context ownership is incomplete")
    return introspection, metrics_by_route, performance


def publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically rename a directory and fail if the destination exists."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "renameat2 is unavailable; atomic no-replace publish is required"
        )
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
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(output),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(output))
    raise OSError(error_number, os.strerror(error_number), str(output))


def build_contract_document(
    *,
    state_contracts: tuple[StateContract, ...],
    introspection: dict,
) -> dict:
    return {
        "schema_version": 1,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "samples_per_frame": SAMPLES_PER_FRAME,
        "codebooks": EXPECTED_CODEBOOKS,
        "schedule": {
            "initial_frames": INITIAL_FRAMES,
            "steady_frames": STEADY_FRAMES,
            "tail_min_frames": TAIL_MIN_FRAMES,
            "tail_max_frames": TAIL_MAX_FRAMES,
        },
        "stateful": True,
        "state_bindings": [
            {
                "logical_name": state.logical_name,
                "dtype": "fp32",
                "shape": list(state.shape),
                "initial_output_binding": state.output_binding,
                "steady_input_binding": state.input_binding,
                "steady_output_binding": state.output_binding,
                "tail_input_binding": state.input_binding,
                "tail_output_binding": state.output_binding,
            }
            for state in state_contracts
        ],
        "inspected_engines": {
            ENGINE_ROLES[route]: metadata
            for route, metadata in introspection.items()
        },
    }


def artifact_records(staging: Path) -> list[dict]:
    ignored = {
        RECEIPT_FILE_NAME,
        RECEIPT_DIGEST_FILE_NAME,
    }
    records = []
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        if path.name in ignored:
            continue
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"NanoCodec staging contains an invalid artifact: {path}"
            )
        records.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-model", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trtexec",
        type=Path,
        default=Path("/usr/bin/trtexec"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock, lock_sha256 = verify_locked_inputs(args)
    validate_boundary_fixture(args.fixture, args.lock)
    fixture = verify_nanocodec_fixture(
        args.fixture,
        lock,
        lock_sha256,
    )

    speech_root = args.speech_root.resolve(strict=True)
    if str(speech_root) in sys.path:
        sys.path.remove(str(speech_root))
    sys.path.insert(0, str(speech_root))
    os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy
    import onnx
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU NanoCodec export is forbidden")
    if torch.cuda.get_device_capability(0) != (11, 0):
        raise RuntimeError(
            "NanoCodec export requires the accepted Thor sm_110 device, got "
            f"{torch.cuda.get_device_capability(0)}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from nanocodec_wrapper import (
        ExplicitStateNanoCodec,
        build_codebook_table,
        enumerate_persistent_state,
        initial_input_names,
        output_names,
        stateful_input_names,
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    started = time.monotonic()
    try:
        codec, weight_receipt, streaming_codec = load_locked_codec(
            torch,
            args.codec_model,
            speech_root,
        )
        wrapper_specs = enumerate_persistent_state(codec.audio_decoder)
        state_contracts = build_state_contracts(wrapper_specs)
        require_state_fixture_contract(fixture, state_contracts)
        codebook_table = build_codebook_table(codec.vector_quantizer)
        first_codes = fixture_chunk_inputs(torch, numpy, fixture, 0)
        verify_codebook_table(
            torch,
            codec,
            codebook_table,
            first_codes,
        )
        initial_wrapper = ExplicitStateNanoCodec(
            codec.audio_decoder,
            codebook_table,
            wrapper_specs,
            initial=True,
        ).eval()
        stateful_wrapper = ExplicitStateNanoCodec(
            codec.audio_decoder,
            codebook_table,
            wrapper_specs,
            initial=False,
        ).eval()
        pytorch_parity, tail_cases, schedule_cases = verify_pytorch_routes(
            torch,
            numpy,
            fixture,
            codec,
            streaming_codec,
            initial_wrapper,
            stateful_wrapper,
            state_contracts,
        )

        state_zero_boundary = fixture_state(
            torch,
            numpy,
            fixture,
            state_contracts,
            0,
        )
        steady_codes = fixture_chunk_inputs(torch, numpy, fixture, 1)
        names_out = output_names(wrapper_specs)
        onnx_metadata = {}
        examples = {
            "initial_4": (
                initial_wrapper,
                (first_codes,),
                initial_input_names(),
                {},
            ),
            "steady_8": (
                stateful_wrapper,
                (steady_codes, *state_zero_boundary),
                stateful_input_names(wrapper_specs),
                {},
            ),
            "tail_1_8": (
                stateful_wrapper,
                (
                    steady_codes[:, :, :INITIAL_FRAMES].contiguous(),
                    *state_zero_boundary,
                ),
                stateful_input_names(wrapper_specs),
                {
                    "codec_tokens": {2: "codec_frames"},
                    "pcm": {1: "pcm_samples"},
                },
            ),
        }
        for route, (
            wrapper,
            example_inputs,
            names_in,
            dynamic_axes,
        ) in examples.items():
            onnx_path = staging / ONNX_FILES[route]
            export_onnx_graph(
                torch,
                wrapper=wrapper,
                example_inputs=tuple(example_inputs),
                input_names=names_in,
                output_names=names_out,
                dynamic_axes=dynamic_axes,
                output_path=onnx_path,
            )
            onnx_metadata[route] = verify_onnx_contract(
                onnx,
                onnx_path,
                route=route,
                state_contracts=state_contracts,
            )

        # ONNX now owns all inference weights.  Release the PyTorch modules
        # before external TensorRT builds so the exporter does not retain a
        # second copy of the decoder on the GPU.
        del initial_wrapper
        del stateful_wrapper
        del codebook_table
        del codec
        torch.cuda.empty_cache()

        build_metadata = {}
        for route in ("initial_4", "steady_8", "tail_1_8"):
            build_metadata[route] = build_plan(
                trtexec=args.trtexec,
                route=route,
                onnx_path=staging / ONNX_FILES[route],
                plan_path=staging / PLAN_FILES[route],
                log_path=staging / f"{route}.trtexec.log",
                layer_info_path=staging / f"{route}.layers.json",
            )

        trt = import_tensorrt()
        if trt.__version__ != "10.16.2.10":
            raise RuntimeError(
                f"TensorRT version mismatch: {trt.__version__} != 10.16.2.10"
            )
        introspection, tensorrt_parity, performance = verify_tensorrt_plans(
            torch,
            trt,
            staging=staging,
            fixture=fixture,
            numpy=numpy,
            state_contracts=state_contracts,
            tail_cases=tail_cases,
            schedule_cases=schedule_cases,
        )
        contract_document = build_contract_document(
            state_contracts=state_contracts,
            introspection=introspection,
        )
        (staging / CONTRACT_FILE_NAME).write_bytes(
            canonical_json_bytes(contract_document)
        )

        driver_version = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        receipt = {
            "schema_version": 1,
            "artifact_role": "stateful_nanocodec",
            "status": "accepted",
            "created_at_utc": datetime.datetime.now(
                datetime.UTC
            ).isoformat(),
            "source": {
                "exporter_sha256": sha256_file(
                    Path(__file__).resolve(strict=True)
                ),
                "wrapper_sha256": sha256_file(
                    Path(__file__).with_name(
                        "nanocodec_wrapper.py"
                    ).resolve(strict=True)
                ),
                "oracle_lock_sha256": lock_sha256,
                "oracle_source_revision": lock["oracle_source"][
                    "base_revision"
                ],
                "oracle_source_bundle_sha256": lock["oracle_source"][
                    "optimized_source_bundle_sha256"
                ],
                "model_sha256": lock["model"]["sha256"],
                "codec_model_sha256": lock["codec"]["sha256"],
                "acceptance_receipt_sha256": lock["acceptance"][
                    "receipt_sha256"
                ],
                "boundary_fixture_manifest_sha256": (
                    fixture.manifest_sha256
                ),
            },
            "contract": {
                "codebooks": EXPECTED_CODEBOOKS,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "samples_per_frame": SAMPLES_PER_FRAME,
                "initial_frames": INITIAL_FRAMES,
                "steady_frames": STEADY_FRAMES,
                "tail_frames": [TAIL_MIN_FRAMES, TAIL_MAX_FRAMES],
                "persistent_state_tensors": EXPECTED_STATE_TENSORS,
                "persistent_state_bytes": sum(
                    math.prod(state.shape) * 4
                    for state in state_contracts
                ),
                "work_buffers_are_engine_workspace": True,
                "cumulative_redecode": False,
            },
            "export": {
                "onnx_opset": ONNX_OPSET,
                "onnx_frontend": "torch.onnx legacy TorchScript exporter",
                "external_data": False,
                "weight_norm_target_count": weight_receipt.target_count,
                "weight_norm_target_names": list(
                    weight_receipt.target_names
                ),
                "onnx": onnx_metadata,
            },
            "build": {
                "tf32": False,
                "strongly_typed": True,
                "builder_optimization_level": 5,
                "profiling_verbosity": "DETAILED",
                "workspace_limit_bytes": 4 * 1024 * 1024 * 1024,
                "routes": build_metadata,
            },
            "introspection": introspection,
            "parity": {
                "pytorch": pytorch_parity,
                "tensorrt": tensorrt_parity,
                "tolerance_status": "predeclared_component_acceptance",
                "pytorch_tolerances": {
                    "fp32_atol": PYTORCH_FP32_ATOL,
                    "fp32_rtol": PYTORCH_FP32_RTOL,
                },
                "tensorrt_tolerances": {
                    "fp32_atol": TENSORRT_FP32_ATOL,
                    "fp32_rtol": TENSORRT_FP32_RTOL,
                },
                "tail_frame_counts": list(
                    range(TAIL_MIN_FRAMES, TAIL_MAX_FRAMES + 1)
                ),
            },
            "artifact_acceptance": {
                "status": "accepted",
                "scope": "stateful_nanocodec_component",
                "passed_gates": [
                    "authenticated locked model, source, and fixture",
                    "exact 97-state manifest and engine I/O contract",
                    "single-fixture PyTorch 4/8/tail closed-loop parity",
                    "single-fixture TensorRT 4/8/tail closed-loop parity",
                    "all terminal frame counts 1 through 8",
                    "non-isolated latency measurement",
                ],
                "release_evidence_not_claimed": [
                    "multiple independent locked Japanese utterance fixtures",
                    "long-duration closed-loop state-drift measurement",
                    "isolated AGX Thor performance measurement",
                ],
            },
            "numerical_equivalence": {
                "oracle_state_update": (
                    "mutable preallocated work-buffer copy followed by CUDA "
                    "convolution"
                ),
                "export_state_update": (
                    "functional Concat followed by CUDA convolution"
                ),
                "observed_weight_norm_probe": (
                    "materialized and non-materialized wrapper probes had the "
                    "same 2.7865171432495117e-06 initial-boundary maximum; "
                    "weight-norm materialization was not the observed primary "
                    "source"
                ),
                "remaining_difference_hypothesis": (
                    "allocation-dependent CUDA convolution tactic or reduction "
                    "order; this is an inference, not a confirmed root cause"
                ),
            },
            "performance": performance,
            "runtime": {
                "os": platform.platform(),
                "architecture": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "onnx": onnx.__version__,
                "tensorrt": trt.__version__,
                "driver": driver_version,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        receipt["artifacts"] = artifact_records(staging)
        receipt_payload = canonical_json_bytes(receipt)
        (staging / RECEIPT_FILE_NAME).write_bytes(receipt_payload)
        (staging / RECEIPT_DIGEST_FILE_NAME).write_text(
            f"{sha256_bytes(receipt_payload)}  {RECEIPT_FILE_NAME}\n",
            encoding="ascii",
        )
        publish_directory_no_replace(staging, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "receipt_sha256": sha256_bytes(receipt_payload),
                    "pytorch_parity": pytorch_parity,
                    "tensorrt_parity": tensorrt_parity,
                    "performance": performance,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"NanoCodec export failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

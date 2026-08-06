#!/usr/bin/env python3
"""Execute the fixed-shape Local AR TensorRT plan.

The runner is intentionally independent from NeMo and the export wrapper. It
loads the plugin through the public versioned C ABI, validates the complete
TensorRT I/O contract, executes one fixed 16-position Local AR step, and
publishes raw outputs plus an authenticated receipt without replacing an
existing directory.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import errno
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tensorrt
import torch


CFG_BATCH = 2
ACTUAL_BATCH = 1
MODEL_WIDTH = 768
CODEBOOKS = 8
FRAMES_PER_STEP = 2
PHILOX_STRIDE = 2048
INT64_COUNTER_MAX = ((1 << 63) - 1 - (PHILOX_STRIDE - 1)) // PHILOX_STRIDE

PLUGIN_ABI_VERSION = 1
PLUGIN_CREATOR_COUNT = 5
EXPECTED_PLUGIN_CREATORS = (
    ("MagpieLocalARSampling", "1", "magpie_tts_rt"),
    ("MagpieLocalAREos", "1", "magpie_tts_rt"),
    ("MagpieLayerNorm", "1", "magpie_tts_rt"),
    ("MagpieGeluTanh", "1", "magpie_tts_rt"),
    ("MagpieSoftmax", "1", "magpie_tts_rt"),
)
PLUGIN_STATUS_OK = 0
PLUGIN_STATUS_ALREADY_REGISTERED = 1

INPUT_NAMES = [
    "decoder_hidden",
    "unfinished",
    "finished",
    "forbid_eos",
    "rng_seed",
    "rng_counter",
]
OUTPUT_NAMES = [
    "codec_tokens",
    "updated_rng_counter",
    "invalid_rows",
    "end_frame_index",
]

RECEIPT_NAME = "execution-receipt.json"
RECEIPT_DIGEST_NAME = "execution-receipt.json.sha256"


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


@dataclass(frozen=True)
class RegisteredPlugin:
    library: ctypes.CDLL
    api: PluginApi
    registration_status: int


@dataclass(frozen=True)
class LocalARTensors:
    """Accepted outputs from one reusable-session execution."""

    codec_tokens: torch.Tensor
    updated_rng_counter: torch.Tensor
    invalid_rows: torch.Tensor
    end_frame_index: torch.Tensor


@dataclass(frozen=True)
class LocalARResult:
    """One accepted Local AR execution and its durable evidence."""

    tensors: LocalARTensors
    output: Path
    receipt_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def expected_plan_contract() -> dict[str, tuple[str, tuple[int, ...], str]]:
    return {
        "decoder_hidden": ("bf16", (CFG_BATCH, MODEL_WIDTH), "input"),
        "unfinished": ("bool", (ACTUAL_BATCH,), "input"),
        "finished": ("bool", (ACTUAL_BATCH,), "input"),
        "forbid_eos": ("bool", (ACTUAL_BATCH,), "input"),
        "rng_seed": ("int64", (1,), "input"),
        "rng_counter": ("int64", (ACTUAL_BATCH,), "input"),
        "codec_tokens": (
            "int64",
            (ACTUAL_BATCH, CODEBOOKS, FRAMES_PER_STEP),
            "output",
        ),
        "updated_rng_counter": ("int64", (ACTUAL_BATCH,), "output"),
        "invalid_rows": ("int32", (ACTUAL_BATCH,), "output"),
        "end_frame_index": ("int32", (ACTUAL_BATCH,), "output"),
    }


def register_plugin(path: Path) -> RegisteredPlugin:
    library = ctypes.CDLL(str(path.resolve(strict=True)), mode=ctypes.RTLD_LOCAL)
    get_api = library.mtt_plugin_get_api_v1
    get_api.argtypes = [ctypes.POINTER(PluginApi)]
    get_api.restype = ctypes.c_int32
    api = PluginApi()
    api.struct_size = ctypes.sizeof(PluginApi)
    api.abi_version = PLUGIN_ABI_VERSION
    status = int(get_api(ctypes.byref(api)))
    if status != PLUGIN_STATUS_OK:
        raise RuntimeError(f"plugin API load failed with status {status}")
    if api.creator_count != PLUGIN_CREATOR_COUNT:
        raise RuntimeError(
            "plugin creator count mismatch: expected "
            f"{PLUGIN_CREATOR_COUNT}, got {api.creator_count}"
        )
    creators = tuple(
        (
            creator.name.decode("ascii"),
            creator.version.decode("ascii"),
            creator.plugin_namespace.decode("ascii"),
        )
        for creator in api.creators[: api.creator_count]
    )
    if creators != EXPECTED_PLUGIN_CREATORS:
        raise RuntimeError(
            "plugin creator contract mismatch: "
            f"expected={EXPECTED_PLUGIN_CREATORS}, got={creators}"
        )
    registration_status = int(api.register_plugins())
    if registration_status not in (
        PLUGIN_STATUS_OK,
        PLUGIN_STATUS_ALREADY_REGISTERED,
    ):
        raise RuntimeError(
            "plugin registration failed with status "
            f"{registration_status}"
        )
    return RegisteredPlugin(
        library=library,
        api=api,
        registration_status=registration_status,
    )


def plugin_metadata(plugin: RegisteredPlugin) -> dict:
    return {
        "abi_version": plugin.api.abi_version,
        "registration_status": plugin.registration_status,
        "creators": [
            {
                "name": creator.name.decode("ascii"),
                "version": creator.version.decode("ascii"),
                "namespace": creator.plugin_namespace.decode("ascii"),
            }
            for creator in plugin.api.creators[: plugin.api.creator_count]
        ],
    }


def require_plan_contract(engine: tensorrt.ICudaEngine) -> list[dict]:
    expected = expected_plan_contract()
    actual_names = {
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    }
    if actual_names != set(expected):
        raise RuntimeError(
            "Local AR plan I/O mismatch: "
            f"missing={sorted(set(expected) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected))}"
        )
    dtype_names = {
        tensorrt.DataType.BF16: "bf16",
        tensorrt.DataType.BOOL: "bool",
        tensorrt.DataType.INT32: "int32",
        tensorrt.DataType.INT64: "int64",
    }
    records: list[dict] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        actual = (
            dtype_names.get(engine.get_tensor_dtype(name)),
            tuple(engine.get_tensor_shape(name)),
            "input"
            if engine.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT
            else "output",
        )
        if actual != expected[name]:
            raise RuntimeError(
                f"Local AR plan tensor {name} mismatch: "
                f"expected {expected[name]}, got {actual}"
            )
        records.append(
            {
                "name": name,
                "dtype": actual[0],
                "shape": list(actual[1]),
                "mode": actual[2],
            }
        )
    return records


def load_decoder_hidden(path: Path) -> torch.Tensor:
    resolved = path.resolve(strict=True)
    expected_size = CFG_BATCH * MODEL_WIDTH * 2
    payload = resolved.read_bytes()
    if len(payload) != expected_size:
        raise RuntimeError(
            "decoder hidden byte size mismatch: expected "
            f"{expected_size}, got {len(payload)}"
        )
    value = torch.frombuffer(
        bytearray(payload),
        dtype=torch.uint16,
    ).view(torch.bfloat16)
    return value.reshape(CFG_BATCH, MODEL_WIDTH).clone().to("cuda")


def tensor_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def publish_directory_no_replace(staging: Path, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for atomic publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(output))
    raise OSError(error_number, os.strerror(error_number), str(output))


def _validate_scalar_inputs(
    seed: int,
    counter: int,
    unfinished: bool,
    finished: bool,
    forbid_eos: bool,
) -> None:
    if seed < 0 or seed >= (1 << 32):
        raise ValueError("seed must be in the unsigned 32-bit range")
    if counter < 0 or counter > INT64_COUNTER_MAX:
        raise ValueError(
            f"counter must be in the inclusive range [0, {INT64_COUNTER_MAX}]"
        )
    if unfinished and finished:
        raise ValueError("unfinished and finished cannot both be true")
    if finished and forbid_eos:
        raise ValueError("finished and forbid_eos cannot both be true")


class LocalARSession:
    """Own one plugin library, TensorRT engine, context, and fixed buffers."""

    def __init__(self, *, plan_path: Path, plugin_path: Path) -> None:
        if sys.byteorder != "little":
            raise RuntimeError("Local AR raw tensor contract requires little endian")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU execution is forbidden")
        if torch.cuda.get_device_capability(0) != (11, 0):
            raise RuntimeError(
                "Local AR plan requires Thor sm_110, got "
                f"{torch.cuda.get_device_capability(0)}"
            )

        self.plan_path = plan_path.resolve(strict=True)
        self.plugin_path = plugin_path.resolve(strict=True)
        self.plugin = register_plugin(self.plugin_path)
        self.logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        self.runtime = tensorrt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            self.plan_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError("failed to deserialize Local AR plan")
        self.tensor_contract = require_plan_contract(self.engine)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create Local AR execution context")

        self._inputs = {
            "unfinished": torch.empty((1,), dtype=torch.bool, device="cuda"),
            "finished": torch.empty((1,), dtype=torch.bool, device="cuda"),
            "forbid_eos": torch.empty((1,), dtype=torch.bool, device="cuda"),
            "rng_seed": torch.empty((1,), dtype=torch.int64, device="cuda"),
            "rng_counter": torch.empty((1,), dtype=torch.int64, device="cuda"),
        }
        dtype_map = {
            tensorrt.DataType.BF16: torch.bfloat16,
            tensorrt.DataType.BOOL: torch.bool,
            tensorrt.DataType.INT32: torch.int32,
            tensorrt.DataType.INT64: torch.int64,
        }
        self._outputs: dict[str, torch.Tensor] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT:
                if name == "decoder_hidden":
                    continue
                value = self._inputs[name]
            else:
                value = torch.empty(
                    tuple(self.engine.get_tensor_shape(name)),
                    dtype=dtype_map[self.engine.get_tensor_dtype(name)],
                    device="cuda",
                )
                self._outputs[name] = value
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"failed to bind Local AR tensor {name}")

    def execute(
        self,
        decoder_hidden: torch.Tensor,
        seed: int,
        counter: int,
        unfinished: bool,
        finished: bool,
        forbid_eos: bool,
    ) -> LocalARTensors:
        """Execute one step without reloading the plan or allocating I/O buffers."""

        _validate_scalar_inputs(seed, counter, unfinished, finished, forbid_eos)
        if (
            decoder_hidden.dtype != torch.bfloat16
            or tuple(decoder_hidden.shape) != (CFG_BATCH, MODEL_WIDTH)
            or decoder_hidden.device.type != "cuda"
            or decoder_hidden.device.index != torch.cuda.current_device()
            or not decoder_hidden.is_contiguous()
        ):
            raise ValueError(
                "decoder_hidden must be a contiguous CUDA BF16 tensor with "
                f"shape ({CFG_BATCH}, {MODEL_WIDTH}) on the current device"
            )

        self._inputs["unfinished"].fill_(unfinished)
        self._inputs["finished"].fill_(finished)
        self._inputs["forbid_eos"].fill_(forbid_eos)
        self._inputs["rng_seed"].fill_(seed)
        self._inputs["rng_counter"].fill_(counter)
        if not self.context.set_tensor_address(
            "decoder_hidden", decoder_hidden.data_ptr()
        ):
            raise RuntimeError("failed to bind Local AR decoder_hidden")
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("Local AR plan execution failed")

        invalid_rows = self._outputs["invalid_rows"]
        if torch.any(invalid_rows != 0).item():
            raise RuntimeError(
                "Local AR plugin rejected the input row: "
                f"invalid_rows={invalid_rows.detach().cpu().tolist()}"
            )
        return LocalARTensors(
            codec_tokens=self._outputs["codec_tokens"].clone(),
            updated_rng_counter=self._outputs["updated_rng_counter"].clone(),
            invalid_rows=invalid_rows.clone(),
            end_frame_index=self._outputs["end_frame_index"].clone(),
        )


def execute_local_ar_plan(
    *,
    plan_path: Path,
    plugin_path: Path,
    decoder_hidden_path: Path,
    seed: int,
    counter: int,
    unfinished: bool,
    finished: bool,
    forbid_eos: bool,
    output: Path,
) -> LocalARResult:
    """Execute and atomically publish one fixed-shape Local AR step."""

    resolved_hidden = decoder_hidden_path.resolve(strict=True)
    resolved_output = output.resolve()
    if resolved_output.exists():
        raise FileExistsError(f"output already exists: {resolved_output}")

    session = LocalARSession(plan_path=plan_path, plugin_path=plugin_path)
    hidden = load_decoder_hidden(resolved_hidden)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    tensors = session.execute(
        hidden,
        seed,
        counter,
        unfinished,
        finished,
        forbid_eos,
    )
    end.record()
    end.synchronize()
    duration_ms = float(start.elapsed_time(end))

    result_tensors = {
        name: getattr(tensors, name).detach().cpu().clone() for name in OUTPUT_NAMES
    }

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_output.name}.",
            dir=resolved_output.parent,
        )
    )
    try:
        output_specs = [
            ("codec_tokens", "codec_tokens.int64.bin", "int64", [1, 8, 2]),
            (
                "updated_rng_counter",
                "updated_rng_counter.int64.bin",
                "int64",
                [1],
            ),
            ("invalid_rows", "invalid_rows.int32.bin", "int32", [1]),
            (
                "end_frame_index",
                "end_frame_index.int32.bin",
                "int32",
                [1],
            ),
        ]
        artifact_records = []
        for tensor_name, filename, dtype, shape in output_specs:
            artifact_path = staging / filename
            artifact_path.write_bytes(tensor_bytes(result_tensors[tensor_name]))
            artifact_records.append(
                {
                    "name": tensor_name,
                    "path": filename,
                    "dtype": dtype,
                    "shape": shape,
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": sha256_file(artifact_path),
                }
            )

        receipt = {
            "schema_version": 1,
            "artifact_role": "local_ar_execution",
            "status": "accepted",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "tensorrt": tensorrt.__version__,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
            },
            "inputs": {
                "plan": {
                    "path": str(session.plan_path),
                    "sha256": sha256_file(session.plan_path),
                },
                "plugin": {
                    "path": str(session.plugin_path),
                    "sha256": sha256_file(session.plugin_path),
                    "api": plugin_metadata(session.plugin),
                },
                "decoder_hidden": {
                    "path": str(resolved_hidden),
                    "dtype": "bf16",
                    "shape": [CFG_BATCH, MODEL_WIDTH],
                    "size_bytes": resolved_hidden.stat().st_size,
                    "sha256": sha256_file(resolved_hidden),
                },
                "seed": seed,
                "counter": counter,
                "unfinished": unfinished,
                "finished": finished,
                "forbid_eos": forbid_eos,
            },
            "plan_contract": session.tensor_contract,
            "execution": {
                "duration_ms": duration_ms,
                "cuda_stream": "torch_current_stream",
            },
            "artifacts": artifact_records,
        }
        receipt_payload = canonical_json_bytes(receipt)
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        (staging / RECEIPT_NAME).write_bytes(receipt_payload)
        (staging / RECEIPT_DIGEST_NAME).write_text(
            f"{receipt_sha256}  {RECEIPT_NAME}\n",
            encoding="ascii",
        )
        publish_directory_no_replace(staging, resolved_output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return LocalARResult(
        tensors=LocalARTensors(
            codec_tokens=result_tensors["codec_tokens"],
            updated_rng_counter=result_tensors["updated_rng_counter"],
            invalid_rows=result_tensors["invalid_rows"],
            end_frame_index=result_tensors["end_frame_index"],
        ),
        output=resolved_output,
        receipt_sha256=receipt_sha256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plugins", type=Path, required=True)
    parser.add_argument("--decoder-hidden", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--counter", type=int, required=True)
    parser.add_argument("--unfinished", type=int, choices=(0, 1), required=True)
    parser.add_argument("--finished", type=int, choices=(0, 1), required=True)
    parser.add_argument("--forbid-eos", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = execute_local_ar_plan(
        plan_path=args.plan,
        plugin_path=args.plugins,
        decoder_hidden_path=args.decoder_hidden,
        seed=args.seed,
        counter=args.counter,
        unfinished=bool(args.unfinished),
        finished=bool(args.finished),
        forbid_eos=bool(args.forbid_eos),
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(result.output),
                "receipt_sha256": result.receipt_sha256,
                "codec_tokens": result.tensors.codec_tokens.tolist(),
                "updated_rng_counter": (
                    result.tensors.updated_rng_counter.tolist()
                ),
                "invalid_rows": result.tensors.invalid_rows.tolist(),
                "end_frame_index": result.tensors.end_frame_index.tolist(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Local AR execution failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

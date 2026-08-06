#!/usr/bin/env python3
"""Export, build, and validate the fixed 16-position Local AR engine.

Every external input is explicit and authenticated. The output directory is
published atomically without replacing an existing artifact. PyTorch parity,
TensorRT plan parity, plugin identity, and the learned position table are all
recorded in the receipt.
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
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import onnx
import tensorrt
import torch
from nemo.collections.tts.models import MagpieTTSModel

from local_ar_wrapper import (
    ACTUAL_BATCH,
    CFG_BATCH,
    CODEBOOKS,
    FRAMES_PER_STEP,
    INPUT_NAMES,
    LOCAL_LAYERS,
    LOCAL_POSITIONS,
    MODEL_WIDTH,
    OUTPUT_NAMES,
    LocalARWrapper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_TOOLS = PROJECT_ROOT / "tools" / "oracle"
if str(ORACLE_TOOLS) not in sys.path:
    sys.path.insert(0, str(ORACLE_TOOLS))

from validate_boundary_fixture import validate_boundary_fixture  # noqa: E402
from verify_oracle_lock import (  # noqa: E402
    FileExpectation,
    require_file,
    require_model_configs,
    require_source_checkout,
    sha256_file,
)


ONNX_OPSET = 20
ONNX_NAME = "local-ar.onnx"
PLAN_NAME = "local-ar.plan"
PLUGIN_NAME = "libmagpie_tts_rt_plugins.so"
RECEIPT_NAME = "export-receipt.json"
RECEIPT_DIGEST_NAME = "export-receipt.json.sha256"
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
POSITION_TABLE_SHAPE = (18, MODEL_WIDTH)
EXPECTED_FIXTURE_MANIFEST_SHA256 = frozenset(
    {
        "0ca05d9b613aa4b3923ded357a812260da91cb2f1d60f04de6e57ba3f6b8004c",
        "c0ae5528df93eb93335a49f3487dc69725a75e6372712db062a4d33a181c9996",
        "2259a6bb48e0098ea3cbfde126417c8727cdecbf5ce6d357b302510413ef8119",
    }
)


@dataclass(frozen=True)
class FixtureTensor:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class LocalARFixture:
    manifest_sha256: str
    hidden: FixtureTensor
    seed: FixtureTensor
    initial_counter: FixtureTensor
    codes: FixtureTensor
    updated_counter: FixtureTensor


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-model", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        required=True,
        help=(
            "authenticated boundary fixture; repeat exactly three times for "
            "the locked canonical, filler, and status fixtures"
        ),
    )
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    return parser.parse_args()


def canonical_json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_locked_inputs(args: argparse.Namespace) -> tuple[dict, str]:
    lock_path = args.lock.resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require_file(
        args.model,
        FileExpectation(lock["model"]["sha256"], lock["model"]["size_bytes"]),
        "Magpie model",
    )
    require_model_configs(
        args.model,
        lock["model"]["config_member_sha256"],
        lock["model"]["active_config_policy"],
        lock["model"]["active_config_sha256"],
    )
    require_file(
        args.codec_model,
        FileExpectation(lock["codec"]["sha256"], lock["codec"]["size_bytes"]),
        "NanoCodec model",
    )
    require_file(
        args.acceptance_receipt,
        FileExpectation(
            lock["acceptance"]["receipt_sha256"],
            lock["acceptance"]["receipt_size_bytes"],
        ),
        "acceptance receipt",
    )
    require_source_checkout(
        args.speech_root,
        lock["oracle_source"]["base_revision"],
        lock["oracle_source"]["files"],
    )
    return lock, sha256_file(lock_path)


def load_fixture(fixture_path: Path, lock_path: Path) -> LocalARFixture:
    root = fixture_path.resolve(strict=True)
    validate_boundary_fixture(root, lock_path.resolve(strict=True))
    manifest_path = root / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 not in EXPECTED_FIXTURE_MANIFEST_SHA256:
        raise RuntimeError(
            "Local AR export requires one of the three locked boundary "
            f"fixtures {sorted(EXPECTED_FIXTURE_MANIFEST_SHA256)}, "
            f"got {manifest_sha256}"
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {record["name"]: record for record in document["tensors"]}

    def require_tensor(
        name: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> FixtureTensor:
        record = records.get(name)
        if record is None:
            raise RuntimeError(f"fixture tensor is missing: {name}")
        if record["dtype"] != dtype or tuple(record["shape"]) != shape:
            raise RuntimeError(
                f"fixture tensor {name} mismatch: expected {dtype} {shape}, "
                f"got {record['dtype']} {tuple(record['shape'])}"
            )
        path = (root / record["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise RuntimeError(f"fixture tensor escapes fixture root: {name}")
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"fixture tensor size mismatch: {name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"fixture tensor SHA-256 mismatch: {name}")
        return FixtureTensor(
            path=path,
            dtype=dtype,
            shape=shape,
            sha256=record["sha256"],
            size_bytes=record["size_bytes"],
        )

    return LocalARFixture(
        manifest_sha256=manifest_sha256,
        hidden=require_tensor("prefill.hidden", "bf16", (2, 1, MODEL_WIDTH)),
        seed=require_tensor("local_ar.initial_seed", "int64", (1,)),
        initial_counter=require_tensor(
            "local_ar.initial_counters", "int64", (1,)
        ),
        codes=require_tensor(
            "local_ar.step_000.codes",
            "int64",
            (ACTUAL_BATCH, CODEBOOKS, FRAMES_PER_STEP),
        ),
        updated_counter=require_tensor(
            "local_ar.step_000.counters", "int64", (1,)
        ),
    )


def tensor_from_fixture(tensor: FixtureTensor) -> torch.Tensor:
    storage_dtype = {
        "bf16": torch.uint16,
        "int64": torch.int64,
    }[tensor.dtype]
    value = torch.frombuffer(
        bytearray(tensor.path.read_bytes()), dtype=storage_dtype
    ).reshape(tensor.shape).clone()
    return value.view(torch.bfloat16) if tensor.dtype == "bf16" else value


def load_wrapper(model_path: Path, speech_root: Path) -> LocalARWrapper:
    module = sys.modules.get(MagpieTTSModel.__module__)
    module_path = getattr(module, "__file__", None)
    expected_path = (
        speech_root.resolve(strict=True)
        / "nemo"
        / "collections"
        / "tts"
        / "models"
        / "magpietts.py"
    ).resolve(strict=True)
    if not isinstance(module_path, str) or Path(module_path).resolve() != expected_path:
        raise RuntimeError(
            f"MagpieTTS imported from the wrong source: {module_path!r}"
        )
    model = MagpieTTSModel.restore_from(
        str(model_path.resolve(strict=True)),
        map_location="cpu",
    )
    model.eval()
    wrapper = LocalARWrapper(model).eval()
    wrapper.to(device="cuda", dtype=torch.bfloat16)
    return wrapper


def fixture_inputs(
    fixture: LocalARFixture,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    hidden = tensor_from_fixture(fixture.hidden).squeeze(1).to("cuda")
    seed = tensor_from_fixture(fixture.seed).to("cuda")
    initial_counter = tensor_from_fixture(fixture.initial_counter).to("cuda")
    values = (
        hidden,
        torch.ones((ACTUAL_BATCH,), dtype=torch.bool, device="cuda"),
        torch.zeros((ACTUAL_BATCH,), dtype=torch.bool, device="cuda"),
        torch.ones((ACTUAL_BATCH,), dtype=torch.bool, device="cuda"),
        seed,
        initial_counter,
    )
    return (
        values,
        tensor_from_fixture(fixture.codes).to("cuda"),
        tensor_from_fixture(fixture.updated_counter).to("cuda"),
    )


def position_table_digest(wrapper: LocalARWrapper) -> tuple[str, tuple[int, ...]]:
    embeddings = wrapper.local_transformer.position_embeddings
    if embeddings is None:
        raise RuntimeError("accepted Local AR position table is absent")
    table = embeddings.weight.detach().contiguous()
    shape = tuple(table.shape)
    if shape != POSITION_TABLE_SHAPE or table.dtype != torch.bfloat16:
        raise RuntimeError(
            "Local AR position table mismatch: expected BF16 "
            f"{POSITION_TABLE_SHAPE}, got {table.dtype} {shape}"
        )
    payload = table.cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest(), shape


def require_pytorch_parity(
    wrapper: LocalARWrapper,
    inputs: tuple[torch.Tensor, ...],
    expected_codes: torch.Tensor,
    expected_counter: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    with torch.inference_mode():
        outputs = wrapper(*inputs)
    expected = (
        expected_codes,
        expected_counter,
        torch.zeros((ACTUAL_BATCH,), dtype=torch.int32, device="cuda"),
        torch.full((ACTUAL_BATCH,), -1, dtype=torch.int32, device="cuda"),
    )
    for name, actual, wanted in zip(OUTPUT_NAMES, outputs, expected, strict=True):
        if actual.dtype != wanted.dtype or tuple(actual.shape) != tuple(wanted.shape):
            raise RuntimeError(
                f"PyTorch {name} contract mismatch: expected "
                f"{wanted.dtype} {tuple(wanted.shape)}, got "
                f"{actual.dtype} {tuple(actual.shape)}"
            )
        if not torch.equal(actual, wanted):
            raise RuntimeError(
                f"PyTorch {name} is not bit-exact: "
                f"actual={actual.cpu().tolist()}, expected={wanted.cpu().tolist()}"
            )
    return outputs


def require_position_embedding_negative(
    wrapper: LocalARWrapper,
    inputs: tuple[torch.Tensor, ...],
    accepted_codes: torch.Tensor,
) -> list[list[list[int]]]:
    embeddings = wrapper.local_transformer.position_embeddings
    if embeddings is None:
        raise RuntimeError("accepted Local AR position table is absent")
    saved = embeddings.weight.detach().clone()
    try:
        with torch.inference_mode():
            embeddings.weight[:LOCAL_POSITIONS].zero_()
            codes = wrapper(*inputs)[0]
    finally:
        with torch.inference_mode():
            embeddings.weight.copy_(saved)
    if torch.equal(codes, accepted_codes):
        raise RuntimeError(
            "omitting Local AR position embeddings did not change fixture codes"
        )
    return codes.cpu().tolist()


def export_onnx(
    wrapper: LocalARWrapper,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
) -> dict:
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            str(path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=ONNX_OPSET,
            do_constant_folding=False,
            dynamo=False,
            external_data=False,
        )
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    if input_names != INPUT_NAMES or output_names != OUTPUT_NAMES:
        raise RuntimeError(
            "Local AR ONNX I/O mismatch: "
            f"inputs={input_names}, outputs={output_names}"
        )
    custom_nodes: dict[str, int] = {}
    for node in model.graph.node:
        if node.domain == "magpie_tts_rt":
            custom_nodes[node.op_type] = custom_nodes.get(node.op_type, 0) + 1
    expected_custom = {
        "MagpieLocalARSampling": LOCAL_POSITIONS,
        "MagpieLocalAREos": 1,
        "MagpieLayerNorm": LOCAL_POSITIONS * LOCAL_LAYERS * 2,
        "MagpieGeluTanh": LOCAL_POSITIONS * LOCAL_LAYERS,
        "MagpieSoftmax": LOCAL_POSITIONS * LOCAL_LAYERS,
    }
    if custom_nodes != expected_custom:
        raise RuntimeError(
            f"Local AR custom-node count mismatch: {custom_nodes}"
        )
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "custom_nodes": custom_nodes,
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
        raise RuntimeError(f"unexpected plugin creator count {api.creator_count}")
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
    creators = []
    for index in range(plugin.api.creator_count):
        creator = plugin.api.creators[index]
        creators.append(
            {
                "name": creator.name.decode("ascii"),
                "version": creator.version.decode("ascii"),
                "namespace": creator.plugin_namespace.decode("ascii"),
            }
        )
    return {
        "abi_version": plugin.api.abi_version,
        "registration_status": plugin.registration_status,
        "creators": creators,
    }


def build_plan(onnx_path: Path, plan_path: Path) -> dict:
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(logger)
    flags = 1 << int(tensorrt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = tensorrt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.clear_flag(tensorrt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.profiling_verbosity = (
        tensorrt.ProfilingVerbosity.DETAILED
    )
    config.set_memory_pool_limit(
        tensorrt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the Local AR plan")
    plan_path.write_bytes(bytes(serialized))
    return {
        "network_flags": ["strongly_typed"],
        "builder_optimization_level": config.builder_optimization_level,
        "profiling_verbosity": "DETAILED",
        "tf32": False,
        "workspace_bytes": 8 * 1024 * 1024 * 1024,
    }


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


def inspect_plan(plan_path: Path) -> dict:
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError("failed to deserialize Local AR plan")
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
    records = []
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
    return {
        "tensorrt_version": tensorrt.__version__,
        "optimization_profiles": engine.num_optimization_profiles,
        "tensors": records,
    }


def execute_and_measure(
    plan_path: Path,
    cases: list[
        tuple[
            str,
            tuple[torch.Tensor, ...],
            tuple[torch.Tensor, ...],
        ]
    ],
    iterations: int,
) -> tuple[dict, dict]:
    if iterations < 1:
        raise ValueError("benchmark iterations must be positive")
    if not cases:
        raise ValueError("at least one Local AR fixture case is required")
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError("failed to deserialize Local AR plan")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create Local AR execution context")
    dtype_map = {
        tensorrt.DataType.BF16: torch.bfloat16,
        tensorrt.DataType.BOOL: torch.bool,
        tensorrt.DataType.INT32: torch.int32,
        tensorrt.DataType.INT64: torch.int64,
    }
    outputs: dict[str, torch.Tensor] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == tensorrt.TensorIOMode.OUTPUT:
            value = torch.empty(
                tuple(engine.get_tensor_shape(name)),
                dtype=dtype_map[engine.get_tensor_dtype(name)],
                device="cuda",
            )
            outputs[name] = value
        elif name not in INPUT_NAMES:
            raise RuntimeError(f"unexpected Local AR plan input {name}")
        else:
            continue
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"failed to bind Local AR tensor {name}")

    stream = torch.cuda.current_stream().cuda_stream
    parity: dict[str, dict] = {}
    for manifest_sha256, inputs, expected_outputs in cases:
        values = dict(zip(INPUT_NAMES, inputs, strict=True))
        for name, value in values.items():
            if not context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(
                    f"failed to bind Local AR fixture input {name}"
                )
        if not context.execute_async_v3(stream):
            raise RuntimeError(
                "Local AR multi-fixture parity execution failed"
            )
        torch.cuda.synchronize()
        fixture_parity = {}
        for name, expected in zip(
            OUTPUT_NAMES, expected_outputs, strict=True
        ):
            actual = outputs[name]
            fixture_parity[name] = {
                "equal": torch.equal(actual, expected),
                "actual": actual.cpu().tolist(),
                "expected": expected.cpu().tolist(),
            }
            if not fixture_parity[name]["equal"]:
                raise RuntimeError(
                    f"TensorRT fixture {manifest_sha256} {name} is not "
                    "bit-exact: "
                    f"actual={fixture_parity[name]['actual']}, "
                    f"expected={fixture_parity[name]['expected']}"
                )
        parity[manifest_sha256] = fixture_parity

    benchmark_inputs = dict(
        zip(INPUT_NAMES, cases[0][1], strict=True)
    )
    for name, value in benchmark_inputs.items():
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(
                f"failed to bind Local AR benchmark input {name}"
            )
    for _ in range(3):
        if not context.execute_async_v3(stream):
            raise RuntimeError("Local AR warmup execution failed")
    torch.cuda.synchronize()

    durations_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if not context.execute_async_v3(stream):
            raise RuntimeError("Local AR benchmark execution failed")
        end.record()
        end.synchronize()
        durations_ms.append(float(start.elapsed_time(end)))

    ordered = sorted(durations_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    benchmark = {
        "status": "measured-not-accepted",
        "reason": "GPU was shared with the running robot stack",
        "iterations": iterations,
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }
    return parity, benchmark


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


def artifact_record(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU export is forbidden")
    if torch.cuda.get_device_capability(0) != (11, 0):
        raise RuntimeError(
            "Local AR export requires Thor sm_110, got "
            f"{torch.cuda.get_device_capability(0)}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    lock, lock_sha256 = verify_locked_inputs(args)
    if len(args.fixture) != len(EXPECTED_FIXTURE_MANIFEST_SHA256):
        raise RuntimeError(
            "Local AR export requires exactly three --fixture arguments"
        )
    fixtures = [
        load_fixture(path, args.lock) for path in args.fixture
    ]
    fixture_manifests = {
        fixture.manifest_sha256 for fixture in fixtures
    }
    if fixture_manifests != EXPECTED_FIXTURE_MANIFEST_SHA256:
        raise RuntimeError(
            "Local AR fixture set mismatch: "
            f"expected={sorted(EXPECTED_FIXTURE_MANIFEST_SHA256)}, "
            f"got={sorted(fixture_manifests)}"
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        wrapper = load_wrapper(args.model, args.speech_root)
        pytorch_cases: list[
            tuple[
                str,
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
            ]
        ] = []
        for fixture in fixtures:
            inputs, expected_codes, expected_counter = fixture_inputs(
                fixture
            )
            expected_outputs = require_pytorch_parity(
                wrapper, inputs, expected_codes, expected_counter
            )
            pytorch_cases.append(
                (
                    fixture.manifest_sha256,
                    inputs,
                    expected_outputs,
                )
            )
        canonical_manifest = (
            "0ca05d9b613aa4b3923ded357a812260da91cb2f1d60f04de6e57ba3f6b8004c"
        )
        canonical_case = next(
            case for case in pytorch_cases if case[0] == canonical_manifest
        )
        canonical_inputs = canonical_case[1]
        canonical_codes = canonical_case[2][0]
        position_sha256, position_shape = position_table_digest(wrapper)
        no_position_codes = require_position_embedding_negative(
            wrapper, canonical_inputs, canonical_codes
        )

        onnx_path = staging / ONNX_NAME
        onnx_metadata = export_onnx(
            wrapper, canonical_inputs, onnx_path
        )

        plugin_path = staging / PLUGIN_NAME
        shutil.copyfile(args.plugin.resolve(strict=True), plugin_path)
        plugin = register_plugin(plugin_path)

        plan_path = staging / PLAN_NAME
        build_metadata = build_plan(onnx_path, plan_path)
        plan_metadata = inspect_plan(plan_path)
        plan_parity, benchmark = execute_and_measure(
            plan_path,
            pytorch_cases,
            args.benchmark_iterations,
        )

        receipt = {
            "schema_version": 1,
            "artifact_role": "local_ar_fixed_16",
            "status": "measured-not-accepted",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "source": {
                "exporter_sha256": sha256_file(Path(__file__).resolve(strict=True)),
                "wrapper_sha256": sha256_file(
                    (Path(__file__).parent / "local_ar_wrapper.py").resolve(
                        strict=True
                    )
                ),
                "oracle_lock_sha256": lock_sha256,
                "oracle_source_revision": lock["oracle_source"]["base_revision"],
                "model_sha256": lock["model"]["sha256"],
                "codec_model_sha256": lock["codec"]["sha256"],
                "acceptance_receipt_sha256": lock["acceptance"][
                    "receipt_sha256"
                ],
                "boundary_fixture_manifest_sha256s": sorted(
                    fixture_manifests
                ),
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "onnx": onnx.__version__,
                "tensorrt": tensorrt.__version__,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(0)
                ),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "contract": {
                "positions": LOCAL_POSITIONS,
                "codebooks": CODEBOOKS,
                "frames_per_step": FRAMES_PER_STEP,
                "position_embedding": {
                    "kind": "learned_absolute",
                    "source_shape": list(position_shape),
                    "used_rows": list(range(LOCAL_POSITIONS)),
                    "dtype": "bf16",
                    "source_table_sha256": position_sha256,
                },
                "inputs": [
                    {
                        "name": name,
                        "dtype": dtype,
                        "shape": list(shape),
                    }
                    for name, (dtype, shape, mode) in expected_plan_contract().items()
                    if mode == "input"
                ],
                "outputs": [
                    {
                        "name": name,
                        "dtype": dtype,
                        "shape": list(shape),
                    }
                    for name, (dtype, shape, mode) in expected_plan_contract().items()
                    if mode == "output"
                ],
            },
            "pytorch_fixture_parity": {
                "status": "bit-exact-all-outputs",
                "boundary_fixture_manifest_sha256s": sorted(
                    fixture_manifests
                ),
            },
            "position_embedding_negative_test": {
                "status": "passed",
                "omitted_position_codes": no_position_codes,
            },
            "export": {
                "frontend": "torch.onnx legacy TorchScript exporter",
                "opset": ONNX_OPSET,
                "constant_folding": False,
                "external_data": False,
                "onnx": onnx_metadata,
            },
            "plugin": plugin_metadata(plugin),
            "build": build_metadata,
            "plan_inspection": plan_metadata,
            "plan_fixture_parity": plan_parity,
            "benchmark": benchmark,
            "artifacts": [
                artifact_record(path, staging)
                for path in (onnx_path, plan_path, plugin_path)
            ],
        }
        receipt_path = staging / RECEIPT_NAME
        receipt_payload = canonical_json_bytes(receipt)
        receipt_path.write_bytes(receipt_payload)
        (staging / RECEIPT_DIGEST_NAME).write_text(
            f"{hashlib.sha256(receipt_payload).hexdigest()}  {RECEIPT_NAME}\n",
            encoding="ascii",
        )
        publish_directory_no_replace(staging, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "plan_sha256": sha256_file(output / PLAN_NAME),
                    "plugin_sha256": sha256_file(output / PLUGIN_NAME),
                    "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
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
        TypeError,
        ValueError,
    ) as error:
        print(f"Local AR export failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

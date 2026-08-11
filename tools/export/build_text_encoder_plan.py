#!/usr/bin/env python3
"""Build, inspect, and validate the locked Text Encoder TensorRT plan."""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import math
import platform
import shutil
import sys
import tempfile
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from export_text_encoder import (  # noqa: E402
    MODEL_WIDTH,
    ONNX_FILE_NAME,
    PROFILE_MAX_T,
    PROFILE_MIN_T,
    PROFILE_OPT_T,
    RECEIPT_DIGEST_FILE_NAME,
    RECEIPT_FILE_NAME,
    TEXT_CONDITION,
    TEXT_MASK,
    TEXT_TOKEN_IDS,
    canonical_json_bytes,
    publish_directory_no_replace,
    require_json_document,
    require_manifest_checksum,
    sha256_bytes,
    sha256_file,
    verify_onnx_contract,
    verify_text_encoder_fixture,
)
from validate_boundary_fixture import validate_boundary_fixture  # noqa: E402


PLAN_FILE_NAME = "text_encoder.plan"
BUILD_REPORT_FILE_NAME = "builder-report.json"
PLAN_RECEIPT_FILE_NAME = "plan-receipt.json"
PLAN_RECEIPT_DIGEST_FILE_NAME = "plan-receipt.json.sha256"
MAX_ABSOLUTE_ERROR = 0.125
MAX_MEAN_ABSOLUTE_ERROR = 0.00125
MAX_P99_ABSOLUTE_ERROR = 0.006
MIN_COSINE_SIMILARITY = 0.99998
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


def require_canonical_checksum(
    root: Path,
    file_name: str,
    checksum_name: str,
) -> tuple[bytes, str]:
    payload_path = root / file_name
    checksum_path = root / checksum_name
    if payload_path.is_symlink() or checksum_path.is_symlink():
        raise RuntimeError(f"{file_name} and its checksum must not be symbolic links")
    payload = payload_path.resolve(strict=True).read_bytes()
    digest = sha256_bytes(payload)
    expected = f"{digest}  {file_name}\n"
    actual = checksum_path.resolve(strict=True).read_text(encoding="ascii")
    if actual != expected:
        raise RuntimeError(f"{checksum_name} does not exactly match {file_name}")
    return payload, digest


def register_plugin(path: Path) -> tuple[ctypes.CDLL, dict]:
    if path.is_symlink():
        raise RuntimeError(f"plugin must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(
            f"plugin must be a regular non-symlink file: {path}"
        )
    library = ctypes.CDLL(str(resolved), mode=ctypes.RTLD_LOCAL)
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
            f"plugin creator count mismatch: {api.creator_count}"
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
            f"plugin registration failed with status {registration_status}"
        )
    metadata = {
        "abi_version": api.abi_version,
        "registration_status": registration_status,
        "creators": [
            {
                "name": name,
                "version": version,
                "namespace": namespace,
            }
            for name, version, namespace in creators
        ],
    }
    return library, metadata


def verify_export_directory(
    export_path: Path,
    lock: dict,
    lock_sha256: str,
    fixture_manifest_sha256: str,
) -> tuple[Path, str]:
    if export_path.is_symlink():
        raise RuntimeError(
            f"export directory must not be a symbolic link: {export_path}"
        )
    root = export_path.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"export path is not a directory: {export_path}")
    expected_names = {
        ONNX_FILE_NAME,
        RECEIPT_FILE_NAME,
        RECEIPT_DIGEST_FILE_NAME,
    }
    actual_names = {entry.name for entry in root.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(
            "Text Encoder export entries mismatch: "
            f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in root.iterdir()):
        raise RuntimeError(
            "Text Encoder export entries must be regular non-symlink files"
        )
    receipt_payload, receipt_sha256 = require_canonical_checksum(
        root,
        RECEIPT_FILE_NAME,
        RECEIPT_DIGEST_FILE_NAME,
    )
    try:
        receipt = json.loads(receipt_payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"export receipt is not valid JSON: {error}") from error
    if not isinstance(receipt, dict):
        raise RuntimeError("export receipt must be a JSON object")
    if receipt.get("status") != "accepted":
        raise RuntimeError("text encoder export receipt is not accepted")
    source = receipt.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("export receipt source must be a JSON object")
    required_source = {
        "oracle_lock_sha256": lock_sha256,
        "boundary_fixture_manifest_sha256": fixture_manifest_sha256,
        "locked_magpie_restore_sha256": sha256_file(
            (SCRIPT_DIRECTORY / "locked_magpie_restore.py").resolve(strict=True)
        ),
        "codec_restore": {
            "embedded_codec_model_id": lock["codec"]["model_id"],
            "codec_model_sha256": lock["codec"]["sha256"],
            "codec_model_size_bytes": lock["codec"]["size_bytes"],
            "codec_resolution": "authenticated_local_file",
            "use_scl_loss": False,
            "network_resolution": False,
        },
    }
    for key, expected in required_source.items():
        actual = source.get(key)
        if actual != expected:
            raise RuntimeError(
                f"export receipt {key} mismatch: expected {expected}, got {actual}"
            )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise RuntimeError("export receipt must declare exactly one ONNX artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or artifact.get("path") != ONNX_FILE_NAME:
        raise RuntimeError("export receipt does not identify text_encoder.onnx")
    onnx_path = (root / ONNX_FILE_NAME).resolve(strict=True)
    if artifact.get("size_bytes") != onnx_path.stat().st_size:
        raise RuntimeError("ONNX size does not match its export receipt")
    onnx_sha256 = sha256_file(onnx_path)
    if artifact.get("sha256") != onnx_sha256:
        raise RuntimeError("ONNX SHA-256 does not match its export receipt")
    verify_onnx_contract(onnx_path)
    return onnx_path, receipt_sha256


def build_plan(
    trt,
    onnx_path: Path,
    plan_path: Path,
    report_path: Path,
) -> dict:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.resolve(strict=True).read_bytes()):
        errors = [
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        ]
        raise RuntimeError(
            "TensorRT Text Encoder ONNX parse failed:\n"
            + "\n".join(errors)
        )
    profile = builder.create_optimization_profile()
    expected_profile = (
        (1, PROFILE_MIN_T),
        (1, PROFILE_OPT_T),
        (1, PROFILE_MAX_T),
    )
    for name in (TEXT_TOKEN_IDS, TEXT_MASK):
        profile.set_shape(
            name,
            expected_profile[0],
            expected_profile[1],
            expected_profile[2],
        )
        actual_profile = tuple(
            tuple(shape) for shape in profile.get_shape(name)
        )
        if actual_profile != expected_profile:
            raise RuntimeError(
                "TensorRT rejected Text Encoder profile for "
                f"{name}: {actual_profile}"
            )
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 5
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    tactic_source_names = (
        "CUBLAS",
        "CUBLAS_LT",
        "CUDNN",
    )
    tactic_sources = 0
    for name in tactic_source_names:
        tactic_sources |= 1 << int(getattr(trt.TacticSource, name))
    if not config.set_tactic_sources(tactic_sources):
        raise RuntimeError(
            "TensorRT rejected the locked Text Encoder tactic sources"
        )
    if config.get_tactic_sources() != tactic_sources:
        raise RuntimeError(
            "TensorRT Text Encoder tactic-source readback mismatch: "
            f"expected={tactic_sources}, actual={config.get_tactic_sources()}"
        )
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        4 * 1024 * 1024 * 1024,
    )
    if config.add_optimization_profile(profile) != 0:
        raise RuntimeError("Text Encoder optimization profile index is not zero")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the Text Encoder plan")
    plan_path.write_bytes(bytes(serialized))
    metadata = {
        "network_flags": ["strongly_typed"],
        "precision": "bf16_from_onnx_types",
        "tf32": False,
        "builder_optimization_level": 5,
        "profiling_verbosity": "DETAILED",
        "tactic_sources": list(tactic_source_names),
        "workspace_bytes": 4 * 1024 * 1024 * 1024,
        "optimization_profile": {
            "name": "text_1_512",
            TEXT_TOKEN_IDS: {
                "min": [1, PROFILE_MIN_T],
                "opt": [1, PROFILE_OPT_T],
                "max": [1, PROFILE_MAX_T],
            },
            TEXT_MASK: {
                "min": [1, PROFILE_MIN_T],
                "opt": [1, PROFILE_OPT_T],
                "max": [1, PROFILE_MAX_T],
            },
        },
    }
    report_path.write_bytes(canonical_json_bytes(metadata))
    if not plan_path.is_file() or plan_path.stat().st_size == 0:
        raise RuntimeError(
            "TensorRT builder did not produce a non-empty Text Encoder plan"
        )
    return metadata


def read_fixture_tensor(torch, tensor):
    dtype_by_name = {
        "int32": torch.int32,
        "bool": torch.bool,
        "bf16": torch.uint16,
    }
    dtype = dtype_by_name.get(tensor.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported Text Encoder fixture dtype: {tensor.dtype}")
    value = torch.frombuffer(bytearray(tensor.path.read_bytes()), dtype=dtype)
    value = value.reshape(tensor.shape).clone()
    if tensor.dtype == "bf16":
        value = value.view(torch.bfloat16)
    return value.to(device="cuda")


def parity_passes(metrics: dict[str, float]) -> bool:
    return (
        math.isfinite(metrics["max_absolute_error"])
        and metrics["max_absolute_error"] <= MAX_ABSOLUTE_ERROR
        and math.isfinite(metrics["mean_absolute_error"])
        and metrics["mean_absolute_error"] <= MAX_MEAN_ABSOLUTE_ERROR
        and math.isfinite(metrics["p99_absolute_error"])
        and metrics["p99_absolute_error"] <= MAX_P99_ABSOLUTE_ERROR
        and math.isfinite(metrics["cosine_similarity"])
        and metrics["cosine_similarity"] >= MIN_COSINE_SIMILARITY
    )


def inspect_and_validate_plan(
    plan_path: Path,
    fixture,
) -> tuple[dict, dict[str, float | int]]:
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU plan validation is forbidden")
    if torch.cuda.get_device_capability(0) != (11, 0):
        raise RuntimeError(
            "Text Encoder plan validation requires Thor sm_110, got "
            f"{torch.cuda.get_device_capability(0)}"
        )
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the Text Encoder plan")
    if engine.num_optimization_profiles != 1:
        raise RuntimeError(
            f"expected one optimization profile, got {engine.num_optimization_profiles}"
        )
    expected_io = {
        TEXT_TOKEN_IDS: {
            "mode": trt.TensorIOMode.INPUT,
            "dtype": trt.int32,
            "shape": (1, -1),
        },
        TEXT_MASK: {
            "mode": trt.TensorIOMode.INPUT,
            "dtype": trt.bool,
            "shape": (1, -1),
        },
        TEXT_CONDITION: {
            "mode": trt.TensorIOMode.OUTPUT,
            "dtype": trt.bfloat16,
            "shape": (1, -1, MODEL_WIDTH),
        },
    }
    actual_names = {
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    }
    if actual_names != set(expected_io):
        raise RuntimeError(
            f"TensorRT I/O names mismatch: expected {sorted(expected_io)}, "
            f"got {sorted(actual_names)}"
        )
    normalized_io: list[dict] = []
    for name in (TEXT_TOKEN_IDS, TEXT_MASK, TEXT_CONDITION):
        expected = expected_io[name]
        actual_mode = engine.get_tensor_mode(name)
        actual_dtype = engine.get_tensor_dtype(name)
        actual_shape = tuple(engine.get_tensor_shape(name))
        if (
            actual_mode != expected["mode"]
            or actual_dtype != expected["dtype"]
            or actual_shape != expected["shape"]
        ):
            raise RuntimeError(
                f"TensorRT contract mismatch for {name}: "
                f"mode={actual_mode}, dtype={actual_dtype}, shape={actual_shape}"
            )
        io_record = {
            "name": name,
            "mode": "input" if actual_mode == trt.TensorIOMode.INPUT else "output",
            "dtype": {
                trt.int32: "int32",
                trt.bool: "bool",
                trt.bfloat16: "bf16",
            }[actual_dtype],
            "shape": list(actual_shape),
        }
        if actual_mode == trt.TensorIOMode.INPUT:
            profile = tuple(
                tuple(dimensions)
                for dimensions in engine.get_tensor_profile_shape(name, 0)
            )
            expected_profile = (
                (1, PROFILE_MIN_T),
                (1, PROFILE_OPT_T),
                (1, PROFILE_MAX_T),
            )
            if profile != expected_profile:
                raise RuntimeError(
                    f"TensorRT profile mismatch for {name}: "
                    f"expected {expected_profile}, got {profile}"
                )
            io_record["profile"] = {
                "min": list(profile[0]),
                "opt": list(profile[1]),
                "max": list(profile[2]),
            }
        normalized_io.append(io_record)

    token_ids = read_fixture_tensor(torch, fixture.token_ids)
    mask = read_fixture_tensor(torch, fixture.mask)
    expected_output = read_fixture_tensor(torch, fixture.condition)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT failed to create a Text Encoder context")
    if not context.set_input_shape(TEXT_TOKEN_IDS, tuple(token_ids.shape)):
        raise RuntimeError("TensorRT rejected the fixture token shape")
    if not context.set_input_shape(TEXT_MASK, tuple(mask.shape)):
        raise RuntimeError("TensorRT rejected the fixture mask shape")
    output = torch.empty_like(expected_output)
    bindings = {
        TEXT_TOKEN_IDS: token_ids,
        TEXT_MASK: mask,
        TEXT_CONDITION: output,
    }
    for name, tensor in bindings.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"TensorRT rejected the address for {name}")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT Text Encoder fixture execution failed")
    stream.synchronize()

    absolute_error = (output.float() - expected_output.float()).abs()
    metrics = {
        "max_absolute_error": float(absolute_error.max().item()),
        "mean_absolute_error": float(absolute_error.mean().item()),
        "p99_absolute_error": float(torch.quantile(absolute_error, 0.99).item()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                output.float().flatten(),
                expected_output.float().flatten(),
                dim=0,
            ).item()
        ),
        "bit_mismatch_count": int(
            torch.count_nonzero(
                output.view(torch.uint16) != expected_output.view(torch.uint16)
            ).item()
        ),
        "value_count": output.numel(),
    }
    if not parity_passes(metrics):
        raise RuntimeError(f"TensorRT Text Encoder fixture parity failed: {metrics}")

    for _ in range(20):
        with torch.cuda.stream(stream):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT Text Encoder warmup failed")
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iterations = 200
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iterations):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT Text Encoder benchmark failed")
        end.record(stream)
    stream.synchronize()
    metrics["mean_device_latency_ms"] = float(start.elapsed_time(end) / iterations)
    metrics["benchmark_iterations"] = iterations
    inspection = {
        "tensorrt_version": trt.__version__,
        "optimization_profiles": engine.num_optimization_profiles,
        "io": normalized_io,
        "device_memory_size_bytes": engine.device_memory_size_v2,
    }
    return inspection, metrics


def build_plan_receipt(
    args: argparse.Namespace,
    lock: dict,
    lock_sha256: str,
    fixture_manifest_sha256: str,
    export_receipt_sha256: str,
    build_metadata: dict,
    plugin_metadata: dict,
    inspection: dict,
    metrics: dict[str, float | int],
    staging: Path,
) -> dict:
    import tensorrt as trt
    import torch

    plan_path = staging / PLAN_FILE_NAME
    report_path = staging / BUILD_REPORT_FILE_NAME
    return {
        "schema_version": 1,
        "artifact_role": "text_encoder_plan",
        "status": "measured-not-accepted",
        "reason": (
            "plan construction and canonical-fixture numeric parity are "
            "candidate evidence only; acceptance requires exact codec "
            "sequences over all predeclared Japanese fixtures"
        ),
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": {
            "builder_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "locked_magpie_restore_sha256": sha256_file(
                (SCRIPT_DIRECTORY / "locked_magpie_restore.py").resolve(
                    strict=True
                )
            ),
            "codec_restore": {
                "embedded_codec_model_id": lock["codec"]["model_id"],
                "codec_model_sha256": lock["codec"]["sha256"],
                "codec_model_size_bytes": lock["codec"]["size_bytes"],
                "codec_resolution": "authenticated_local_file",
                "use_scl_loss": False,
                "network_resolution": False,
            },
            "oracle_lock_sha256": lock_sha256,
            "boundary_fixture_manifest_sha256": fixture_manifest_sha256,
            "export_receipt_sha256": export_receipt_sha256,
            "onnx_sha256": sha256_file(
                args.export.resolve(strict=True) / ONNX_FILE_NAME
            ),
            "plugin_sha256": sha256_file(args.plugin.resolve(strict=True)),
        },
        "build": {
            **build_metadata,
            "plugin": plugin_metadata,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "tensorrt": trt.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "inspection": inspection,
        "parity": {
            "thresholds": {
                "max_absolute_error": MAX_ABSOLUTE_ERROR,
                "max_mean_absolute_error": MAX_MEAN_ABSOLUTE_ERROR,
                "max_p99_absolute_error": MAX_P99_ABSOLUTE_ERROR,
                "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            },
            "metrics": metrics,
            "passed": True,
            "acceptance_effect": "candidate_only",
        },
        "artifacts": [
            {
                "path": PLAN_FILE_NAME,
                "size_bytes": plan_path.stat().st_size,
                "sha256": sha256_file(plan_path),
            },
            {
                "path": BUILD_REPORT_FILE_NAME,
                "size_bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--tensorrt-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = require_json_document(args.lock, "oracle lock")
    lock_sha256 = sha256_file(args.lock.resolve(strict=True))
    validate_boundary_fixture(args.fixture, args.lock)
    fixture_manifest, fixture_manifest_sha256 = require_manifest_checksum(
        args.fixture.resolve(strict=True)
    )
    del fixture_manifest
    fixture = verify_text_encoder_fixture(
        args.fixture,
        lock,
        lock_sha256,
    )
    onnx_path, export_receipt_sha256 = verify_export_directory(
        args.export,
        lock,
        lock_sha256,
        fixture_manifest_sha256,
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        from export_main_decoder import import_tensorrt

        trt = import_tensorrt(args.tensorrt_python_path)
        plugin_library, plugin_metadata = register_plugin(args.plugin)

        plan_path = staging / PLAN_FILE_NAME
        report_path = staging / BUILD_REPORT_FILE_NAME
        build_metadata = build_plan(
            trt,
            onnx_path,
            plan_path,
            report_path,
        )
        inspection, metrics = inspect_and_validate_plan(plan_path, fixture)
        if plugin_library is None:
            raise AssertionError("plugin library ownership was lost")
        receipt = build_plan_receipt(
            args,
            lock,
            lock_sha256,
            fixture_manifest_sha256,
            export_receipt_sha256,
            build_metadata,
            plugin_metadata,
            inspection,
            metrics,
            staging,
        )
        payload = canonical_json_bytes(receipt)
        (staging / PLAN_RECEIPT_FILE_NAME).write_bytes(payload)
        (staging / PLAN_RECEIPT_DIGEST_FILE_NAME).write_text(
            f"{sha256_bytes(payload)}  {PLAN_RECEIPT_FILE_NAME}\n",
            encoding="ascii",
        )
        publish_directory_no_replace(staging, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "plan_sha256": receipt["artifacts"][0]["sha256"],
                    "receipt_sha256": sha256_bytes(payload),
                    "parity": metrics,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception:
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
        print(f"Text Encoder plan build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

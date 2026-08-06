#!/usr/bin/env python3
"""Export the locked MagpieTTS Text Encoder through the legacy ONNX path.

The exporter accepts no implicit asset locations. It verifies every oracle
input, checks the imported NeMo source location, proves PyTorch parity against
the authenticated boundary fixture, validates the ONNX graph contract, and
publishes the output directory atomically without replacing an existing path.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_TOOLS = PROJECT_ROOT / "tools" / "oracle"
EXPORT_TOOLS = PROJECT_ROOT / "tools" / "export"
if str(ORACLE_TOOLS) not in sys.path:
    sys.path.insert(0, str(ORACLE_TOOLS))
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))

from verify_oracle_lock import (  # noqa: E402
    FileExpectation,
    require_file,
    require_model_configs,
    require_source_checkout,
    sha256_file,
)
from validate_boundary_fixture import validate_boundary_fixture  # noqa: E402


ONNX_OPSET = 20
PROFILE_MIN_T = 1
PROFILE_OPT_T = 64
PROFILE_MAX_T = 512
MODEL_WIDTH = 768
TEXT_ENCODER_LAYERS = 6
TEXT_TOKEN_IDS = "text_token_ids"
TEXT_MASK = "text_mask"
TEXT_CONDITION = "text_condition"
ONNX_FILE_NAME = "text_encoder.onnx"
RECEIPT_FILE_NAME = "export-receipt.json"
RECEIPT_DIGEST_FILE_NAME = "export-receipt.json.sha256"


@dataclass(frozen=True)
class FixtureTensor:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TextEncoderFixture:
    manifest_sha256: str
    token_ids: FixtureTensor
    mask: FixtureTensor
    condition: FixtureTensor
    text_tokens: int


def canonical_json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_json_document(path: Path, label: str) -> dict:
    try:
        document = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
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
    )
    return lock, sha256_file(lock_path)


def require_manifest_checksum(fixture_root: Path) -> tuple[dict, str]:
    manifest_path = fixture_root / "manifest.json"
    checksum_path = fixture_root / "manifest.json.sha256"
    manifest_digest = sha256_file(manifest_path.resolve(strict=True))
    checksum_text = checksum_path.resolve(strict=True).read_text(encoding="ascii")
    expected_line = f"{manifest_digest}  manifest.json\n"
    if checksum_text != expected_line:
        raise RuntimeError(
            "boundary fixture manifest checksum file is not canonical or does "
            "not match manifest.json"
        )
    return (
        require_json_document(manifest_path, "boundary fixture manifest"),
        manifest_digest,
    )


def require_fixture_tensor(
    fixture_root: Path,
    records: dict[str, dict],
    name: str,
) -> FixtureTensor:
    record = records.get(name)
    if record is None:
        raise RuntimeError(f"boundary fixture is missing tensor {name}")
    expected_path = f"tensors/{name}.bin"
    if record.get("path") != expected_path:
        raise RuntimeError(
            f"boundary fixture tensor {name} path mismatch: "
            f"expected {expected_path}, got {record.get('path')!r}"
        )
    relative_path = Path(expected_path)
    tensor_path = (fixture_root / relative_path).resolve(strict=True)
    if not tensor_path.is_relative_to(fixture_root):
        raise RuntimeError(f"boundary fixture tensor escapes its root: {name}")
    shape_value = record.get("shape")
    if (
        not isinstance(shape_value, list)
        or not shape_value
        or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in shape_value
        )
    ):
        raise RuntimeError(f"boundary fixture tensor {name} has an invalid shape")
    size_bytes = record.get("size_bytes")
    digest = record.get("sha256")
    dtype = record.get("dtype")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise RuntimeError(f"boundary fixture tensor {name} has an invalid byte size")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"boundary fixture tensor {name} has an invalid SHA-256")
    if not isinstance(dtype, str):
        raise RuntimeError(f"boundary fixture tensor {name} has an invalid dtype")
    if tensor_path.stat().st_size != size_bytes:
        raise RuntimeError(f"boundary fixture tensor {name} size mismatch")
    actual_digest = sha256_file(tensor_path)
    if actual_digest != digest:
        raise RuntimeError(
            f"boundary fixture tensor {name} SHA-256 mismatch: "
            f"expected {digest}, got {actual_digest}"
        )
    return FixtureTensor(
        path=tensor_path,
        dtype=dtype,
        shape=tuple(shape_value),
        size_bytes=size_bytes,
        sha256=digest,
    )


def verify_text_encoder_fixture(
    fixture_path: Path,
    lock: dict,
    lock_sha256: str,
) -> TextEncoderFixture:
    fixture_root = fixture_path.resolve(strict=True)
    if not fixture_root.is_dir():
        raise RuntimeError(f"boundary fixture is not a directory: {fixture_root}")
    manifest, manifest_digest = require_manifest_checksum(fixture_root)
    expected_manifest_values = {
        "oracle_lock_sha256": lock_sha256,
        "model_sha256": lock["model"]["sha256"],
        "codec_model_sha256": lock["codec"]["sha256"],
        "acceptance_receipt_sha256": lock["acceptance"]["receipt_sha256"],
        "source_bundle_sha256": lock["oracle_source"]["optimized_source_bundle_sha256"],
    }
    for key, expected in expected_manifest_values.items():
        actual = manifest.get(key)
        if actual != expected:
            raise RuntimeError(
                f"boundary fixture {key} mismatch: expected {expected}, got {actual}"
            )
    raw_records = manifest.get("tensors")
    if not isinstance(raw_records, list):
        raise RuntimeError("boundary fixture tensors must be a JSON array")
    records: dict[str, dict] = {}
    for record in raw_records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise RuntimeError("boundary fixture contains an invalid tensor record")
        name = record["name"]
        if name in records:
            raise RuntimeError(f"boundary fixture contains duplicate tensor {name}")
        records[name] = record

    token_ids = require_fixture_tensor(fixture_root, records, "input.text_token_ids")
    mask = require_fixture_tensor(fixture_root, records, "text.mask")
    condition = require_fixture_tensor(fixture_root, records, "text.condition")
    if token_ids.dtype != "int32":
        raise RuntimeError(f"fixture token IDs must be int32, got {token_ids.dtype}")
    if mask.dtype != "bool":
        raise RuntimeError(f"fixture text mask must be bool, got {mask.dtype}")
    if condition.dtype != "bf16":
        raise RuntimeError(
            f"fixture text condition must be bf16, got {condition.dtype}"
        )
    if len(token_ids.shape) != 2 or token_ids.shape[0] != 1:
        raise RuntimeError(
            f"fixture token IDs must have shape [1,T], got {token_ids.shape}"
        )
    text_tokens = token_ids.shape[1]
    if text_tokens < PROFILE_MIN_T or text_tokens > PROFILE_MAX_T:
        raise RuntimeError(
            f"fixture text length {text_tokens} is outside "
            f"[{PROFILE_MIN_T},{PROFILE_MAX_T}]"
        )
    if mask.shape != token_ids.shape:
        raise RuntimeError(
            f"fixture mask shape {mask.shape} does not match {token_ids.shape}"
        )
    if condition.shape != (1, text_tokens, MODEL_WIDTH):
        raise RuntimeError(
            f"fixture condition must have shape [1,T,{MODEL_WIDTH}], "
            f"got {condition.shape}"
        )
    decoder_contract = manifest.get("decoder_contract")
    if (
        not isinstance(decoder_contract, dict)
        or decoder_contract.get("text_tokens") != text_tokens
    ):
        raise RuntimeError(
            "boundary fixture decoder text-token count does not match its tensors"
        )
    return TextEncoderFixture(
        manifest_sha256=manifest_digest,
        token_ids=token_ids,
        mask=mask,
        condition=condition,
        text_tokens=text_tokens,
    )


def require_imported_nemo_source(speech_root: Path, model_class) -> None:
    module = sys.modules.get(model_class.__module__)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("unable to resolve the imported MagpieTTS module path")
    expected = (
        speech_root.resolve(strict=True)
        / "nemo"
        / "collections"
        / "tts"
        / "models"
        / "magpietts.py"
    ).resolve(strict=True)
    actual = Path(module_file).resolve(strict=True)
    if actual != expected:
        raise RuntimeError(
            f"MagpieTTS was imported from the wrong source: expected {expected}, got {actual}"
        )


def tensor_from_fixture(torch, tensor: FixtureTensor):
    dtype_by_name = {
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
        "bf16": torch.uint16,
    }
    dtype = dtype_by_name[tensor.dtype]
    storage = torch.frombuffer(bytearray(tensor.path.read_bytes()), dtype=dtype)
    value = storage.reshape(tensor.shape).clone()
    if tensor.dtype == "bf16":
        value = value.view(torch.bfloat16)
    return value


def load_text_encoder(torch, model_path: Path, speech_root: Path, lock: dict):
    from nemo.collections.tts.models import MagpieTTSModel
    from text_encoder_wrapper import TextEncoderWrapper

    require_imported_nemo_source(speech_root, MagpieTTSModel)
    model = MagpieTTSModel.restore_from(
        str(model_path.resolve(strict=True)),
        map_location="cpu",
    )
    model.eval()
    if model.model_type != "decoder_ce":
        raise RuntimeError(f"expected decoder_ce model, got {model.model_type}")
    if len(model.encoder.layers) != TEXT_ENCODER_LAYERS:
        raise RuntimeError(
            f"expected {TEXT_ENCODER_LAYERS} Text Encoder layers, "
            f"got {len(model.encoder.layers)}"
        )
    embedding_rows, embedding_width = model.text_embedding.weight.shape
    expected_rows = lock["frontend"]["text_embedding_rows"]
    if embedding_rows != expected_rows or embedding_width != MODEL_WIDTH:
        raise RuntimeError(
            "Text Encoder embedding shape mismatch: "
            f"expected [{expected_rows},{MODEL_WIDTH}], "
            f"got [{embedding_rows},{embedding_width}]"
        )
    if model.encoder.use_moe:
        raise RuntimeError("the locked Text Encoder must not use MoE")

    module = TextEncoderWrapper(model.text_embedding, model.encoder)
    module.eval()
    module.to(device="cuda", dtype=torch.bfloat16)
    del model
    parameter_dtypes = {
        parameter.dtype
        for parameter in module.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.bfloat16}:
        raise RuntimeError(
            f"Text Encoder floating parameters must all be BF16, got {parameter_dtypes}"
        )
    return module


def require_pytorch_fixture_parity(torch, module, fixture: TextEncoderFixture):
    token_ids = tensor_from_fixture(torch, fixture.token_ids).to(
        device="cuda",
        dtype=torch.int32,
    )
    mask = tensor_from_fixture(torch, fixture.mask).to(
        device="cuda",
        dtype=torch.bool,
    )
    expected = tensor_from_fixture(torch, fixture.condition)
    with torch.inference_mode():
        output = module(token_ids, mask)
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"Text Encoder output must be BF16, got {output.dtype}")
    if tuple(output.shape) != tuple(expected.shape):
        raise RuntimeError(
            f"Text Encoder output shape mismatch: expected {tuple(expected.shape)}, "
            f"got {tuple(output.shape)}"
        )
    actual_bits = output.detach().cpu().contiguous().view(torch.uint16)
    expected_bits = expected.contiguous().view(torch.uint16)
    if not torch.equal(actual_bits, expected_bits):
        mismatch_count = int(torch.count_nonzero(actual_bits != expected_bits))
        maximum_error = float(
            torch.max(
                torch.abs(output.detach().float().cpu() - expected.float())
            ).item()
        )
        raise RuntimeError(
            "PyTorch Text Encoder does not reproduce the locked fixture: "
            f"{mismatch_count} BF16 values differ, max_abs_error={maximum_error}"
        )
    return token_ids, mask


def export_legacy_onnx(torch, module, token_ids, mask, output_path: Path) -> None:
    torch.onnx.export(
        module,
        (token_ids, mask),
        str(output_path),
        input_names=[TEXT_TOKEN_IDS, TEXT_MASK],
        output_names=[TEXT_CONDITION],
        opset_version=ONNX_OPSET,
        dynamo=False,
        external_data=False,
        dynamic_axes={
            TEXT_TOKEN_IDS: {1: "text_tokens"},
            TEXT_MASK: {1: "text_tokens"},
            TEXT_CONDITION: {1: "text_tokens"},
        },
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False,
    )


def onnx_dimensions(value_info) -> tuple[int | str, ...]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            raise RuntimeError(f"ONNX value {value_info.name} has an unknown dimension")
    return tuple(dimensions)


def verify_onnx_contract(onnx_path: Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(onnx_path.resolve(strict=True)), load_external_data=True)
    # ONNX 1.22 full_check invokes its reference shape inferencer, which does
    # not support BF16 Conv and rejects an otherwise valid BF16 graph. The
    # structural checker still validates the complete protobuf and operator
    # schemas. Dtypes and all externally visible shapes are checked below;
    # TensorRT parsing/build and fixture parity are separate acceptance gates.
    onnx.checker.check_model(model, full_check=False)
    opsets = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
    expected_opsets = {
        "ai.onnx": ONNX_OPSET,
        "magpie_tts_rt": 1,
    }
    if opsets != expected_opsets:
        raise RuntimeError(
            f"ONNX opset mismatch: expected {expected_opsets}, got {opsets}"
        )
    if any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
    ):
        raise RuntimeError("external ONNX initializer data is not accepted")
    custom_nodes: dict[str, int] = {}
    complex_types = {onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128}
    for node in model.graph.node:
        if node.domain == "magpie_tts_rt":
            custom_nodes[node.op_type] = custom_nodes.get(node.op_type, 0) + 1
            continue
        if node.domain:
            raise RuntimeError(
                f"non-standard ONNX node domain is not accepted: "
                f"{node.domain}:{node.op_type}"
            )
        if node.op_type != "Cast":
            continue
        destination_types = [
            attribute.i for attribute in node.attribute if attribute.name == "to"
        ]
        if any(destination in complex_types for destination in destination_types):
            raise RuntimeError(
                f"complex ONNX Cast is not accepted: {node.name or '<unnamed>'}"
            )
    for initializer in model.graph.initializer:
        if initializer.data_type in complex_types:
            raise RuntimeError(
                f"complex ONNX initializer is not accepted: {initializer.name}"
            )
    expected_custom_nodes = {
        "MagpieLayerNorm": TEXT_ENCODER_LAYERS * 2 + 1,
        "MagpieGeluTanh": TEXT_ENCODER_LAYERS,
        "MagpieSoftmax": TEXT_ENCODER_LAYERS,
    }
    if custom_nodes != expected_custom_nodes:
        raise RuntimeError(
            "Text Encoder custom-node count mismatch: "
            f"expected {expected_custom_nodes}, got {custom_nodes}"
        )
    native_convolutions = [
        node for node in model.graph.node if node.op_type == "Conv"
    ]
    if (
        len(native_convolutions) != TEXT_ENCODER_LAYERS
        or any("o_net" in node.name for node in native_convolutions)
    ):
        raise RuntimeError(
            "Text Encoder must retain exactly the six input FFN "
            "convolutions; every output FFN convolution must be lowered to "
            "the exact im2col/MatMul graph"
        )
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if set(inputs) != {TEXT_TOKEN_IDS, TEXT_MASK}:
        raise RuntimeError(f"ONNX input names mismatch: got {sorted(inputs)}")
    if set(outputs) != {TEXT_CONDITION}:
        raise RuntimeError(f"ONNX output names mismatch: got {sorted(outputs)}")
    expected_values = {
        TEXT_TOKEN_IDS: (onnx.TensorProto.INT32, (1, "text_tokens")),
        TEXT_MASK: (onnx.TensorProto.BOOL, (1, "text_tokens")),
        TEXT_CONDITION: (
            onnx.TensorProto.BFLOAT16,
            (1, "text_tokens", MODEL_WIDTH),
        ),
    }
    for name, (expected_dtype, expected_shape) in expected_values.items():
        value = inputs.get(name, outputs.get(name))
        if value.type.tensor_type.elem_type != expected_dtype:
            raise RuntimeError(
                f"ONNX {name} dtype mismatch: expected {expected_dtype}, "
                f"got {value.type.tensor_type.elem_type}"
            )
        actual_shape = onnx_dimensions(value)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"ONNX {name} shape mismatch: expected {expected_shape}, "
                f"got {actual_shape}"
            )
    return {
        "graph_nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "custom_node_count": sum(custom_nodes.values()),
    }


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


def build_receipt(
    lock: dict,
    lock_sha256: str,
    fixture: TextEncoderFixture,
    onnx_path: Path,
    onnx_metadata: dict[str, int],
    torch,
) -> dict:
    import onnx

    return {
        "schema_version": 1,
        "artifact_role": "text_encoder",
        "status": "accepted",
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": {
            "exporter_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "wrapper_sha256": sha256_file(
                (
                    Path(__file__).parent / "text_encoder_wrapper.py"
                ).resolve(strict=True)
            ),
            "oracle_lock_sha256": lock_sha256,
            "oracle_source_revision": lock["oracle_source"]["base_revision"],
            "oracle_source_bundle_sha256": lock["oracle_source"][
                "optimized_source_bundle_sha256"
            ],
            "model_sha256": lock["model"]["sha256"],
            "codec_model_sha256": lock["codec"]["sha256"],
            "acceptance_receipt_sha256": lock["acceptance"]["receipt_sha256"],
            "boundary_fixture_manifest_sha256": fixture.manifest_sha256,
        },
        "export": {
            "frontend": "torch.onnx legacy TorchScript exporter",
            "dynamo": False,
            "opset": ONNX_OPSET,
            "constant_folding": True,
            "external_data": False,
            "pytorch_fixture_parity": "bit_exact_bf16",
            "bool_mask_lowering": "where_bool_bf16_one_zero",
            "onnx_checker": "structural",
            "onnx_full_shape_inference": "unsupported_for_bf16_conv_in_onnx_1.22",
            "causal_output_projection": (
                "kernel3-im2col-bf16-matmul-v1"
            ),
            "oracle_math_plugins": [
                {
                    "name": "MagpieLayerNorm",
                    "version": "1",
                    "namespace": "magpie_tts_rt",
                },
                {
                    "name": "MagpieGeluTanh",
                    "version": "1",
                    "namespace": "magpie_tts_rt",
                },
                {
                    "name": "MagpieSoftmax",
                    "version": "1",
                    "namespace": "magpie_tts_rt",
                },
            ],
        },
        "contract": {
            "inputs": [
                {
                    "name": TEXT_TOKEN_IDS,
                    "dtype": "int32",
                    "shape": [1, "T"],
                },
                {
                    "name": TEXT_MASK,
                    "dtype": "bool",
                    "shape": [1, "T"],
                },
            ],
            "outputs": [
                {
                    "name": TEXT_CONDITION,
                    "dtype": "bf16",
                    "shape": [1, "T", MODEL_WIDTH],
                }
            ],
            "optimization_profile": {
                "name": "text_1_512",
                "T": {
                    "min": PROFILE_MIN_T,
                    "opt": PROFILE_OPT_T,
                    "max": PROFILE_MAX_T,
                },
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "onnx": onnx.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "artifacts": [
            {
                "path": ONNX_FILE_NAME,
                "size_bytes": onnx_path.stat().st_size,
                "sha256": sha256_file(onnx_path),
                "graph_nodes": onnx_metadata["graph_nodes"],
                "initializers": onnx_metadata["initializers"],
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-model", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock, lock_sha256 = verify_locked_inputs(args)
    validate_boundary_fixture(args.fixture, args.lock)
    fixture = verify_text_encoder_fixture(args.fixture, lock, lock_sha256)

    speech_root = args.speech_root.resolve(strict=True)
    if str(speech_root) in sys.path:
        sys.path.remove(str(speech_root))
    sys.path.insert(0, str(speech_root))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU export is forbidden")
    if torch.cuda.get_device_capability(0) != (11, 0):
        raise RuntimeError(
            "Text Encoder export requires the accepted Thor sm_110 device, got "
            f"{torch.cuda.get_device_capability(0)}"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        module = load_text_encoder(torch, args.model, speech_root, lock)
        token_ids, mask = require_pytorch_fixture_parity(torch, module, fixture)
        onnx_path = staging / ONNX_FILE_NAME
        export_legacy_onnx(torch, module, token_ids, mask, onnx_path)
        unexpected = sorted(
            path.name for path in staging.iterdir() if path.name != ONNX_FILE_NAME
        )
        if unexpected:
            raise RuntimeError(f"ONNX export created unexpected files: {unexpected}")
        onnx_metadata = verify_onnx_contract(onnx_path)
        receipt = build_receipt(
            lock,
            lock_sha256,
            fixture,
            onnx_path,
            onnx_metadata,
            torch,
        )
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
                    "onnx_sha256": receipt["artifacts"][0]["sha256"],
                    "receipt_sha256": sha256_bytes(receipt_payload),
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
        print(f"Text Encoder export failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

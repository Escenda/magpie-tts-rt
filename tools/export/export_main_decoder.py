#!/usr/bin/env python3
"""Export, build, and measure the locked Sofia Main Decoder engines.

The generated receipt deliberately reports TensorRT equivalence as
``measured-not-accepted``.  A single fixture cannot define a numerical
tolerance.  The receipt records the measurements needed to design that gate
without silently accepting the observed error.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import errno
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from nemo.collections.tts.models import MagpieTTSModel

from main_decoder_wrapper import (
    AUDIO_CODEBOOKS,
    CROSS_HEADS,
    CROSS_HEAD_WIDTH,
    DECODER_LAYERS,
    FRAME_STACKING,
    MODEL_WIDTH,
    PREFILL_LENGTH,
    PRIOR_LAYERS,
    SELF_CACHE_CAPACITY,
    SELF_HEADS,
    SELF_HEAD_WIDTH,
    MainDecoderPrefillWrapper,
    MainDecoderStepWrapper,
    prefill_dynamic_axes,
    prefill_output_names,
    require_unique_names,
    step_dynamic_axes,
    step_input_names,
    step_output_names,
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


TARGET_DTYPE = torch.bfloat16
ONNX_OPSET = 20
MIN_TEXT_TOKENS = 1
OPT_TEXT_TOKENS = 64
MAX_TEXT_TOKENS = 512
EXAMPLE_POSITION = PREFILL_LENGTH
PREFILL_ONNX = "main-decoder-prefill.onnx"
STEP_ONNX = "main-decoder-step.onnx"
PREFILL_PLAN = "main-decoder-prefill.plan"
STEP_PLAN = "main-decoder-step.plan"
PREFILL_BUILD_REPORT = "main-decoder-prefill.builder-report.json"
STEP_BUILD_REPORT = "main-decoder-step.builder-report.json"
RECEIPT = "export-receipt.json"
RECEIPT_CHECKSUM = "export-receipt.json.sha256"


@dataclass(frozen=True)
class FixtureTensor:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DecoderFixture:
    root: Path
    manifest_sha256: str
    text_tokens: int
    tensors: dict[str, FixtureTensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-model", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-python-path",
        type=Path,
        required=True,
        help="Directory containing the TensorRT Python package used for plan inspection.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical_json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_locked_inputs(args: argparse.Namespace) -> tuple[dict, str]:
    lock_path = args.lock.resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model_lock = lock["model"]
    codec_lock = lock["codec"]
    acceptance_lock = lock["acceptance"]
    source_lock = lock["oracle_source"]
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


def require_imported_nemo_source(speech_root: Path) -> None:
    module = sys.modules.get(MagpieTTSModel.__module__)
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str):
        raise RuntimeError("unable to resolve imported MagpieTTS source")
    expected = (
        speech_root.resolve(strict=True)
        / "nemo"
        / "collections"
        / "tts"
        / "models"
        / "magpietts.py"
    ).resolve(strict=True)
    actual = Path(module_path).resolve(strict=True)
    if actual != expected:
        raise RuntimeError(
            f"MagpieTTS imported from the wrong source: expected {expected}, got {actual}"
        )


def load_fixture(path: Path, lock_path: Path) -> DecoderFixture:
    root = path.resolve(strict=True)
    tensor_count = validate_boundary_fixture(root, lock_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_records = manifest["tensors"]
    if tensor_count != len(raw_records):
        raise RuntimeError("fixture validator count does not match manifest")
    records: dict[str, FixtureTensor] = {}
    for raw in raw_records:
        name = raw["name"]
        records[name] = FixtureTensor(
            path=(root / raw["path"]).resolve(strict=True),
            dtype=raw["dtype"],
            shape=tuple(raw["shape"]),
            size_bytes=raw["size_bytes"],
            sha256=raw["sha256"],
        )
    decoder_contract = manifest.get("decoder_contract")
    if not isinstance(decoder_contract, dict):
        raise RuntimeError("fixture decoder_contract is absent")
    required_contract = {
        "prefill_positions": PREFILL_LENGTH,
        "next_position_after_step_001": PREFILL_LENGTH + 1,
        "attention_lookahead": 6,
        "attention_sink_threshold": 4,
        "unconditional_condition_mask_true_indices": [0],
    }
    for key, expected in required_contract.items():
        actual = decoder_contract.get(key)
        if actual != expected:
            raise RuntimeError(
                f"fixture decoder_contract.{key} mismatch: "
                f"expected {expected}, got {actual}"
            )
    text_tokens = decoder_contract.get("text_tokens")
    if not isinstance(text_tokens, int) or not (
        MIN_TEXT_TOKENS <= text_tokens <= MAX_TEXT_TOKENS
    ):
        raise RuntimeError(f"invalid fixture text-token count: {text_tokens}")
    fixture = DecoderFixture(
        root=root,
        manifest_sha256=sha256_file(manifest_path),
        text_tokens=text_tokens,
        tensors=records,
    )
    require_fixture_contract(fixture)
    return fixture


def require_tensor(
    fixture: DecoderFixture,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> None:
    tensor = fixture.tensors.get(name)
    if tensor is None:
        raise RuntimeError(f"fixture tensor is missing: {name}")
    if tensor.dtype != dtype or tensor.shape != shape:
        raise RuntimeError(
            f"fixture tensor {name} mismatch: expected {dtype} {shape}, "
            f"got {tensor.dtype} {tensor.shape}"
        )


def require_fixture_contract(fixture: DecoderFixture) -> None:
    text_tokens = fixture.text_tokens
    require_tensor(
        fixture, "cfg.conditional_condition", "bf16", (1, text_tokens, MODEL_WIDTH)
    )
    require_tensor(
        fixture, "cfg.unconditional_condition", "bf16", (1, text_tokens, MODEL_WIDTH)
    )
    require_tensor(fixture, "cfg.conditional_mask", "bool", (1, text_tokens))
    require_tensor(fixture, "cfg.unconditional_mask", "bool", (1, text_tokens))
    require_tensor(fixture, "prefill.hidden", "bf16", (2, 1, MODEL_WIDTH))
    require_tensor(fixture, "prefill.alignment", "bf16", (2, text_tokens))
    require_tensor(
        fixture,
        "local_ar.step_000.codes",
        "int64",
        (1, AUDIO_CODEBOOKS, FRAME_STACKING),
    )
    require_tensor(
        fixture, "step_000.next_prior", "bf16", (2, 1, text_tokens)
    )
    require_tensor(fixture, "step_001.hidden", "bf16", (2, 1, MODEL_WIDTH))
    require_tensor(fixture, "step_001.alignment", "bf16", (2, text_tokens))
    for layer_index in range(DECODER_LAYERS):
        for prefix, valid_positions in (
            ("prefill.state", PREFILL_LENGTH),
            ("step_001.state", PREFILL_LENGTH + 1),
        ):
            stem = f"{prefix}.layer_{layer_index:02d}"
            require_tensor(
                fixture,
                f"{stem}.self_key",
                "bf16",
                (2, valid_positions, SELF_HEADS, SELF_HEAD_WIDTH),
            )
            require_tensor(
                fixture,
                f"{stem}.self_value",
                "bf16",
                (2, valid_positions, SELF_HEADS, SELF_HEAD_WIDTH),
            )
            require_tensor(
                fixture,
                f"{stem}.self_mask",
                "bool",
                (2, valid_positions),
            )
            require_tensor(
                fixture,
                f"{stem}.cross_key",
                "bf16",
                (2, text_tokens, CROSS_HEADS, CROSS_HEAD_WIDTH),
            )
            require_tensor(
                fixture,
                f"{stem}.cross_value",
                "bf16",
                (2, text_tokens, CROSS_HEADS, CROSS_HEAD_WIDTH),
            )


def tensor_from_fixture(fixture: DecoderFixture, name: str) -> torch.Tensor:
    record = fixture.tensors[name]
    dtype_map = {
        "bf16": torch.uint16,
        "bool": torch.bool,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map.get(record.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported decoder fixture dtype: {record.dtype}")
    value = torch.frombuffer(bytearray(record.path.read_bytes()), dtype=dtype)
    value = value.reshape(record.shape).clone()
    if record.dtype == "bf16":
        value = value.view(torch.bfloat16)
    return value.to(device="cuda")


def pad_self_cache(value: torch.Tensor) -> torch.Tensor:
    shape = list(value.shape)
    shape[1] = SELF_CACHE_CAPACITY
    padded = torch.zeros(shape, dtype=value.dtype, device=value.device)
    padded[:, : value.size(1)].copy_(value)
    return padded


def fixture_inputs(
    fixture: DecoderFixture,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, ...],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    condition = torch.cat(
        [
            tensor_from_fixture(fixture, "cfg.conditional_condition"),
            tensor_from_fixture(fixture, "cfg.unconditional_condition"),
        ],
        dim=0,
    )
    condition_mask = torch.cat(
        [
            tensor_from_fixture(fixture, "cfg.conditional_mask"),
            tensor_from_fixture(fixture, "cfg.unconditional_mask"),
        ],
        dim=0,
    )
    prefill_expected: dict[str, torch.Tensor] = {
        "last_hidden": tensor_from_fixture(fixture, "prefill.hidden"),
        "alignment": tensor_from_fixture(fixture, "prefill.alignment"),
    }
    step_values: list[torch.Tensor] = [
        tensor_from_fixture(fixture, "local_ar.step_000.codes"),
        torch.tensor(EXAMPLE_POSITION, dtype=torch.int64, device="cuda"),
        tensor_from_fixture(fixture, "step_000.next_prior"),
        condition_mask,
    ]
    step_expected: dict[str, torch.Tensor] = {
        "decoder_hidden": tensor_from_fixture(fixture, "step_001.hidden")[:, 0],
        "alignment": tensor_from_fixture(fixture, "step_001.alignment"),
    }
    for layer_index in range(DECODER_LAYERS):
        prefill_stem = f"prefill.state.layer_{layer_index:02d}"
        step_stem = f"step_001.state.layer_{layer_index:02d}"
        prefill_layer = [
            pad_self_cache(
                tensor_from_fixture(fixture, f"{prefill_stem}.self_key")
            ),
            pad_self_cache(
                tensor_from_fixture(fixture, f"{prefill_stem}.self_value")
            ),
            pad_self_cache(
                tensor_from_fixture(fixture, f"{prefill_stem}.self_mask")
            ),
            tensor_from_fixture(fixture, f"{prefill_stem}.cross_key"),
            tensor_from_fixture(fixture, f"{prefill_stem}.cross_value"),
        ]
        step_values.extend(prefill_layer)
        for name, value in zip(
            (
                f"prefill_self_key_{layer_index}",
                f"prefill_self_value_{layer_index}",
                f"prefill_self_mask_{layer_index}",
                f"prefill_cross_key_{layer_index}",
                f"prefill_cross_value_{layer_index}",
            ),
            prefill_layer,
            strict=True,
        ):
            prefill_expected[name] = value
        step_expected[f"step_self_key_out_{layer_index}"] = pad_self_cache(
            tensor_from_fixture(fixture, f"{step_stem}.self_key")
        )
        step_expected[f"step_self_value_out_{layer_index}"] = pad_self_cache(
            tensor_from_fixture(fixture, f"{step_stem}.self_value")
        )
        step_expected[f"step_self_mask_out_{layer_index}"] = pad_self_cache(
            tensor_from_fixture(fixture, f"{step_stem}.self_mask")
        )
    return (
        (condition, condition_mask),
        tuple(step_values),
        prefill_expected,
        step_expected,
    )


def load_model(model_path: Path, speech_root: Path) -> MagpieTTSModel:
    require_imported_nemo_source(speech_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU export is forbidden")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if (
        torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("failed to establish the locked no-TF32 runtime policy")
    model = MagpieTTSModel.restore_from(
        str(model_path.resolve(strict=True)),
        map_location="cpu",
    )
    model.eval()
    model.to("cuda")
    for module in (
        model.decoder,
        model.baked_context_embedding,
        model.audio_embeddings,
    ):
        if module is None:
            raise RuntimeError("accepted Main Decoder module is absent")
        module.to(dtype=TARGET_DTYPE)
    return model


def require_bit_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise RuntimeError(
            f"{label} contract mismatch: expected {expected.dtype} "
            f"{tuple(expected.shape)}, got {actual.dtype} {tuple(actual.shape)}"
        )
    if actual.dtype == torch.bfloat16:
        equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    else:
        equal = torch.equal(actual, expected)
    if not equal:
        difference = (
            (actual.float() - expected.float()).abs()
            if actual.dtype == torch.bfloat16
            else actual != expected
        )
        raise RuntimeError(
            f"{label} is not bit-exact: "
            f"mismatches={int(torch.count_nonzero(difference))}, "
            f"max_abs={float(difference.max())}"
        )


def require_pytorch_fixture_parity(
    prefill: MainDecoderPrefillWrapper,
    step: MainDecoderStepWrapper,
    prefill_inputs: tuple[torch.Tensor, torch.Tensor],
    step_inputs: tuple[torch.Tensor, ...],
    prefill_expected: dict[str, torch.Tensor],
    step_expected: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        prefill_outputs = prefill(*prefill_inputs)
        step_outputs = step(*step_inputs)
    for name, actual in zip(
        prefill_output_names(), prefill_outputs, strict=True
    ):
        require_bit_exact(actual, prefill_expected[name], f"PyTorch prefill {name}")
    for name, actual in zip(step_output_names(), step_outputs, strict=True):
        require_bit_exact(actual, step_expected[name], f"PyTorch step {name}")


def export_graph(
    *,
    wrapper: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    output_path: Path,
) -> dict[str, int]:
    require_unique_names(input_names, "input names")
    require_unique_names(output_names, "output names")
    wrapper.eval()
    with torch.no_grad():
        outputs = wrapper(*example_inputs)
    if len(outputs) != len(output_names):
        raise RuntimeError(
            f"wrapper produced {len(outputs)} outputs for {len(output_names)} names"
        )
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(output_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        # PyTorch 2.11's legacy exporter aborts while constant-folding this
        # CUDA graph. TensorRT folds the retained constants after parsing.
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
    graph = onnx.load(str(output_path.resolve(strict=True)), load_external_data=True)
    onnx.checker.check_model(graph, full_check=False)
    if any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in graph.graph.initializer
    ):
        raise RuntimeError("external ONNX initializer data is not accepted")
    opsets = {
        entry.domain or "ai.onnx": entry.version
        for entry in graph.opset_import
    }
    expected_opsets = {
        "ai.onnx": ONNX_OPSET,
        "magpie_tts_rt": 1,
    }
    if opsets != expected_opsets:
        raise RuntimeError(
            f"ONNX opset mismatch: expected {expected_opsets}, got {opsets}"
        )
    custom_nodes: dict[str, int] = {}
    softmax_modes: dict[int, int] = {}
    for node in graph.graph.node:
        if node.domain == "magpie_tts_rt":
            custom_nodes[node.op_type] = custom_nodes.get(node.op_type, 0) + 1
            if node.op_type == "MagpieSoftmax":
                modes = [
                    attribute.i
                    for attribute in node.attribute
                    if attribute.name == "mode"
                ]
                if len(modes) != 1 or modes[0] not in (
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ):
                    raise RuntimeError(
                        "Main Decoder Softmax node must declare exactly one "
                        f"accepted mode: {node.name}"
                    )
                mode = int(modes[0])
                expected_inputs = {
                    0: 1,
                    1: 4,
                    2: 3,
                    3: 3,
                    4: 2,
                    5: 3,
                    6: 3,
                    7: 4,
                }[mode]
                if len(node.input) != expected_inputs:
                    raise RuntimeError(
                        f"Main Decoder Softmax mode {mode} input count "
                        f"mismatch: expected={expected_inputs}, "
                        f"actual={len(node.input)}"
                    )
                softmax_modes[mode] = softmax_modes.get(mode, 0) + 1
        elif node.domain:
            raise RuntimeError(
                f"non-standard ONNX node domain is not accepted: "
                f"{node.domain}:{node.op_type}"
            )
    if isinstance(wrapper, MainDecoderPrefillWrapper):
        expected_custom_nodes = {
            "MagpieLayerNorm": DECODER_LAYERS * 4 + 1,
            "MagpieGeluTanh": DECODER_LAYERS,
            "MagpieSoftmax": DECODER_LAYERS * 4 + 1,
        }
        expected_softmax_modes = {
            0: DECODER_LAYERS,
            1: DECODER_LAYERS,
            2: DECODER_LAYERS,
            3: DECODER_LAYERS,
            7: 1,
        }
    elif isinstance(wrapper, MainDecoderStepWrapper):
        expected_custom_nodes = {
            "MagpieLayerNorm": DECODER_LAYERS * 3 + 1,
            "MagpieGeluTanh": DECODER_LAYERS,
            "MagpieSoftmax": (
                DECODER_LAYERS * 5 + len(PRIOR_LAYERS) + 1
            ),
        }
        expected_softmax_modes = {
            0: DECODER_LAYERS,
            1: DECODER_LAYERS,
            3: DECODER_LAYERS,
            4: len(PRIOR_LAYERS),
            5: DECODER_LAYERS,
            6: DECODER_LAYERS,
            7: 1,
        }
    else:
        raise RuntimeError(
            f"unsupported Main Decoder wrapper type: {type(wrapper).__name__}"
        )
    if custom_nodes != expected_custom_nodes:
        raise RuntimeError(
            "Main Decoder custom-node count mismatch: "
            f"expected={expected_custom_nodes}, got={custom_nodes}"
        )
    if softmax_modes != expected_softmax_modes:
        raise RuntimeError(
            "Main Decoder Softmax mode count mismatch: "
            f"expected={expected_softmax_modes}, got={softmax_modes}"
        )
    native_convolutions = [
        node.name for node in graph.graph.node if node.op_type == "Conv"
    ]
    if native_convolutions:
        raise RuntimeError(
            "Main Decoder pointwise FFN convolutions must be lowered to "
            f"BF16 MatMul nodes, got native Conv nodes: {native_convolutions}"
        )
    complex_types = {onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128}
    for node in graph.graph.node:
        if node.op_type == "Cast" and any(
            attribute.name == "to" and attribute.i in complex_types
            for attribute in node.attribute
        ):
            raise RuntimeError(f"complex ONNX Cast is not accepted: {node.name}")
    actual_inputs = {value.name for value in graph.graph.input}
    actual_outputs = {value.name for value in graph.graph.output}
    if actual_inputs != set(input_names) or actual_outputs != set(output_names):
        raise RuntimeError(
            "ONNX external tensor names differ from the export contract"
        )
    return {
        "nodes": len(graph.graph.node),
        "initializers": len(graph.graph.initializer),
        "custom_nodes": custom_nodes,
        "softmax_modes": softmax_modes,
    }


def build_plan(
    *,
    tensorrt,
    role: str,
    onnx_path: Path,
    plan_path: Path,
    report_path: Path,
    tactic_source_names: tuple[str, ...] = ("CUBLAS", "CUBLAS_LT", "CUDNN"),
) -> dict[str, int | float | str | list[str]]:
    if role not in ("prefill", "step"):
        raise ValueError(f"unsupported engine role: {role}")
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(logger)
    network_flags = 1 << int(
        tensorrt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
    )
    network = builder.create_network(network_flags)
    parser = tensorrt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.resolve(strict=True).read_bytes()):
        errors = [
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        ]
        raise RuntimeError(
            f"TensorRT {role} ONNX parse failed:\n" + "\n".join(errors)
        )
    profile = builder.create_optimization_profile()
    profile_tokens = (
        MIN_TEXT_TOKENS,
        OPT_TEXT_TOKENS,
        MAX_TEXT_TOKENS,
    )
    for input_index in range(network.num_inputs):
        tensor = network.get_input(input_index)
        shape = tuple(tensor.shape)
        if role == "step" and tensor.name == "position":
            if not tensor.is_shape_tensor:
                raise RuntimeError(
                    "step position must be a TensorRT shape tensor"
                )
            position_values = (
                (PREFILL_LENGTH,),
                (
                    (PREFILL_LENGTH + SELF_CACHE_CAPACITY - 1)
                    // 2,
                ),
                (SELF_CACHE_CAPACITY - 1,),
            )
            profile.set_shape_input(
                tensor.name,
                position_values[0],
                position_values[1],
                position_values[2],
            )
            actual_position_values = tuple(
                tuple(value)
                for value in profile.get_shape_input(tensor.name)
            )
            if actual_position_values != position_values:
                raise RuntimeError(
                    "TensorRT step position profile readback mismatch: "
                    f"expected={position_values}, "
                    f"actual={actual_position_values}"
                )
            continue
        if -1 not in shape:
            continue
        if role == "prefill" and tensor.name == "condition":
            shapes = tuple((2, tokens, MODEL_WIDTH) for tokens in profile_tokens)
        elif role == "prefill" and tensor.name == "condition_mask":
            shapes = tuple((2, tokens) for tokens in profile_tokens)
        elif role == "step" and tensor.name == "alignment_prior":
            shapes = tuple((2, 1, tokens) for tokens in profile_tokens)
        elif role == "step" and tensor.name == "condition_mask":
            shapes = tuple((2, tokens) for tokens in profile_tokens)
        elif (
            role == "step"
            and (
                tensor.name.startswith("step_cross_key_in_")
                or tensor.name.startswith("step_cross_value_in_")
            )
        ):
            shapes = tuple(
                (2, tokens, CROSS_HEADS, CROSS_HEAD_WIDTH)
                for tokens in profile_tokens
            )
        else:
            raise RuntimeError(
                f"unexpected dynamic {role} input: "
                f"{tensor.name} shape={shape}"
            )
        profile.set_shape(tensor.name, shapes[0], shapes[1], shapes[2])
        actual = tuple(
            tuple(value) for value in profile.get_shape(tensor.name)
        )
        if actual != shapes:
            raise RuntimeError(
                f"TensorRT rejected {role} profile for {tensor.name}: "
                f"expected={shapes}, actual={actual}"
            )
    config = builder.create_builder_config()
    config.clear_flag(tensorrt.BuilderFlag.TF32)
    config.builder_optimization_level = 5
    config.profiling_verbosity = tensorrt.ProfilingVerbosity.DETAILED
    tactic_sources = 0
    for name in tactic_source_names:
        tactic_sources |= 1 << int(getattr(tensorrt.TacticSource, name))
    if not config.set_tactic_sources(tactic_sources):
        raise RuntimeError(
            f"TensorRT rejected the locked {role} tactic sources"
        )
    if config.get_tactic_sources() != tactic_sources:
        raise RuntimeError(
            f"TensorRT {role} tactic-source readback mismatch"
        )
    config.set_memory_pool_limit(
        tensorrt.MemoryPoolType.WORKSPACE,
        4 * 1024 * 1024 * 1024,
    )
    if config.add_optimization_profile(profile) != 0:
        raise RuntimeError(
            f"TensorRT {role} optimization profile index is not zero"
        )
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    elapsed_seconds = time.monotonic() - started
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build the {role} plan")
    plan_path.write_bytes(bytes(serialized))
    metadata: dict[str, int | float | str | list[str]] = {
        "elapsed_seconds": elapsed_seconds,
        "builder_optimization_level": 5,
        "profiling_verbosity": "DETAILED",
        "workspace_bytes": 4 * 1024 * 1024 * 1024,
        "tactic_sources": list(tactic_source_names),
    }
    if role == "step":
        metadata["position_profile_min"] = PREFILL_LENGTH
        metadata["position_profile_opt"] = (
            PREFILL_LENGTH + SELF_CACHE_CAPACITY - 1
        ) // 2
        metadata["position_profile_max"] = (
            SELF_CACHE_CAPACITY - 1
        )
    report_path.write_bytes(canonical_json_bytes(metadata))
    return metadata


def import_tensorrt(path: Path):
    package_path = path.resolve(strict=True)
    if not package_path.is_dir():
        raise RuntimeError(f"TensorRT Python path is not a directory: {package_path}")
    sys.path.insert(0, str(package_path))
    return importlib.import_module("tensorrt")


def expected_plan_contract(role: str) -> dict[str, tuple[str, tuple[int, ...], str]]:
    result: dict[str, tuple[str, tuple[int, ...], str]] = {}
    if role == "prefill":
        result["condition"] = ("bf16", (2, -1, MODEL_WIDTH), "input")
        result["condition_mask"] = ("bool", (2, -1), "input")
        result["last_hidden"] = ("bf16", (2, 1, MODEL_WIDTH), "output")
        result["alignment"] = ("bf16", (2, -1), "output")
        for layer_index in range(DECODER_LAYERS):
            result[f"prefill_self_key_{layer_index}"] = (
                "bf16",
                (2, SELF_CACHE_CAPACITY, SELF_HEADS, SELF_HEAD_WIDTH),
                "output",
            )
            result[f"prefill_self_value_{layer_index}"] = (
                "bf16",
                (2, SELF_CACHE_CAPACITY, SELF_HEADS, SELF_HEAD_WIDTH),
                "output",
            )
            result[f"prefill_self_mask_{layer_index}"] = (
                "bool",
                (2, SELF_CACHE_CAPACITY),
                "output",
            )
            result[f"prefill_cross_key_{layer_index}"] = (
                "bf16",
                (2, -1, CROSS_HEADS, CROSS_HEAD_WIDTH),
                "output",
            )
            result[f"prefill_cross_value_{layer_index}"] = (
                "bf16",
                (2, -1, CROSS_HEADS, CROSS_HEAD_WIDTH),
                "output",
            )
        return result
    if role != "step":
        raise ValueError(f"unsupported engine role: {role}")
    result["previous_codec_tokens"] = (
        "int64",
        (1, AUDIO_CODEBOOKS, FRAME_STACKING),
        "input",
    )
    result["position"] = ("int64", (), "input")
    result["alignment_prior"] = ("bf16", (2, 1, -1), "input")
    result["condition_mask"] = ("bool", (2, -1), "input")
    result["decoder_hidden"] = ("bf16", (2, MODEL_WIDTH), "output")
    result["alignment"] = ("bf16", (2, -1), "output")
    for layer_index in range(DECODER_LAYERS):
        for stem, dtype, shape in (
            (
                "self_key",
                "bf16",
                (2, SELF_CACHE_CAPACITY, SELF_HEADS, SELF_HEAD_WIDTH),
            ),
            (
                "self_value",
                "bf16",
                (2, SELF_CACHE_CAPACITY, SELF_HEADS, SELF_HEAD_WIDTH),
            ),
            ("self_mask", "bool", (2, SELF_CACHE_CAPACITY)),
            ("cross_key", "bf16", (2, -1, CROSS_HEADS, CROSS_HEAD_WIDTH)),
            ("cross_value", "bf16", (2, -1, CROSS_HEADS, CROSS_HEAD_WIDTH)),
        ):
            result[f"step_{stem}_in_{layer_index}"] = (dtype, shape, "input")
        for stem, dtype, shape in (
            (
                "self_key",
                "bf16",
                (2, SELF_CACHE_CAPACITY, SELF_HEADS, SELF_HEAD_WIDTH),
            ),
            (
                "self_value",
                "bf16",
                (2, SELF_CACHE_CAPACITY, SELF_HEADS, SELF_HEAD_WIDTH),
            ),
            ("self_mask", "bool", (2, SELF_CACHE_CAPACITY)),
        ):
            result[f"step_{stem}_out_{layer_index}"] = (dtype, shape, "output")
    return result


def inspect_plan(tensorrt, role: str, plan_path: Path) -> dict:
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT plan: {plan_path}")
    if engine.num_optimization_profiles != 1:
        raise RuntimeError(
            f"{role} plan must have one optimization profile, "
            f"got {engine.num_optimization_profiles}"
        )
    expected = expected_plan_contract(role)
    actual_names = {
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    }
    if actual_names != set(expected):
        raise RuntimeError(
            f"{role} plan I/O names mismatch: "
            f"missing={sorted(set(expected) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected))}"
        )
    dtype_map = {
        tensorrt.DataType.BF16: "bf16",
        tensorrt.DataType.BOOL: "bool",
        tensorrt.DataType.INT64: "int64",
    }
    location_map = {
        tensorrt.TensorLocation.DEVICE: "device",
        tensorrt.TensorLocation.HOST: "host",
    }
    tensors: list[dict] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        expected_dtype, expected_shape, expected_mode = expected[name]
        actual_dtype = dtype_map.get(engine.get_tensor_dtype(name))
        actual_shape = tuple(engine.get_tensor_shape(name))
        actual_mode = (
            "input"
            if engine.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT
            else "output"
        )
        actual_location = location_map.get(engine.get_tensor_location(name))
        actual_shape_inference_io = bool(
            engine.is_shape_inference_io(name)
        )
        expected_location = (
            "host"
            if role == "step" and name == "position"
            else "device"
        )
        expected_shape_inference_io = (
            role == "step" and name == "position"
        )
        if (actual_dtype, actual_shape, actual_mode) != (
            expected_dtype,
            expected_shape,
            expected_mode,
        ):
            raise RuntimeError(
                f"{role} plan tensor {name} mismatch: expected "
                f"{expected_dtype} {expected_shape} {expected_mode}, got "
                f"{actual_dtype} {actual_shape} {actual_mode}"
            )
        if (
            actual_location != expected_location
            or actual_shape_inference_io != expected_shape_inference_io
        ):
            raise RuntimeError(
                f"{role} plan tensor {name} execution contract mismatch: "
                f"expected location={expected_location} "
                f"shape_inference_io={expected_shape_inference_io}, got "
                f"location={actual_location} "
                f"shape_inference_io={actual_shape_inference_io}"
            )
        record: dict = {
            "name": name,
            "dtype": actual_dtype,
            "shape": list(actual_shape),
            "mode": actual_mode,
            "location": actual_location,
            "shape_inference_io": actual_shape_inference_io,
        }
        if actual_mode == "input" and -1 in actual_shape:
            profile = engine.get_tensor_profile_shape(name, 0)
            record["profile"] = {
                "min": list(profile[0]),
                "opt": list(profile[1]),
                "max": list(profile[2]),
            }
        if expected_shape_inference_io:
            profile_values = tuple(
                tuple(int(item) for item in values)
                for values in engine.get_tensor_profile_values(0, name)
            )
            expected_values = (
                (PREFILL_LENGTH,),
                (
                    (
                        PREFILL_LENGTH
                        + SELF_CACHE_CAPACITY
                        - 1
                    )
                    // 2,
                ),
                (SELF_CACHE_CAPACITY - 1,),
            )
            if profile_values != expected_values:
                raise RuntimeError(
                    "step position engine profile mismatch: "
                    f"expected={expected_values}, actual={profile_values}"
                )
            record["profile_values"] = {
                "min": list(profile_values[0]),
                "opt": list(profile_values[1]),
                "max": list(profile_values[2]),
            }
        tensors.append(record)
    return {
        "tensorrt_version": tensorrt.__version__,
        "optimization_profiles": engine.num_optimization_profiles,
        "tensors": tensors,
    }


def execute_plan(tensorrt, plan_path: Path, inputs: dict[str, torch.Tensor]):
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"failed to create execution context: {plan_path}")
    for name, value in inputs.items():
        if -1 in tuple(engine.get_tensor_shape(name)):
            if not context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"failed to set TensorRT input shape: {name}")
    bound_inputs: dict[str, torch.Tensor] = {}
    for name, value in inputs.items():
        if (
            engine.get_tensor_location(name)
            == tensorrt.TensorLocation.HOST
        ):
            source = value.detach().cpu().contiguous()
            value = torch.empty_like(
                source,
                device="cpu",
                pin_memory=True,
            )
            value.copy_(source)
            if value.data_ptr() % 256 != 0:
                raise RuntimeError(
                    f"TensorRT HOST input is not 256-byte aligned: {name}"
                )
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"failed to bind TensorRT input: {name}")
        bound_inputs[name] = value
    missing = context.infer_shapes()
    if missing:
        raise RuntimeError(f"TensorRT shape inference is incomplete: {missing}")
    dtype_map = {
        tensorrt.DataType.BF16: torch.bfloat16,
        tensorrt.DataType.BOOL: torch.bool,
        tensorrt.DataType.INT64: torch.int64,
    }
    outputs: dict[str, torch.Tensor] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT:
            value = bound_inputs[name]
        else:
            dtype = dtype_map.get(engine.get_tensor_dtype(name))
            if dtype is None:
                raise RuntimeError(f"unsupported TensorRT output dtype: {name}")
            value = torch.empty(
                tuple(context.get_tensor_shape(name)),
                dtype=dtype,
                device="cuda",
            )
            outputs[name] = value
        if (
            engine.get_tensor_mode(name)
            == tensorrt.TensorIOMode.OUTPUT
            and not context.set_tensor_address(name, value.data_ptr())
        ):
            raise RuntimeError(f"failed to bind TensorRT tensor: {name}")
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError(f"TensorRT execution failed: {plan_path}")
    torch.cuda.synchronize()
    return outputs


def numerical_metrics(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, int | float]:
    actual = torch.cat([left.reshape(-1).float() for left, _ in pairs])
    expected = torch.cat([right.reshape(-1).float() for _, right in pairs])
    difference = (actual - expected).abs()
    mismatch_count = sum(
        int(torch.count_nonzero(left.view(torch.int16) != right.view(torch.int16)))
        for left, right in pairs
    )
    element_count = sum(left.numel() for left, _ in pairs)
    norm_product = torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(
        expected
    )
    cosine = (
        float(torch.dot(actual, expected) / norm_product)
        if float(norm_product) > 0
        else (1.0 if torch.equal(actual, expected) else 0.0)
    )
    return {
        "elements": element_count,
        "bit_mismatch_count": mismatch_count,
        "bit_mismatch_ratio": mismatch_count / element_count,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(torch.quantile(difference, 0.99)),
        "cosine_similarity": cosine,
    }


def boolean_metrics(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, int | float]:
    mismatch_count = sum(
        int(torch.count_nonzero(left != right)) for left, right in pairs
    )
    element_count = sum(left.numel() for left, _ in pairs)
    return {
        "elements": element_count,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_count / element_count,
    }


def grouped_parity(
    role: str,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    bool_groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if "self_key" in name:
            group = "self_key"
        elif "self_value" in name:
            group = "self_value"
        elif "self_mask" in name:
            group = "self_mask"
        elif "cross_key" in name:
            group = "cross_key"
        elif "cross_value" in name:
            group = "cross_value"
        elif "alignment" in name:
            group = "alignment"
        else:
            group = "hidden"
        if group.startswith("self_"):
            valid = PREFILL_LENGTH if role == "prefill" else PREFILL_LENGTH + 1
            actual_value = actual_value[:, :valid]
            expected_value = expected_value[:, :valid]
        if actual_value.dtype == torch.bool:
            bool_groups.setdefault(group, []).append((actual_value, expected_value))
        else:
            groups.setdefault(group, []).append((actual_value, expected_value))
    result = {name: numerical_metrics(values) for name, values in groups.items()}
    result.update(
        {name: boolean_metrics(values) for name, values in bool_groups.items()}
    )
    return result


def write_bf16_validation_tensor(
    directory: Path,
    name: str,
    value: torch.Tensor,
) -> dict:
    if value.dtype != torch.bfloat16:
        raise RuntimeError(f"validation tensor must be BF16: {name}")
    payload = (
        value.detach()
        .cpu()
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
        .tobytes()
    )
    relative = Path("validation") / name
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative.as_posix(),
        "dtype": "bf16",
        "shape": list(value.shape),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def measure_plan_parity(
    tensorrt,
    staging: Path,
    prefill_plan: Path,
    step_plan: Path,
    prefill_inputs: tuple[torch.Tensor, torch.Tensor],
    step_inputs: tuple[torch.Tensor, ...],
    prefill_expected: dict[str, torch.Tensor],
    step_expected: dict[str, torch.Tensor],
) -> dict:
    prefill_input_map = {
        "condition": prefill_inputs[0],
        "condition_mask": prefill_inputs[1],
    }
    prefill_actual = execute_plan(tensorrt, prefill_plan, prefill_input_map)

    # The measured step is deliberately closed-loop: it consumes the K/V
    # produced by TensorRT prefill, not the oracle K/V. This exposes drift
    # across the real engine boundary.
    step_input_map = dict(zip(step_input_names(), step_inputs, strict=True))
    for layer_index in range(DECODER_LAYERS):
        for stem in ("self_key", "self_value", "self_mask", "cross_key", "cross_value"):
            step_input_map[f"step_{stem}_in_{layer_index}"] = prefill_actual[
                f"prefill_{stem}_{layer_index}"
            ]
    step_actual = execute_plan(tensorrt, step_plan, step_input_map)
    validation_tensors = [
        write_bf16_validation_tensor(
            staging,
            "main-decoder-prefill.hidden.bf16.bin",
            prefill_actual["last_hidden"],
        ),
        write_bf16_validation_tensor(
            staging,
            "main-decoder-step-001.hidden.bf16.bin",
            step_actual["decoder_hidden"],
        ),
    ]
    return {
        "status": "measured-not-accepted",
        "reason": (
            "per-output tolerances require multi-input, closed-loop Local AR, "
            "EOS, and sequence-level evidence"
        ),
        "step_cache_source": "TensorRT prefill outputs",
        "prefill": grouped_parity(
            "prefill", prefill_actual, prefill_expected
        ),
        "step": grouped_parity("step", step_actual, step_expected),
        "validation_tensors": validation_tensors,
    }


def publish_directory_no_replace(staging: Path, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for atomic no-replace publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(output),
        1,
    )
    if result == 0:
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
    lock, lock_sha256 = verify_locked_inputs(args)
    fixture = load_fixture(args.fixture, args.lock)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        model = load_model(args.model, args.speech_root)
        prefill = MainDecoderPrefillWrapper(model).eval()
        step = MainDecoderStepWrapper(model).eval()
        (
            prefill_inputs,
            step_inputs,
            prefill_expected,
            step_expected,
        ) = fixture_inputs(fixture)
        require_pytorch_fixture_parity(
            prefill,
            step,
            prefill_inputs,
            step_inputs,
            prefill_expected,
            step_expected,
        )

        prefill_onnx = staging / PREFILL_ONNX
        step_onnx = staging / STEP_ONNX
        prefill_onnx_metadata = export_graph(
            wrapper=prefill,
            example_inputs=prefill_inputs,
            input_names=["condition", "condition_mask"],
            output_names=prefill_output_names(),
            dynamic_axes=prefill_dynamic_axes(),
            output_path=prefill_onnx,
        )
        step_onnx_metadata = export_graph(
            wrapper=step,
            example_inputs=step_inputs,
            input_names=step_input_names(),
            output_names=step_output_names(),
            dynamic_axes=step_dynamic_axes(),
            output_path=step_onnx,
        )

        from build_text_encoder_plan import register_plugin

        tensorrt = import_tensorrt(args.tensorrt_python_path)
        plugin_library, plugin_metadata = register_plugin(args.plugin)

        prefill_plan = staging / PREFILL_PLAN
        step_plan = staging / STEP_PLAN
        prefill_report = staging / PREFILL_BUILD_REPORT
        step_report = staging / STEP_BUILD_REPORT
        prefill_build = build_plan(
            tensorrt=tensorrt,
            role="prefill",
            onnx_path=prefill_onnx,
            plan_path=prefill_plan,
            report_path=prefill_report,
        )
        step_build = build_plan(
            tensorrt=tensorrt,
            role="step",
            onnx_path=step_onnx,
            plan_path=step_plan,
            report_path=step_report,
        )

        prefill_inspection = inspect_plan(tensorrt, "prefill", prefill_plan)
        step_inspection = inspect_plan(tensorrt, "step", step_plan)
        plan_parity = measure_plan_parity(
            tensorrt,
            staging,
            prefill_plan,
            step_plan,
            prefill_inputs,
            step_inputs,
            prefill_expected,
            step_expected,
        )
        del model

        artifact_paths = [
            prefill_onnx,
            step_onnx,
            prefill_plan,
            step_plan,
            prefill_report,
            step_report,
        ]
        receipt = {
            "schema_version": 1,
            "artifact_role": "main_decoder_prefill_and_step",
            "status": "measured-not-accepted",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "source": {
                "exporter_sha256": sha256_file(Path(__file__).resolve(strict=True)),
                "wrapper_sha256": sha256_file(
                    (Path(__file__).parent / "main_decoder_wrapper.py").resolve(
                        strict=True
                    )
                ),
                "oracle_lock_sha256": lock_sha256,
                "oracle_source_revision": lock["oracle_source"]["base_revision"],
                "oracle_source_bundle_sha256": lock["oracle_source"][
                    "optimized_source_bundle_sha256"
                ],
                "model_sha256": lock["model"]["sha256"],
                "codec_model_sha256": lock["codec"]["sha256"],
                "acceptance_receipt_sha256": lock["acceptance"][
                    "receipt_sha256"
                ],
                "boundary_fixture_manifest_sha256": fixture.manifest_sha256,
                "plugin_sha256": sha256_file(
                    args.plugin.resolve(strict=True)
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
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "export": {
                "frontend": "torch.onnx legacy TorchScript exporter",
                "opset": ONNX_OPSET,
                "constant_folding": False,
                "external_data": False,
                "pytorch_fixture_parity": "bit-exact-all-declared-outputs",
                "pointwise_ffn_projection": "kernel1-bf16-matmul-v1",
                "cross_attention_qk_softmax": (
                    "deterministic-k128-plugin-mode-1"
                ),
                "one_step_self_attention_qk": (
                    "cublas-bf16-compute32f-plugin-mode-5"
                ),
                "one_step_self_attention_context": (
                    "position-derived-active-cache-cublas-bf16-"
                    "compute32f-plugin-mode-6"
                ),
                "prefill_onnx": prefill_onnx_metadata,
                "step_onnx": step_onnx_metadata,
            },
            "build": {
                "network_flags": ["strongly_typed"],
                "tf32": False,
                "plugin": plugin_metadata,
                "profile": {
                    "name": "text_1_512",
                    "T": {
                        "min": MIN_TEXT_TOKENS,
                        "opt": OPT_TEXT_TOKENS,
                        "max": MAX_TEXT_TOKENS,
                    },
                },
                "prefill": prefill_build,
                "step": step_build,
            },
            "plan_inspection": {
                "prefill": prefill_inspection,
                "step": step_inspection,
            },
            "plan_parity": plan_parity,
            "artifacts": [
                artifact_record(path, staging) for path in artifact_paths
            ]
            + plan_parity["validation_tensors"],
        }
        receipt_path = staging / RECEIPT
        if plugin_library is None:
            raise AssertionError("plugin library ownership was lost")
        receipt_payload = canonical_json_bytes(receipt)
        receipt_path.write_bytes(receipt_payload)
        (staging / RECEIPT_CHECKSUM).write_text(
            f"{hashlib.sha256(receipt_payload).hexdigest()}  {RECEIPT}\n",
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

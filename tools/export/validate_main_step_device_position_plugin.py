#!/usr/bin/env python3
"""Accept the fixed-I/O, device-position Main self-attention component.

This component gate proves that TensorRT keeps ``position`` as DEVICE
execution data, exercises every supported cache length, verifies captured
CUDA Graph replay, and authenticates the runtime-discovered 7-QK/14-PV cuBLAS
kernel-class table. Full-decoder recurrence and closed-loop sequence gates
remain mandatory after this component passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

import build_text_encoder_plan
from cublas_runtime_identity import collect_cublas_runtime_identity
from cuda_runtime_identity import collect_cuda_runtime_identity
from mode8_class_table_identity import collect_mode8_class_table_identity


BATCH = 2
HEADS = 12
HEAD_WIDTH = 64
CACHE_CAPACITY = 467
FIRST_POSITION = 218
LAST_POSITION = 466
SCALE = 0.125
INVALID_K_LAYER_ZERO_STATUS = 1 << 28
SEEDS = (2026080601, 2026080602, 2026080603)
MASK_CASES = ("all_valid", "staggered_holes", "batch_split_holes")

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(tensorrt: ModuleType) -> bytes:
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(logger)
    network = builder.create_network(
        1 << int(tensorrt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    query = network.add_input(
        "query", tensorrt.bfloat16, (BATCH, HEADS, 1, HEAD_WIDTH)
    )
    key = network.add_input(
        "key",
        tensorrt.bfloat16,
        (BATCH, HEADS, CACHE_CAPACITY, HEAD_WIDTH),
    )
    value = network.add_input(
        "value",
        tensorrt.bfloat16,
        (BATCH, HEADS, CACHE_CAPACITY, HEAD_WIDTH),
    )
    key_mask = network.add_input(
        "key_mask", tensorrt.bool, (BATCH, CACHE_CAPACITY)
    )
    position = network.add_input("position", tensorrt.int64, ())
    shape_reference = network.add_input(
        "shape_reference",
        tensorrt.bfloat16,
        (BATCH, HEADS, 1, HEAD_WIDTH),
    )
    execution_status = network.add_input(
        "execution_status_in", tensorrt.int32, ()
    )
    if any(
        tensor is None
        for tensor in (
            query,
            key,
            value,
            key_mask,
            position,
            shape_reference,
            execution_status,
        )
    ):
        raise RuntimeError("TensorRT mode-8 input creation failed")

    creator = tensorrt.get_plugin_registry().get_creator(
        "MagpieSoftmax", "1", "magpie_tts_rt"
    )
    if creator is None:
        raise RuntimeError("MagpieSoftmax creator is unavailable")
    mode = np.asarray([8], dtype=np.int32)
    layer_index = np.asarray([0], dtype=np.int32)
    fields = tensorrt.PluginFieldCollection(
        [
            tensorrt.PluginField(
                "mode", mode, tensorrt.PluginFieldType.INT32
            ),
            tensorrt.PluginField(
                "layer_index", layer_index, tensorrt.PluginFieldType.INT32
            ),
        ]
    )
    plugin = creator.create_plugin(
        "main_step_device_position_mode8",
        fields,
        tensorrt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("MagpieSoftmax mode-8 creation failed")
    layer = network.add_plugin_v3(
        [
            query,
            key,
            value,
            key_mask,
            position,
            shape_reference,
            execution_status,
        ],
        [],
        plugin,
    )
    if layer is None:
        raise RuntimeError("TensorRT mode-8 plugin layer creation failed")
    output = layer.get_output(0)
    status_output = layer.get_output(1)
    output.name = "context"
    status_output.name = "execution_status_out"
    network.mark_output(output)
    network.mark_output(status_output)

    config = builder.create_builder_config()
    config.clear_flag(tensorrt.BuilderFlag.TF32)
    config.builder_optimization_level = 5
    config.profiling_verbosity = tensorrt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(
        tensorrt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the mode-8 plan")
    return bytes(serialized)


def bind_addresses(
    context,
    tensors: dict[str, torch.Tensor],
) -> None:
    for name, tensor in tensors.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"TensorRT rejected {name} address")
    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"mode-8 plan has unresolved tensors: {unresolved}")


def enqueue(context, stream: torch.cuda.Stream) -> None:
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT mode-8 enqueue failed")


def reference_context(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_mask: torch.Tensor,
    position: int,
) -> torch.Tensor:
    active_length = position + 1
    scores = torch.matmul(
        query, key[..., :active_length, :].transpose(-1, -2).contiguous()
    )
    scores = scores * SCALE
    valid = key_mask[:, None, None, :active_length]
    scores = scores.masked_fill(~valid, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = probabilities.masked_fill(~valid, 0.0)
    return torch.matmul(probabilities, value[..., :active_length, :])


def compare_all_positions(
    context,
    tensors: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> dict[str, JsonValue]:
    total_elements = 0
    mismatch_count = 0
    positions_with_mismatch: list[int] = []
    worst_abs = 0.0
    first_mismatch: dict[str, JsonValue] | None = None
    status_mismatch_count = 0
    for seed in SEEDS:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        tensors["query"].copy_(
            torch.randn(
                tensors["query"].shape,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
        )
        tensors["key"].copy_(
            torch.randn(
                tensors["key"].shape,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
        )
        tensors["value"].copy_(
            torch.randn(
                tensors["value"].shape,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
        )
        for mask_case in MASK_CASES:
            tensors["key_mask"].fill_(True)
            if mask_case == "staggered_holes":
                tensors["key_mask"][0, 7::17] = False
                tensors["key_mask"][1, 11::19] = False
            elif mask_case == "batch_split_holes":
                tensors["key_mask"][0, 32::23] = False
                tensors["key_mask"][1, 48::29] = False
            for position in range(FIRST_POSITION, LAST_POSITION + 1):
                tensors["position"].fill_(position)
                enqueue(context, stream)
                expected = reference_context(
                    tensors["query"],
                    tensors["key"],
                    tensors["value"],
                    tensors["key_mask"],
                    position,
                )
                stream.synchronize()
                if int(tensors["execution_status_out"].item()) != 0:
                    status_mismatch_count += 1
                mismatch = (
                    tensors["context"].view(torch.int16)
                    != expected.view(torch.int16)
                )
                current_count = int(torch.count_nonzero(mismatch))
                current_abs = float(
                    (tensors["context"].float() - expected.float())
                    .abs()
                    .max()
                )
                total_elements += expected.numel()
                mismatch_count += current_count
                worst_abs = max(worst_abs, current_abs)
                if current_count:
                    positions_with_mismatch.append(position)
                    if first_mismatch is None:
                        first_index = int(torch.nonzero(mismatch.flatten())[0])
                        first_mismatch = {
                            "seed": seed,
                            "mask_case": mask_case,
                            "position": position,
                            "flat_index": first_index,
                            "actual": float(
                                tensors["context"].flatten()[first_index]
                            ),
                            "reference": float(expected.flatten()[first_index]),
                        }
    unique_positions = sorted(set(positions_with_mismatch))
    return {
        "seeds": list(SEEDS),
        "mask_cases": list(MASK_CASES),
        "position_range": [FIRST_POSITION, LAST_POSITION],
        "case_count": (
            len(SEEDS)
            * len(MASK_CASES)
            * (LAST_POSITION - FIRST_POSITION + 1)
        ),
        "total_elements": total_elements,
        "bit_mismatch_count": mismatch_count,
        "bit_mismatch_ratio": mismatch_count / total_elements,
        "positions_with_mismatch": unique_positions,
        "position_mismatch_count": len(unique_positions),
        "worst_max_abs": worst_abs,
        "first_mismatch": first_mismatch,
        "execution_status_mismatch_count": status_mismatch_count,
        "bit_exact": mismatch_count == 0 and status_mismatch_count == 0,
    }


def validate_graph_replay(
    context,
    tensors: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> dict[str, JsonValue]:
    tensors["position"].fill_(FIRST_POSITION)
    tensors["execution_status_in"].zero_()
    enqueue(context, stream)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        enqueue(context, stream)
    cases: list[JsonValue] = []
    for position in (FIRST_POSITION, 256, 300, LAST_POSITION):
        tensors["position"].fill_(position)
        graph.replay()
        expected = reference_context(
            tensors["query"],
            tensors["key"],
            tensors["value"],
            tensors["key_mask"],
            position,
        )
        stream.synchronize()
        mismatch_count = int(
            torch.count_nonzero(
                tensors["context"].view(torch.int16)
                != expected.view(torch.int16)
            )
        )
        cases.append(
            {
                "position": position,
                "bit_mismatch_count": mismatch_count,
                "execution_status": int(
                    tensors["execution_status_out"].item()
                ),
                "max_abs": float(
                    (tensors["context"].float() - expected.float())
                    .abs()
                    .max()
                ),
            }
        )
    return {
        "single_capture": True,
        "device_position_changed_between_replays": True,
        "cases": cases,
        "replay_completed": True,
    }


def validate_execution_status(
    context,
    tensors: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> dict[str, JsonValue]:
    invalid_cases: list[JsonValue] = []
    for position in (FIRST_POSITION - 1, LAST_POSITION + 1):
        tensors["position"].fill_(position)
        tensors["execution_status_in"].zero_()
        enqueue(context, stream)
        stream.synchronize()
        status = int(tensors["execution_status_out"].item())
        if status != INVALID_K_LAYER_ZERO_STATUS:
            raise RuntimeError(
                "mode-8 returned the wrong invalid-K status for position "
                f"{position}: expected={INVALID_K_LAYER_ZERO_STATUS} "
                f"actual={status}"
            )
        invalid_cases.append({"position": position, "status": status})

    first_case = invalid_cases[0]
    assert isinstance(first_case, dict)
    sticky_status_value = first_case["status"]
    assert isinstance(sticky_status_value, int)
    tensors["position"].fill_(FIRST_POSITION)
    tensors["execution_status_in"].fill_(sticky_status_value)
    enqueue(context, stream)
    stream.synchronize()
    sticky_output = int(tensors["execution_status_out"].item())
    if sticky_output != sticky_status_value:
        raise RuntimeError(
            "mode-8 did not preserve the first nonzero execution status"
        )
    tensors["execution_status_in"].zero_()
    tensors["position"].fill_(FIRST_POSITION)
    return {
        "layer_index": 0,
        "invalid_cases": invalid_cases,
        "sticky_input_status": sticky_status_value,
        "sticky_output_status": sticky_output,
        "sticky_first_error_preserved": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--tensorrt-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if not args.plugin.is_file():
        raise FileNotFoundError(args.plugin)
    args.output.mkdir(parents=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(args.tensorrt_python_path))
    plugin_library, _ = build_text_encoder_plan.register_plugin(args.plugin)
    import tensorrt

    plan = build_plan(tensorrt)
    plan_path = args.output / "main-step-device-position-mode8.plan"
    plan_path.write_bytes(plan)
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("mode-8 plan deserialization failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("mode-8 context creation failed")
    if engine.get_tensor_location("position") != tensorrt.TensorLocation.DEVICE:
        raise RuntimeError("position is not a DEVICE tensor")
    if engine.is_shape_inference_io("position"):
        raise RuntimeError("position unexpectedly participates in shape inference")
    for name, mode in (
        ("execution_status_in", tensorrt.TensorIOMode.INPUT),
        ("execution_status_out", tensorrt.TensorIOMode.OUTPUT),
    ):
        if (
            engine.get_tensor_mode(name) != mode
            or engine.get_tensor_dtype(name) != tensorrt.int32
            or tuple(engine.get_tensor_shape(name)) != ()
            or engine.get_tensor_location(name)
            != tensorrt.TensorLocation.DEVICE
            or engine.is_shape_inference_io(name)
        ):
            raise RuntimeError(f"{name} is not canonical DEVICE INT32 scalar I/O")

    tensors = {
        "query": torch.empty(
            (BATCH, HEADS, 1, HEAD_WIDTH),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        "key": torch.empty(
            (BATCH, HEADS, CACHE_CAPACITY, HEAD_WIDTH),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        "value": torch.empty(
            (BATCH, HEADS, CACHE_CAPACITY, HEAD_WIDTH),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        "key_mask": torch.empty(
            (BATCH, CACHE_CAPACITY), dtype=torch.bool, device="cuda"
        ),
        "position": torch.empty((), dtype=torch.int64, device="cuda"),
        "execution_status_in": torch.zeros(
            (), dtype=torch.int32, device="cuda"
        ),
        "shape_reference": torch.zeros(
            (BATCH, HEADS, 1, HEAD_WIDTH),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        "context": torch.empty(
            (BATCH, HEADS, 1, HEAD_WIDTH),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        "execution_status_out": torch.empty(
            (), dtype=torch.int32, device="cuda"
        ),
    }
    bind_addresses(context, tensors)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        component = compare_all_positions(context, tensors, stream)
        graph = validate_graph_replay(context, tensors, stream)
        execution_status = validate_execution_status(
            context, tensors, stream
        )

    if plugin_library is None:
        raise RuntimeError("Plugin library lifetime was not retained")
    graph_cases = graph.get("cases")
    if (
        component.get("bit_exact") is not True
        or not isinstance(graph_cases, list)
        or len(graph_cases) != 4
        or any(
            not isinstance(case, dict)
            or case.get("bit_mismatch_count") != 0
            or case.get("max_abs") != 0.0
            or case.get("execution_status") != 0
            for case in graph_cases
        )
    ):
        raise RuntimeError(
            "mode-8 component is not bit-exact for all direct and graph cases"
        )
    class_table = collect_mode8_class_table_identity(plugin_library)
    cuda_identity = collect_cuda_runtime_identity()
    cublas_identity = collect_cublas_runtime_identity(plugin_library)
    receipt: dict[str, JsonValue] = {
        "schema_version": 1,
        "artifact_role": "main_decoder_device_position_mode8_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted",
        "reason": (
            "all 2241 predeclared direct cases and four graph replays are "
            "bit-exact, and the 7-QK/14-PV class table is authenticated"
        ),
        "contract": {
            "fixed_io": True,
            "position_location": "DEVICE",
            "position_is_shape_inference_io": False,
            "first_position": FIRST_POSITION,
            "last_position": LAST_POSITION,
            "execution_status_dtype": "INT32",
            "execution_status_location": "DEVICE",
            "execution_status_is_shape_inference_io": False,
            "execution_status_sticky_first_error": True,
            "component_layer_index": 0,
        },
        "source": {
            "validator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "plugin_sha256": sha256_file(args.plugin),
            "plugin_size_bytes": args.plugin.stat().st_size,
        },
        "class_table": class_table.document,
        "class_table_sha256": class_table.sha256,
        "artifacts": [{
            "path": plan_path.name,
            "sha256": sha256_file(plan_path),
            "size_bytes": plan_path.stat().st_size,
        }],
        "component": component,
        "cuda_graph": graph,
        "execution_status": execution_status,
        "runtime": {
            "cuda": cuda_identity.to_json(),
            "cublas": cublas_identity.to_json(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda_build": str(torch.version.cuda),
            "tensorrt": tensorrt.__version__,
        },
    }
    receipt_path = args.output / "validation-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "validation-receipt.json.sha256").write_text(
        f"{sha256_file(receipt_path)}  validation-receipt.json\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

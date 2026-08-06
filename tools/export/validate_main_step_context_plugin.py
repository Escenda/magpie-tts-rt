#!/usr/bin/env python3
"""Validate Main Decoder mode-6 context for every supported cache length."""

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


BATCH = 2
HEADS = 12
CACHE_CAPACITY = 467
HEAD_WIDTH = 64
MINIMUM_ACTIVE_LENGTH = 219
OPTIMUM_ACTIVE_LENGTH = 343
BENCHMARK_ITERATIONS = 100


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
    probabilities = network.add_input(
        "probabilities",
        tensorrt.bfloat16,
        (BATCH, HEADS, 1, -1),
    )
    value = network.add_input(
        "value",
        tensorrt.bfloat16,
        (BATCH, HEADS, CACHE_CAPACITY, HEAD_WIDTH),
    )
    shape_reference = network.add_input(
        "shape_reference",
        tensorrt.bfloat16,
        (BATCH, HEADS, 1, HEAD_WIDTH),
    )
    if probabilities is None or value is None or shape_reference is None:
        raise RuntimeError("TensorRT mode-6 input creation failed")

    creator = tensorrt.get_plugin_registry().get_creator(
        "MagpieSoftmax", "1", "magpie_tts_rt"
    )
    if creator is None:
        raise RuntimeError("MagpieSoftmax creator is unavailable")
    mode = np.asarray([6], dtype=np.int32)
    fields = tensorrt.PluginFieldCollection(
        [
            tensorrt.PluginField(
                "mode",
                mode,
                tensorrt.PluginFieldType.INT32,
            )
        ]
    )
    plugin = creator.create_plugin(
        "main_step_context_validation",
        fields,
        tensorrt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("MagpieSoftmax mode-6 creation failed")
    layer = network.add_plugin_v3(
        [probabilities, value, shape_reference],
        [],
        plugin,
    )
    if layer is None:
        raise RuntimeError("MagpieSoftmax mode-6 layer creation failed")
    output = layer.get_output(0)
    output.name = "context"
    network.mark_output(output)

    profile = builder.create_optimization_profile()
    expected_profile = (
        (BATCH, HEADS, 1, MINIMUM_ACTIVE_LENGTH),
        (BATCH, HEADS, 1, OPTIMUM_ACTIVE_LENGTH),
        (BATCH, HEADS, 1, CACHE_CAPACITY),
    )
    profile.set_shape("probabilities", *expected_profile)
    actual_profile = tuple(
        tuple(shape) for shape in profile.get_shape("probabilities")
    )
    if actual_profile != expected_profile:
        raise RuntimeError(
            "TensorRT rejected the mode-6 profile: "
            f"expected={expected_profile}, actual={actual_profile}"
        )

    config = builder.create_builder_config()
    config.clear_flag(tensorrt.BuilderFlag.TF32)
    config.builder_optimization_level = 5
    config.profiling_verbosity = tensorrt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(
        tensorrt.MemoryPoolType.WORKSPACE,
        1024 * 1024 * 1024,
    )
    if config.add_optimization_profile(profile) != 0:
        raise RuntimeError("mode-6 profile index is not zero")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the mode-6 plan")
    return bytes(serialized)


def bind_and_enqueue(
    context,
    probabilities: torch.Tensor,
    value: torch.Tensor,
    shape_reference: torch.Tensor,
    output: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    if not context.set_input_shape(
        "probabilities", tuple(probabilities.shape)
    ):
        raise RuntimeError(
            "TensorRT rejected a supported mode-6 shape: "
            f"{tuple(probabilities.shape)}"
        )
    for name, tensor in (
        ("probabilities", probabilities),
        ("value", value),
        ("shape_reference", shape_reference),
        ("context", output),
    ):
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"TensorRT rejected the {name} address")
    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"mode-6 unresolved tensors: {unresolved}")
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT mode-6 enqueue failed")


def validate_rejected_shapes(engine) -> list[int]:
    rejected: list[int] = []
    for active_length in (
        MINIMUM_ACTIVE_LENGTH - 1,
        CACHE_CAPACITY + 1,
    ):
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("mode-6 rejection context creation failed")
        accepted = context.set_input_shape(
            "probabilities",
            (BATCH, HEADS, 1, active_length),
        )
        if accepted:
            raise RuntimeError(
                "TensorRT accepted an out-of-profile mode-6 shape: "
                f"K={active_length}"
            )
        rejected.append(active_length)
    return rejected


def benchmark_lengths(
    context,
    full_probabilities: torch.Tensor,
    value: torch.Tensor,
    shape_reference: torch.Tensor,
    output: torch.Tensor,
    stream: torch.cuda.Stream,
) -> list[dict[str, int | float | str]]:
    measurements: list[dict[str, int | float | str]] = []
    for active_length in (
        MINIMUM_ACTIVE_LENGTH,
        OPTIMUM_ACTIVE_LENGTH,
        CACHE_CAPACITY,
    ):
        probabilities = (
            full_probabilities[..., :active_length].contiguous()
        )
        for _ in range(10):
            bind_and_enqueue(
                context,
                probabilities,
                value,
                shape_reference,
                output,
                stream,
            )
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(BENCHMARK_ITERATIONS):
            bind_and_enqueue(
                context,
                probabilities,
                value,
                shape_reference,
                output,
                stream,
            )
        end.record(stream)
        end.synchronize()
        measurements.append(
            {
                "active_length": active_length,
                "iterations": BENCHMARK_ITERATIONS,
                "mean_latency_ms": (
                    start.elapsed_time(end) / BENCHMARK_ITERATIONS
                ),
                "measurement_scope": (
                    "probability_staging_plus_value_staging_plus_cublas_gemm"
                ),
            }
        )
    return measurements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-python-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite validation output: {args.output}"
        )
    if not args.plugin.is_file():
        raise FileNotFoundError(args.plugin)
    args.output.mkdir(parents=True)

    sys.path.insert(0, str(args.tensorrt_python_path))
    plugin_library, _ = build_text_encoder_plan.register_plugin(
        args.plugin
    )
    # TensorRT is supplied by the target Thor runtime rather than the venv.
    import tensorrt

    plan = build_plan(tensorrt)
    plan_path = args.output / "main-step-context-validation.plan"
    plan_path.write_bytes(plan)
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("mode-6 validation plan deserialize failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("mode-6 validation context creation failed")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260730)
    full_probabilities = torch.randn(
        (BATCH, HEADS, 1, CACHE_CAPACITY),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    token_major_value = torch.randn(
        (BATCH, CACHE_CAPACITY, HEADS, HEAD_WIDTH),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    source_value = token_major_value.transpose(1, 2)
    plan_value = source_value.contiguous()
    shape_reference = torch.zeros(
        (BATCH, HEADS, 1, HEAD_WIDTH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(shape_reference)
    stream = torch.cuda.Stream()

    failures: list[dict[str, int | float]] = []
    with torch.cuda.stream(stream):
        for active_length in range(
            MINIMUM_ACTIVE_LENGTH, CACHE_CAPACITY + 1
        ):
            source_probabilities = full_probabilities[
                ..., :active_length
            ]
            compact_probabilities = source_probabilities.contiguous()
            expected = torch.matmul(
                source_probabilities,
                source_value[..., :active_length, :],
            )
            bind_and_enqueue(
                context,
                compact_probabilities,
                plan_value,
                shape_reference,
                output,
                stream,
            )
            stream.synchronize()
            mismatch = (
                output.view(torch.int16)
                != expected.view(torch.int16)
            )
            mismatch_count = int(torch.count_nonzero(mismatch))
            if mismatch_count:
                failures.append(
                    {
                        "active_length": active_length,
                        "bit_mismatch_count": mismatch_count,
                        "max_abs": float(
                            (output.float() - expected.float())
                            .abs()
                            .max()
                        ),
                    }
                )
        latency = benchmark_lengths(
            context,
            full_probabilities,
            plan_value,
            shape_reference,
            output,
            stream,
        )

    rejected_lengths = validate_rejected_shapes(engine)
    if plugin_library is None:
        raise RuntimeError("plugin library lifetime was not retained")
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if not failures else "rejected",
        "plugin": {
            "path": str(args.plugin.resolve()),
            "sha256": sha256_file(args.plugin),
            "size_bytes": args.plugin.stat().st_size,
        },
        "plan": {
            "path": plan_path.name,
            "sha256": sha256_file(plan_path),
            "size_bytes": plan_path.stat().st_size,
            "profiling_verbosity": "DETAILED",
            "tf32": False,
        },
        "source_contract": {
            "probabilities": (
                "slice_of_2x12x1x467_with_physical_head_stride_467"
            ),
            "value": (
                "slice_of_2x467x12x64_transposed_to_2x12xKx64"
            ),
            "reference_operation": "torch.matmul",
        },
        "validation": {
            "minimum_active_length": MINIMUM_ACTIVE_LENGTH,
            "maximum_active_length": CACHE_CAPACITY,
            "case_count": (
                CACHE_CAPACITY - MINIMUM_ACTIVE_LENGTH + 1
            ),
            "failures": failures,
            "rejected_active_lengths": rejected_lengths,
            "exact_gate": "zero_bf16_bit_mismatch_every_case",
        },
        "latency": latency,
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "tensorrt": tensorrt.__version__,
        },
    }
    receipt_path = args.output / "validation-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "validation-receipt.json.sha256").write_text(
        f"{sha256_file(receipt_path)}  validation-receipt.json\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(
            f"mode-6 exact validation failed: {failures[:8]}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Accept full Main Decoder recurrence against the locked PyTorch oracle."""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import torch

import export_main_decoder as export_main
import main_decoder_wrapper as wrapper
from alignment_controller import SofiaAlignmentController
from build_text_encoder_plan import register_plugin
from validate_main_decoder_plans import (
    JsonValue,
    authenticate_plan_export,
    load_fixture_metadata,
    require_sha256,
    sha256_bytes,
    sha256_file,
)
from validate_main_local_ar_sequence import TensorRTPlanSession


RECEIPT = "recurrence-receipt.json"
RECEIPT_CHECKSUM = "recurrence-receipt.json.sha256"
MINIMUM_DISTINCT_FIXTURES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan-export", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        required=True,
        help="Repeat for three or more locked Japanese fixtures.",
    )
    parser.add_argument("--tensorrt-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_exact_tensor(
    *,
    fixture_id: str,
    stage: str,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> int:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise RuntimeError(
            f"{fixture_id} {stage} {name} contract mismatch: "
            f"actual={actual.dtype}/{tuple(actual.shape)}, "
            f"expected={expected.dtype}/{tuple(expected.shape)}"
        )
    mismatch = actual != expected
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    if mismatch_count != 0:
        flat_index = int(
            torch.nonzero(mismatch.reshape(-1), as_tuple=False)[0].item()
        )
        raise RuntimeError(
            f"{fixture_id} {stage} {name} is not bit-exact: "
            f"mismatches={mismatch_count}/{actual.numel()}, "
            f"first_flat_index={flat_index}, "
            f"actual={actual.reshape(-1)[flat_index].item()}, "
            f"expected={expected.reshape(-1)[flat_index].item()}"
        )
    return actual.numel()


def require_exact_output_set(
    *,
    fixture_id: str,
    stage: str,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> int:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{fixture_id} {stage} output set mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    return sum(
        require_exact_tensor(
            fixture_id=fixture_id,
            stage=stage,
            name=name,
            actual=actual[name],
            expected=expected[name],
        )
        for name in sorted(expected)
    )


def require_exact_alignment_update(
    *,
    fixture_id: str,
    step: int,
    actual_controller: SofiaAlignmentController,
    actual_alignment: torch.Tensor,
    expected_controller: SofiaAlignmentController,
    expected_alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    expected = expected_controller.update(expected_alignment)
    actual = actual_controller.update(actual_alignment)
    require_exact_tensor(
        fixture_id=fixture_id,
        stage=f"step-{step:03d}-alignment-controller",
        name="prior",
        actual=actual.prior,
        expected=expected.prior,
    )
    require_exact_tensor(
        fixture_id=fixture_id,
        stage=f"step-{step:03d}-alignment-controller",
        name="attended",
        actual=actual.attended,
        expected=expected.attended,
    )
    require_exact_tensor(
        fixture_id=fixture_id,
        stage=f"step-{step:03d}-alignment-controller",
        name="counters",
        actual=actual_controller.counters,
        expected=expected_controller.counters,
    )
    return (
        actual.prior,
        expected.prior,
        int(actual.attended.item()),
    )


def run_fixture(
    *,
    fixture_path: Path,
    lock_path: Path,
    prefill_module: wrapper.MainDecoderPrefillWrapper,
    step_module: wrapper.MainDecoderStepWrapper,
    prefill_session: TensorRTPlanSession,
    step_session: TensorRTPlanSession,
) -> dict[str, JsonValue]:
    fixture = export_main.load_fixture(fixture_path, lock_path)
    metadata = load_fixture_metadata(fixture.root)
    prefill_inputs, _, _, _ = export_main.fixture_inputs(fixture)
    expected_codes = export_main.tensor_from_fixture(
        fixture, "generation.codes"
    ).cuda()
    decoder_steps = expected_codes.shape[2] // 2
    if decoder_steps < 1:
        raise RuntimeError(
            f"{metadata.fixture_id} has no complete Main Decoder recurrence"
        )

    with torch.inference_mode():
        expected_prefill = dict(
            zip(
                wrapper.prefill_output_names(),
                prefill_module(*prefill_inputs),
                strict=True,
            )
        )
    actual_prefill = prefill_session.execute(
        {
            "condition": prefill_inputs[0],
            "condition_mask": prefill_inputs[1],
        }
    )
    compared_values = require_exact_output_set(
        fixture_id=metadata.fixture_id,
        stage="prefill",
        actual=actual_prefill,
        expected=expected_prefill,
    )

    expected_controller = SofiaAlignmentController(
        text_length=fixture.text_tokens,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )
    actual_controller = SofiaAlignmentController(
        text_length=fixture.text_tokens,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )
    expected_alignment = expected_prefill["alignment"]
    actual_alignment = actual_prefill["alignment"]
    expected_state = expected_prefill
    actual_state = actual_prefill
    attended_trace: list[JsonValue] = []

    for step_index in range(decoder_steps):
        (
            actual_prior,
            expected_prior,
            attended,
        ) = require_exact_alignment_update(
            fixture_id=metadata.fixture_id,
            step=step_index,
            actual_controller=actual_controller,
            actual_alignment=actual_alignment,
            expected_controller=expected_controller,
            expected_alignment=expected_alignment,
        )
        attended_trace.append(attended)
        previous_codes = expected_codes[
            :, :, step_index * 2 : step_index * 2 + 2
        ].contiguous()
        if previous_codes.shape[2] != 2:
            raise RuntimeError(
                f"{metadata.fixture_id} step {step_index} lacks two codes"
            )
        position = torch.tensor(
            wrapper.PREFILL_LENGTH + step_index,
            dtype=torch.int64,
            device="cuda",
        )
        expected_inputs: list[torch.Tensor] = [
            previous_codes,
            position,
            expected_prior,
            prefill_inputs[1],
        ]
        actual_inputs: dict[str, torch.Tensor] = {
            "previous_codec_tokens": previous_codes,
            "position": position,
            "alignment_prior": actual_prior,
            "condition_mask": prefill_inputs[1],
        }
        for layer in range(wrapper.DECODER_LAYERS):
            expected_prefix = (
                "prefill" if step_index == 0 else "step"
            )
            expected_inputs.extend(
                [
                    expected_state[
                        f"{expected_prefix}_self_key_"
                        f"{'' if step_index == 0 else 'out_'}{layer}"
                    ],
                    expected_state[
                        f"{expected_prefix}_self_value_"
                        f"{'' if step_index == 0 else 'out_'}{layer}"
                    ],
                    expected_state[
                        f"{expected_prefix}_self_mask_"
                        f"{'' if step_index == 0 else 'out_'}{layer}"
                    ],
                    expected_prefill[f"prefill_cross_key_{layer}"],
                    expected_prefill[f"prefill_cross_value_{layer}"],
                ]
            )
            actual_prefix = "prefill" if step_index == 0 else "step"
            actual_suffix = "" if step_index == 0 else "out_"
            actual_inputs[f"step_self_key_in_{layer}"] = actual_state[
                f"{actual_prefix}_self_key_{actual_suffix}{layer}"
            ]
            actual_inputs[f"step_self_value_in_{layer}"] = actual_state[
                f"{actual_prefix}_self_value_{actual_suffix}{layer}"
            ]
            actual_inputs[f"step_self_mask_in_{layer}"] = actual_state[
                f"{actual_prefix}_self_mask_{actual_suffix}{layer}"
            ]
            actual_inputs[f"step_cross_key_in_{layer}"] = actual_prefill[
                f"prefill_cross_key_{layer}"
            ]
            actual_inputs[f"step_cross_value_in_{layer}"] = actual_prefill[
                f"prefill_cross_value_{layer}"
            ]
        with torch.inference_mode():
            expected_outputs = dict(
                zip(
                    wrapper.step_output_names(),
                    step_module(*tuple(expected_inputs)),
                    strict=True,
                )
            )
        actual_outputs = step_session.execute(actual_inputs)
        compared_values += require_exact_output_set(
            fixture_id=metadata.fixture_id,
            stage=f"step-{step_index:03d}",
            actual=actual_outputs,
            expected=expected_outputs,
        )
        expected_alignment = expected_outputs["alignment"]
        actual_alignment = actual_outputs["alignment"]
        expected_state = expected_outputs
        actual_state = actual_outputs

    result: dict[str, JsonValue] = {
        "fixture_id": metadata.fixture_id,
        "fixture_manifest_sha256": fixture.manifest_sha256,
        "text": metadata.text,
        "language": metadata.language,
        "local_ar_seed": metadata.local_ar_seed,
        "text_tokens": fixture.text_tokens,
        "decoder_steps": decoder_steps,
        "first_position": wrapper.PREFILL_LENGTH,
        "last_position": wrapper.PREFILL_LENGTH + decoder_steps - 1,
        "compared_values": compared_values,
        "all_declared_outputs_bit_exact": True,
        "alignment_controller_bit_exact": True,
        "attended_trace": attended_trace,
    }
    del (
        fixture,
        prefill_inputs,
        expected_codes,
        expected_prefill,
        actual_prefill,
        expected_state,
        actual_state,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    lock_path = args.lock.resolve(strict=True)
    fixture_paths = [
        fixture.resolve(strict=True) for fixture in args.fixture
    ]
    if len(fixture_paths) < MINIMUM_DISTINCT_FIXTURES:
        raise RuntimeError(
            f"at least {MINIMUM_DISTINCT_FIXTURES} fixtures are required"
        )
    fixture_inputs_seen: set[tuple[str, int]] = set()
    for fixture_path in fixture_paths:
        metadata = load_fixture_metadata(fixture_path)
        fixture_input = (metadata.text, metadata.local_ar_seed)
        if fixture_input in fixture_inputs_seen:
            raise RuntimeError(
                f"fixture text/seed input is duplicated: {fixture_input!r}"
            )
        fixture_inputs_seen.add(fixture_input)

    plan_export = authenticate_plan_export(args.plan_export)
    lock_sha256 = sha256_file(lock_path)
    if plan_export.oracle_lock_sha256 != lock_sha256:
        raise RuntimeError("plan export oracle lock differs from validator lock")
    plugin_path = args.plugin.resolve(strict=True)
    plugin_sha256 = sha256_file(plugin_path)
    if plugin_sha256 != plan_export.required_plugin_sha256:
        raise RuntimeError(
            "plan export plugin digest mismatch: "
            f"expected={plan_export.required_plugin_sha256}, "
            f"actual={plugin_sha256}"
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
        plugin_library, _ = register_plugin(plugin_path)
        tensorrt = export_main.import_tensorrt(
            args.tensorrt_python_path
        )
        current_fingerprint = (
            tensorrt.__version__,
            str(torch.version.cuda),
            torch.cuda.get_device_name(0),
            tuple(torch.cuda.get_device_capability(0)),
        )
        expected_fingerprint = (
            plan_export.tensorrt_version,
            plan_export.torch_cuda_build,
            plan_export.gpu_name,
            plan_export.gpu_compute_capability,
        )
        if current_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "plan export runtime fingerprint mismatch: "
                f"expected={expected_fingerprint}, "
                f"actual={current_fingerprint}"
            )

        model = export_main.load_model(
            args.model.resolve(strict=True),
            args.speech_root.resolve(strict=True),
        )
        prefill_module = wrapper.MainDecoderPrefillWrapper(model).eval()
        step_module = wrapper.MainDecoderStepWrapper(model).eval()
        prefill_session = TensorRTPlanSession(
            tensorrt=tensorrt,
            plan_path=plan_export.prefill_plan,
        )
        step_session = TensorRTPlanSession(
            tensorrt=tensorrt,
            plan_path=plan_export.step_plan,
        )
        cases = [
            run_fixture(
                fixture_path=fixture_path,
                lock_path=lock_path,
                prefill_module=prefill_module,
                step_module=step_module,
                prefill_session=prefill_session,
                step_session=step_session,
            )
            for fixture_path in fixture_paths
        ]
        receipt: dict[str, JsonValue] = {
            "schema_version": 1,
            "artifact_role": "main_decoder_full_recurrence_validation",
            "status": "accepted",
            "created_at_utc": datetime.datetime.now(
                datetime.UTC
            ).isoformat(),
            "source": {
                "validator_sha256": sha256_file(
                    Path(__file__).resolve(strict=True)
                ),
                "oracle_lock_sha256": lock_sha256,
                "model_sha256": sha256_file(
                    args.model.resolve(strict=True)
                ),
                "plan_export_receipt_sha256": (
                    plan_export.receipt_sha256
                ),
                "prefill_plan_sha256": (
                    plan_export.prefill_plan_sha256
                ),
                "step_plan_sha256": plan_export.step_plan_sha256,
                "plugin_sha256": plugin_sha256,
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
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
                "cudnn_allow_tf32": (
                    torch.backends.cudnn.allow_tf32
                ),
            },
            "fixture_count": len(cases),
            "fixtures": cases,
            "acceptance": {
                "same_prefill_and_step_plan_for_every_fixture": True,
                "all_declared_outputs_bit_exact_at_every_step": True,
                "alignment_controller_bit_exact_at_every_step": True,
            },
        }
        receipt_payload = export_main.canonical_json_bytes(receipt)
        (staging / RECEIPT).write_bytes(receipt_payload)
        (staging / RECEIPT_CHECKSUM).write_text(
            f"{sha256_bytes(receipt_payload)}  {RECEIPT}\n",
            encoding="ascii",
        )
        if plugin_library is None:
            raise RuntimeError("plugin library lifetime was not retained")
        export_main.publish_directory_no_replace(staging, output)
        print(
            json.dumps(
                receipt,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    sys.exit(main())

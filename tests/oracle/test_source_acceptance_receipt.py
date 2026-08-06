from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "tools" / "oracle" / "source_acceptance_receipt.py"
SPEC = importlib.util.spec_from_file_location("source_acceptance_receipt", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def lock_document() -> MODULE.JsonObject:
    return json.loads(
        (PROJECT_ROOT / "reference" / "oracle-lock.json").read_text(
            encoding="utf-8"
        )
    )


def raw_receipt(lock: MODULE.JsonObject) -> MODULE.JsonObject:
    source = lock["oracle_source"]
    model = lock["model"]
    acceptance = lock["acceptance"]
    source_section = {
        "probe_path": "/host/probe.py",
        "probe_sha256": "1" * 64,
        "helper_path": "/host/helper.py",
        "helper_sha256": "2" * 64,
        "optimized_source_file_sha256": deepcopy(source["files"]),
        "optimized_source_bundle_sha256": source[
            "optimized_source_bundle_sha256"
        ],
        "speech_git_head": source["base_revision"],
        "speech_diff_sha256": "3" * 64,
        "speech_optimized_paths_status": [],
        "loaded_magpietts_path": "/host/Speech/magpietts.py",
        "python_executable": "/host/venv/bin/python",
        "model_path": "/host/model.nemo",
        "model_sha256": model["sha256"],
        "model_size_bytes": model["size_bytes"],
        "model_mtime_ns": 1,
        "gstreamer_version": "gst-launch-1.0 version 1.24.2",
    }
    cases = [
        {
            "name": name,
            "speaker_index": acceptance["speaker_index"],
            "seed": acceptance["local_ar_seed"],
            "schedule": [
                acceptance["first_codec_frames"],
                acceptance["steady_codec_frames"],
                True,
            ],
        }
        for name in MODULE.EXPECTED_CASE_NAMES
    ]
    gates = {
        gate: True for gate in MODULE.REQUIRED_PASSED_GATES
    }
    gates["parakeet_gate_pending"] = True
    gates["failures"] = []
    determinism = [
        {
            "case_name": name,
            "local_ar_seed": acceptance["local_ar_seed"],
            "first_codes_sha256": f"{index + 1:x}" * 64,
            "replay_codes_sha256": f"{index + 1:x}" * 64,
            "codes_exact": True,
            "first_pcm_f32le_sha256": f"{index + 5:x}" * 64,
            "replay_pcm_f32le_sha256": f"{index + 5:x}" * 64,
            "pcm_exact": True,
            "first_codec_frame_count": index + 1,
            "replay_codec_frame_count": index + 1,
            "codec_frame_count_exact": True,
            "passed": True,
        }
        for index, name in enumerate(MODULE.EXPECTED_CASE_NAMES)
    ]
    metric = {
        "median": 1.0,
        "p95": 1.0,
        "minimum": 1.0,
        "maximum": 1.0,
    }
    return {
        "schema": MODULE.RAW_SCHEMA,
        "status": MODULE.RAW_ACCEPTED_STATUS,
        "phase": "complete",
        "created_unix_seconds": 1785370000.0,
        "source_start": deepcopy(source_section),
        "source_end": deepcopy(source_section),
        "source_unchanged": True,
        "static_audit": {
            "case_names": list(MODULE.EXPECTED_CASE_NAMES),
            "case_definition_sha256": "4" * 64,
        },
        "runtime_environment": {
            "torch_version": "2.11.0",
            "torch_cuda_build": "13.0",
            "cudnn_version": 92300,
            "cuda_driver_version": "595.78",
            "gpu_name": "NVIDIA Thor",
            "gpu_compute_capability": [11, 0],
            "python_executable": "/host/venv/bin/python",
            "foreign_site_packages": [],
        },
        "cases": cases,
        "determinism": determinism,
        "aggregate": {
            "case_count": 4,
            "raw_ttfa_ms": dict(metric),
            "gapless_start_ms": dict(metric),
            "max_positive_playback_lateness_ms": dict(metric),
            "generation_rtf": dict(metric),
            "total_rtf": dict(metric),
            "maximum_peak_cuda_allocated_bytes": 1,
            "maximum_peak_cuda_reserved_bytes": 1,
        },
        "gates": gates,
        "parakeet_transcription_evaluated": False,
    }


class SourceAcceptanceReceiptTests(unittest.TestCase):
    def test_projects_current_source_and_removes_host_paths(self) -> None:
        lock = lock_document()
        raw = raw_receipt(lock)
        payload = (json.dumps(raw, sort_keys=True) + "\n").encode("utf-8")
        receipt = MODULE.validate_raw_acceptance(raw, payload, lock)

        self.assertEqual(receipt["schema_version"], 2)
        self.assertEqual(
            receipt["oracle_source"]["optimized_source_bundle_sha256"],
            lock["oracle_source"]["optimized_source_bundle_sha256"],
        )
        self.assertEqual(receipt["source_model"]["version"], "v2607")
        self.assertEqual(
            receipt["source_model"]["revision"],
            "5023df68bd3f5b5ce6d666a50979bc501af145cc",
        )
        self.assertEqual(
            receipt["source_model"]["license"],
            lock["model"]["license"],
        )
        serialized = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("/host/", serialized)
        MODULE.validate_public_acceptance(receipt, lock)
        schema = json.loads(
            (
                PROJECT_ROOT
                / "schemas"
                / "source-model-acceptance-receipt.schema.json"
            ).read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(receipt)

    def test_rejects_one_stale_internal_source_hash(self) -> None:
        lock = lock_document()
        raw = raw_receipt(lock)
        relative = next(iter(lock["oracle_source"]["files"]))
        raw["source_start"]["optimized_source_file_sha256"][relative] = "f" * 64
        raw["source_end"] = deepcopy(raw["source_start"])
        payload = json.dumps(raw, sort_keys=True).encode("utf-8")

        with self.assertRaisesRegex(
            MODULE.SourceAcceptanceError,
            "differs from oracle lock",
        ):
            MODULE.validate_raw_acceptance(raw, payload, lock)

    def test_rejects_public_receipt_absolute_path(self) -> None:
        lock = lock_document()
        raw = raw_receipt(lock)
        payload = json.dumps(raw, sort_keys=True).encode("utf-8")
        receipt = MODULE.validate_raw_acceptance(raw, payload, lock)
        receipt["runtime_environment"]["gpu_name"] = "/host/gpu"

        with self.assertRaisesRegex(
            MODULE.SourceAcceptanceError,
            "absolute path",
        ):
            MODULE.validate_public_acceptance(receipt, lock)

    def test_rejects_mutated_model_revision_or_license(self) -> None:
        lock = lock_document()
        raw = raw_receipt(lock)
        payload = json.dumps(raw, sort_keys=True).encode("utf-8")
        receipt = MODULE.validate_raw_acceptance(raw, payload, lock)

        for field, replacement in (
            ("revision", "f" * 40),
            ("version", "v2607-mutated"),
        ):
            candidate = deepcopy(receipt)
            candidate["source_model"][field] = replacement
            with self.assertRaisesRegex(
                MODULE.SourceAcceptanceError,
                "differs from oracle lock",
            ):
                MODULE.validate_public_acceptance(candidate, lock)

        candidate = deepcopy(receipt)
        candidate["source_model"]["license"]["document_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            MODULE.SourceAcceptanceError,
            "differs from oracle lock",
        ):
            MODULE.validate_public_acceptance(candidate, lock)

    def test_rejects_non_exact_same_seed_replay(self) -> None:
        lock = lock_document()
        raw = raw_receipt(lock)
        raw["determinism"][0]["replay_codes_sha256"] = "f" * 64
        payload = json.dumps(raw, sort_keys=True).encode("utf-8")

        with self.assertRaisesRegex(
            MODULE.SourceAcceptanceError,
            "same-seed replay is not exact",
        ):
            MODULE.validate_raw_acceptance(raw, payload, lock)


if __name__ == "__main__":
    unittest.main()

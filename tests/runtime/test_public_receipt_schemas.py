from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCHEMA_PATH = (
    ROOT / "schemas" / "runtime-benchmark-receipt.schema.json"
)
PROFILE_SCHEMA_PATH = (
    ROOT / "schemas" / "thor-profile-evidence.schema.json"
)
SHA256 = "1" * 64
UNSAFE_PUBLIC_NAMES = (
    "/home/x",
    r"C:\runner\artifact.bin",
    "file://runner/artifact.bin",
)


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class PublicReceiptSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark_schema = load_schema(BENCHMARK_SCHEMA_PATH)
        cls.profile_schema = load_schema(PROFILE_SCHEMA_PATH)
        Draft202012Validator.check_schema(cls.benchmark_schema)
        Draft202012Validator.check_schema(cls.profile_schema)
        cls.benchmark_inputs_validator = Draft202012Validator(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$defs": cls.benchmark_schema["$defs"],
                "$ref": "#/$defs/inputs",
            }
        )
        cls.profile_validator = Draft202012Validator(cls.profile_schema)

    @staticmethod
    def benchmark_inputs() -> dict:
        return {
            "library_name": "libmagpie_tts_rt.so.0.1.0",
            "library_sha256": SHA256,
            "library_size_bytes": 1,
            "bundle_name": "sofia-thor-runtime",
            "bundle_manifest_name": "runtime-bundle-manifest.json",
            "bundle_manifest_sha256": SHA256,
            "bundle_manifest_size_bytes": 1,
            "corpus_name": "thor-ja-v1.jsonl",
            "corpus_sha256": SHA256,
            "corpus_size_bytes": 1,
            "corpus_id": "thor-ja-v1",
            "cuda_device_index": 0,
            "cuda_runtime_library_name": "libcudart.so.13.2.75",
            "cuda_runtime_library_sha256": SHA256,
            "cuda_runtime_library_size_bytes": 1,
            "cuda_driver_version": 13020,
            "cuda_runtime_version": 13020,
            "nvml_library_name": "libnvidia-ml.so.1",
            "nvml_library_sha256": SHA256,
            "nvml_library_size_bytes": 1,
            "nvml_device_index": 0,
            "nvml_driver_version": "595.78",
            "nvml_device_name": "NVIDIA Thor",
            "nvml_device_uuid": "GPU-test",
            "nvml_compute_capability_major": 11,
            "nvml_compute_capability_minor": 0,
            "request_timeout_ms": 120000,
            "normal_case_count": 108,
            "long_run_iterations": 1000,
        }

    @staticmethod
    def profile_evidence() -> dict:
        def artifact(file_name: str) -> dict:
            return {
                "file_name": file_name,
                "sha256": SHA256,
                "size_bytes": 1,
            }

        return {
            "schema_version": "magpie-tts-rt.thor-profile-evidence.v1",
            "created_at_unix_ms": 1,
            "artifacts": {
                "benchmark_receipt": artifact(
                    "thor-benchmark-receipt.json"
                ),
                "nsys_report": artifact("thor-benchmark.nsys-rep"),
                "tegrastats_log": artifact(
                    "thor-benchmark.tegrastats.log"
                ),
            },
            "measurement_scope": {
                "nsys_report": "benchmark target process tree",
                "tegrastats_log": "device-wide AGX Thor telemetry",
                "physical_audio": "not measured",
            },
        }

    def test_benchmark_inputs_accept_only_public_artifact_names(
        self,
    ) -> None:
        inputs = self.benchmark_inputs()
        self.benchmark_inputs_validator.validate(inputs)
        fields = (
            "library_name",
            "bundle_name",
            "corpus_name",
            "cuda_runtime_library_name",
            "nvml_library_name",
        )
        for field in fields:
            for unsafe in UNSAFE_PUBLIC_NAMES:
                candidate = copy.deepcopy(inputs)
                candidate[field] = unsafe
                self.assertFalse(
                    self.benchmark_inputs_validator.is_valid(candidate),
                    f"{field} accepted local path {unsafe!r}",
                )

    def test_profile_evidence_uses_names_not_paths(self) -> None:
        evidence = self.profile_evidence()
        self.profile_validator.validate(evidence)
        for artifact_name in evidence["artifacts"]:
            for unsafe in UNSAFE_PUBLIC_NAMES:
                candidate = copy.deepcopy(evidence)
                candidate["artifacts"][artifact_name]["file_name"] = unsafe
                self.assertFalse(
                    self.profile_validator.is_valid(candidate),
                    f"{artifact_name} accepted local path {unsafe!r}",
                )
            candidate = copy.deepcopy(evidence)
            artifact = candidate["artifacts"][artifact_name]
            artifact["path"] = artifact.pop("file_name")
            self.assertFalse(
                self.profile_validator.is_valid(candidate),
                f"{artifact_name} retained the absolute-path contract",
            )

    def test_long_run_failure_cannot_publish_exception_text(self) -> None:
        validator = Draft202012Validator(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$defs": self.benchmark_schema["$defs"],
                "$ref": "#/$defs/long_run_failure",
            }
        )
        failure = {
            "iteration": 17,
            "case_id": "case-17",
            "diagnostic": "request_exception",
            "diagnostic_sha256": SHA256,
        }
        validator.validate(failure)
        for unsafe in UNSAFE_PUBLIC_NAMES:
            candidate = copy.deepcopy(failure)
            candidate["message"] = unsafe
            self.assertFalse(
                validator.is_valid(candidate),
                f"failure receipt accepted exception text {unsafe!r}",
            )


if __name__ == "__main__":
    unittest.main()

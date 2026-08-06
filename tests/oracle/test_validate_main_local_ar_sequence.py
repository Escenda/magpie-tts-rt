from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPORT_TOOLS = Path(__file__).parents[2] / "tools" / "export"
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))
SCRIPT = EXPORT_TOOLS / "validate_main_local_ar_sequence.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_main_local_ar_sequence",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def create_export(root: Path) -> None:
    artifact_payloads = {
        "local-ar.onnx": b"onnx",
        MODULE.LOCAL_AR_PLAN: b"plan",
        MODULE.LOCAL_AR_PLUGIN: b"plugin",
    }
    artifacts = []
    for relative, payload in artifact_payloads.items():
        (root / relative).write_bytes(payload)
        artifacts.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": sha256(payload),
            }
        )
    receipt = {
        "schema_version": 1,
        "artifact_role": "local_ar_fixed_16",
        "status": "measured-not-accepted",
        "source": {
            "oracle_lock_sha256": "1" * 64,
            "boundary_fixture_manifest_sha256s": [
                "2" * 64,
                "3" * 64,
                "4" * 64,
            ],
        },
        "runtime": {
            "tensorrt": "10.16.2.10",
            "torch_cuda_build": "13.0",
            "gpu_name": "NVIDIA Thor",
            "gpu_compute_capability": [11, 0],
        },
        "artifacts": artifacts,
    }
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    (root / MODULE.LOCAL_AR_RECEIPT).write_bytes(payload)
    (root / MODULE.LOCAL_AR_RECEIPT_CHECKSUM).write_text(
        f"{sha256(payload)}  {MODULE.LOCAL_AR_RECEIPT}\n",
        encoding="ascii",
    )


def create_text_encoder_export(root: Path) -> None:
    artifact_payloads = {
        MODULE.TEXT_ENCODER_PLAN: b"text-plan",
        "trtexec-build.log": b"log",
    }
    artifacts = []
    for relative, payload in artifact_payloads.items():
        (root / relative).write_bytes(payload)
        artifacts.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": sha256(payload),
            }
        )
    receipt = {
        "schema_version": 1,
        "artifact_role": "text_encoder_plan",
        "status": "measured-not-accepted",
        "source": {
            "oracle_lock_sha256": "1" * 64,
            "boundary_fixture_manifest_sha256": "2" * 64,
            "plugin_sha256": sha256(b"plugin"),
        },
        "runtime": {
            "tensorrt": "10.16.2.10",
            "torch_cuda_build": "13.0",
            "gpu_name": "NVIDIA Thor",
            "gpu_compute_capability": [11, 0],
        },
        "artifacts": artifacts,
    }
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    (root / MODULE.TEXT_ENCODER_RECEIPT).write_bytes(payload)
    (root / MODULE.TEXT_ENCODER_RECEIPT_CHECKSUM).write_text(
        f"{sha256(payload)}  {MODULE.TEXT_ENCODER_RECEIPT}\n",
        encoding="ascii",
    )


class TextEncoderExportAuthenticationTests(unittest.TestCase):
    def test_authenticates_exact_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_text_encoder_export(root)

            result = MODULE.authenticate_text_encoder_export(root)

            self.assertEqual(result.oracle_lock_sha256, "1" * 64)
            self.assertEqual(
                result.source_fixture_manifest_sha256,
                "2" * 64,
            )
            self.assertEqual(result.required_plugin_sha256, sha256(b"plugin"))
            self.assertEqual(result.plan.read_bytes(), b"text-plan")

    def test_rejects_unlisted_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_text_encoder_export(root)
            (root / "unexpected.bin").write_bytes(b"unexpected")

            with self.assertRaisesRegex(RuntimeError, "entry set mismatch"):
                MODULE.authenticate_text_encoder_export(root)


class LocalARExportAuthenticationTests(unittest.TestCase):
    def test_authenticates_exact_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_export(root)

            result = MODULE.authenticate_local_ar_export(root)

            self.assertEqual(result.oracle_lock_sha256, "1" * 64)
            self.assertEqual(
                result.source_fixture_manifest_sha256s,
                ("2" * 64, "3" * 64, "4" * 64),
            )
            self.assertEqual(result.plan.read_bytes(), b"plan")
            self.assertEqual(result.plugin.read_bytes(), b"plugin")

    def test_rejects_unlisted_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_export(root)
            (root / "unexpected.bin").write_bytes(b"unexpected")

            with self.assertRaisesRegex(RuntimeError, "entry set mismatch"):
                MODULE.authenticate_local_ar_export(root)

    def test_rejects_symbolic_link_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_export(root)
            plugin = root / MODULE.LOCAL_AR_PLUGIN
            target = root / "plugin-target"
            plugin.rename(target)
            plugin.symlink_to(target.name)

            with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                MODULE.authenticate_local_ar_export(root)

    def test_rejects_receipt_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_export(root)
            (root / MODULE.LOCAL_AR_RECEIPT_CHECKSUM).write_text(
                f"{'0' * 64}  {MODULE.LOCAL_AR_RECEIPT}\n",
                encoding="ascii",
            )

            with self.assertRaisesRegex(RuntimeError, "checksum"):
                MODULE.authenticate_local_ar_export(root)


if __name__ == "__main__":
    unittest.main()

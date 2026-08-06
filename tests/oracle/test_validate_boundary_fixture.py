from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "tools"
    / "oracle"
    / "validate_boundary_fixture.py"
)
SPEC = importlib.util.spec_from_file_location("validate_boundary_fixture", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_manifest(root: Path, manifest: dict) -> None:
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )
    (root / "manifest.json").write_bytes(payload)
    (root / "manifest.json.sha256").write_text(
        f"{digest(payload)}  manifest.json\n",
        encoding="ascii",
    )


def create_valid_fixture(root: Path) -> tuple[dict, Path]:
    tensor_payload = b"\x01\x00\x00\x00\x02\x00\x00\x00"
    tensor_path = root / "tensors" / "sample.bin"
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(tensor_payload)
    lock_path = root.parent / "oracle-lock.json"
    lock_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "oracle_lock_sha256": MODULE.sha256_file(lock_path),
        "local_ar_seed": 20260729,
        "runtime": {
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        "tensors": [
            {
                "name": "sample",
                "path": "tensors/sample.bin",
                "dtype": "int32",
                "shape": [2],
                "size_bytes": len(tensor_payload),
                "sha256": digest(tensor_payload),
            }
        ],
    }
    write_manifest(root, manifest)
    return manifest, lock_path


class BoundaryFixtureValidationTests(unittest.TestCase):
    def test_valid_fixture_with_required_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            _, lock_path = create_valid_fixture(root)
            self.assertEqual(
                MODULE.validate_boundary_fixture(root, lock_path),
                1,
            )

    def test_lock_is_optional_but_must_match_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            _, lock_path = create_valid_fixture(root)
            self.assertEqual(MODULE.validate_boundary_fixture(root), 1)
            lock_path.write_text('{"schema_version": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "oracle lock SHA-256 mismatch"):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_manifest_checksum_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            _, lock_path = create_valid_fixture(root)
            (root / "manifest.json.sha256").write_text(
                f"{'0' * 64}  manifest.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(RuntimeError, "does not exactly match"):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_missing_precision_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            del manifest["runtime"]["cudnn_allow_tf32"]
            write_manifest(root, manifest)
            with self.assertRaisesRegex(
                RuntimeError,
                "cudnn_allow_tf32 must be a boolean",
            ):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_missing_local_ar_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            del manifest["local_ar_seed"]
            write_manifest(root, manifest)
            with self.assertRaisesRegex(
                RuntimeError,
                "manifest.local_ar_seed must be an integer",
            ):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_out_of_range_local_ar_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            manifest["local_ar_seed"] = 2**32
            write_manifest(root, manifest)
            with self.assertRaisesRegex(
                RuntimeError,
                r"must be in \[0, 2\^32\)",
            ):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_high_matmul_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            manifest["runtime"]["float32_matmul_precision"] = "high"
            write_manifest(root, manifest)
            with self.assertRaisesRegex(
                RuntimeError,
                "runtime precision policy",
            ):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_cuda_tf32_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            manifest["runtime"]["cuda_matmul_allow_tf32"] = True
            write_manifest(root, manifest)
            with self.assertRaisesRegex(
                RuntimeError,
                "runtime precision policy",
            ):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_cudnn_tf32_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            manifest["runtime"]["cudnn_allow_tf32"] = True
            write_manifest(root, manifest)
            with self.assertRaisesRegex(
                RuntimeError,
                "runtime precision policy",
            ):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_unsafe_tensor_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            manifest["tensors"][0]["path"] = "../sample.bin"
            write_manifest(root, manifest)
            with self.assertRaisesRegex(RuntimeError, "safe POSIX relative path"):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_duplicate_tensor_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            manifest["tensors"].append(dict(manifest["tensors"][0]))
            write_manifest(root, manifest)
            with self.assertRaisesRegex(RuntimeError, "duplicate tensor names"):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_dtype_shape_size_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            manifest, lock_path = create_valid_fixture(root)
            manifest["tensors"][0]["shape"] = [3]
            write_manifest(root, manifest)
            with self.assertRaisesRegex(RuntimeError, "byte count mismatch"):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_tensor_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            _, lock_path = create_valid_fixture(root)
            (root / "tensors" / "sample.bin").write_bytes(b"\x00" * 8)
            with self.assertRaisesRegex(RuntimeError, "tensor SHA-256 mismatch"):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_tensor_file_size_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            _, lock_path = create_valid_fixture(root)
            (root / "tensors" / "sample.bin").write_bytes(b"\x00" * 4)
            with self.assertRaisesRegex(RuntimeError, "tensor size mismatch"):
                MODULE.validate_boundary_fixture(root, lock_path)

    def test_unlisted_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            _, lock_path = create_valid_fixture(root)
            (root / "extra.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "extra_files"):
                MODULE.validate_boundary_fixture(root, lock_path)


if __name__ == "__main__":
    unittest.main()

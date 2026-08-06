from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "export" / "export_text_encoder.py"
SPEC = importlib.util.spec_from_file_location("export_text_encoder", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_fixture_tensor(
    root: Path,
    name: str,
    dtype: str,
    shape: list[int],
    payload: bytes,
) -> dict:
    relative = Path("tensors") / f"{name}.bin"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "name": name,
        "path": relative.as_posix(),
        "dtype": dtype,
        "shape": shape,
        "size_bytes": len(payload),
        "sha256": digest(payload),
    }


class TextEncoderFixtureTests(unittest.TestCase):
    def make_fixture(self, directory: Path) -> tuple[Path, dict, str]:
        lock = {
            "model": {"sha256": "1" * 64},
            "codec": {"sha256": "2" * 64},
            "oracle_source": {"optimized_source_bundle_sha256": "3" * 64},
            "acceptance": {"receipt_sha256": "4" * 64},
        }
        lock_path = directory / "oracle-lock.json"
        lock_path.write_bytes(MODULE.canonical_json_bytes(lock))
        lock_sha256 = MODULE.sha256_file(lock_path)

        fixture = directory / "fixture"
        fixture.mkdir()
        records = [
            write_fixture_tensor(
                fixture,
                "input.text_token_ids",
                "int32",
                [1, 2],
                b"\x01\x00\x00\x00\x02\x00\x00\x00",
            ),
            write_fixture_tensor(
                fixture,
                "text.mask",
                "bool",
                [1, 2],
                b"\x01\x01",
            ),
            write_fixture_tensor(
                fixture,
                "text.condition",
                "bf16",
                [1, 2, MODULE.MODEL_WIDTH],
                b"\x00\x00" * 2 * MODULE.MODEL_WIDTH,
            ),
        ]
        manifest = {
            "oracle_lock_sha256": lock_sha256,
            "model_sha256": lock["model"]["sha256"],
            "codec_model_sha256": lock["codec"]["sha256"],
            "source_bundle_sha256": lock["oracle_source"][
                "optimized_source_bundle_sha256"
            ],
            "acceptance_receipt_sha256": lock["acceptance"]["receipt_sha256"],
            "decoder_contract": {"text_tokens": 2},
            "tensors": records,
        }
        manifest_payload = MODULE.canonical_json_bytes(manifest)
        (fixture / "manifest.json").write_bytes(manifest_payload)
        (fixture / "manifest.json.sha256").write_text(
            f"{digest(manifest_payload)}  manifest.json\n",
            encoding="ascii",
        )
        return fixture, lock, lock_sha256

    def test_accepts_locked_text_encoder_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, lock_sha256 = self.make_fixture(Path(directory))
            result = MODULE.verify_text_encoder_fixture(
                fixture,
                lock,
                lock_sha256,
            )
            self.assertEqual(result.text_tokens, 2)
            self.assertEqual(result.token_ids.dtype, "int32")
            self.assertEqual(result.condition.shape, (1, 2, MODULE.MODEL_WIDTH))

    def test_rejects_fixture_from_another_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, lock_sha256 = self.make_fixture(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "oracle_lock_sha256 mismatch"):
                MODULE.verify_text_encoder_fixture(
                    fixture,
                    lock,
                    "f" * 64,
                )

    def test_rejects_modified_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, lock_sha256 = self.make_fixture(Path(directory))
            (fixture / "tensors" / "text.mask.bin").write_bytes(b"\x01\x00")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                MODULE.verify_text_encoder_fixture(
                    fixture,
                    lock,
                    lock_sha256,
                )

    def test_rejects_wider_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, lock_sha256 = self.make_fixture(Path(directory))
            manifest_path = fixture / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tensors"][0]["dtype"] = "int64"
            manifest_payload = MODULE.canonical_json_bytes(manifest)
            manifest_path.write_bytes(manifest_payload)
            (fixture / "manifest.json.sha256").write_text(
                f"{digest(manifest_payload)}  manifest.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(RuntimeError, "must be int32"):
                MODULE.verify_text_encoder_fixture(
                    fixture,
                    lock,
                    lock_sha256,
                )


class AtomicPublishTests(unittest.TestCase):
    def test_publish_does_not_replace_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            (staging / "artifact").write_text("new", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            (output / "artifact").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                MODULE.publish_directory_no_replace(staging, output)
            self.assertEqual(
                (output / "artifact").read_text(encoding="utf-8"),
                "existing",
            )
            self.assertTrue(staging.exists())

    def test_publish_is_an_atomic_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            (staging / "artifact").write_text("complete", encoding="utf-8")
            output = root / "output"
            MODULE.publish_directory_no_replace(staging, output)
            self.assertFalse(staging.exists())
            self.assertEqual(
                (output / "artifact").read_text(encoding="utf-8"),
                "complete",
            )


if __name__ == "__main__":
    unittest.main()

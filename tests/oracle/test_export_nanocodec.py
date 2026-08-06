from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2] / "tools" / "export" / "export_nanocodec.py"
)
SPEC = importlib.util.spec_from_file_location("export_nanocodec", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_tensor(
    root: Path,
    name: str,
    dtype: str,
    shape: list[int],
) -> dict:
    item_size = {"fp32": 4, "int64": 8}[dtype]
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    payload = bytes(element_count * item_size)
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


class NanoCodecFixtureTests(unittest.TestCase):
    def make_fixture(
        self,
        directory: Path,
    ) -> tuple[Path, dict, str]:
        lock = {
            "model": {"sha256": "1" * 64},
            "codec": {"sha256": "2" * 64},
            "oracle_source": {
                "optimized_source_bundle_sha256": "3" * 64
            },
            "acceptance": {"receipt_sha256": "4" * 64},
        }
        lock_path = directory / "oracle-lock.json"
        lock_path.write_bytes(MODULE.canonical_json_bytes(lock))
        lock_sha256 = MODULE.sha256_file(lock_path)
        root = directory / "fixture"
        root.mkdir()
        schedule = [4, 8, 4]
        records = []
        for sequence, frame_count in enumerate(schedule):
            prefix = f"codec.chunk_{sequence:03d}"
            records.extend(
                (
                    write_tensor(
                        root,
                        f"{prefix}.codes",
                        "int64",
                        [1, 8, frame_count],
                    ),
                    write_tensor(
                        root,
                        f"{prefix}.pcm",
                        "fp32",
                        [1, frame_count * 1024],
                    ),
                    write_tensor(
                        root,
                        f"{prefix}.pcm_lengths",
                        "int64",
                        [1],
                    ),
                )
            )
        records.append(
            write_tensor(
                root,
                "codec.complete_pcm",
                "fp32",
                [1, sum(schedule) * 1024],
            )
        )
        manifest = {
            "oracle_lock_sha256": lock_sha256,
            "model_sha256": lock["model"]["sha256"],
            "codec_model_sha256": lock["codec"]["sha256"],
            "source_bundle_sha256": lock["oracle_source"][
                "optimized_source_bundle_sha256"
            ],
            "acceptance_receipt_sha256": lock["acceptance"][
                "receipt_sha256"
            ],
            "runtime": {
                "float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            },
            "codec_contract": {
                "frame_schedule": schedule,
                "sample_rate_hz": 22050,
                "samples_per_frame": 1024,
                "valid_codec_frames": sum(schedule),
            },
            "tensors": records,
        }
        payload = MODULE.canonical_json_bytes(manifest)
        (root / "manifest.json").write_bytes(payload)
        (root / "manifest.json.sha256").write_text(
            f"{digest(payload)}  manifest.json\n",
            encoding="ascii",
        )
        return root, lock, lock_sha256

    def test_accepts_locked_codec_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, lock_sha256 = self.make_fixture(Path(directory))
            result = MODULE.verify_nanocodec_fixture(
                fixture,
                lock,
                lock_sha256,
            )
            self.assertEqual(result.frame_schedule, (4, 8, 4))
            self.assertEqual(result.valid_codec_frames, 16)

    def test_rejects_fixture_from_another_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, _ = self.make_fixture(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "oracle_lock_sha256"):
                MODULE.verify_nanocodec_fixture(
                    fixture,
                    lock,
                    "f" * 64,
                )

    def test_rejects_nonterminal_partial_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, lock_sha256 = self.make_fixture(Path(directory))
            manifest_path = fixture / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["codec_contract"]["frame_schedule"] = [4, 4, 8]
            payload = MODULE.canonical_json_bytes(manifest)
            manifest_path.write_bytes(payload)
            (fixture / "manifest.json.sha256").write_text(
                f"{digest(payload)}  manifest.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "non-terminal partial",
            ):
                MODULE.verify_nanocodec_fixture(
                    fixture,
                    lock,
                    lock_sha256,
                )

    def test_rejects_tf32_fixture_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, lock, lock_sha256 = self.make_fixture(Path(directory))
            manifest_path = fixture / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime"]["cudnn_allow_tf32"] = True
            payload = MODULE.canonical_json_bytes(manifest)
            manifest_path.write_bytes(payload)
            (fixture / "manifest.json.sha256").write_text(
                f"{digest(payload)}  manifest.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "runtime precision policy mismatch",
            ):
                MODULE.verify_nanocodec_fixture(
                    fixture,
                    lock,
                    lock_sha256,
                )


class PlanIntrospectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = (
            MODULE.StateContract("pre_conv.input_history", (1, 32, 6)),
            MODULE.StateContract(
                "upsample_convs.0.pending_overlap",
                (1, 432, 8),
            ),
        )

    def metadata(self, route: str) -> dict:
        inputs, outputs = MODULE.expected_tensor_contract(
            self.states,
            route=route,
        )
        tensors = [
            {
                "name": name,
                "mode": "input",
                "dtype": dtype,
                "shape": [
                    -1 if isinstance(dimension, str) else dimension
                    for dimension in shape
                ],
            }
            for name, (dtype, shape) in inputs.items()
        ]
        tensors.extend(
            {
                "name": name,
                "mode": "output",
                "dtype": dtype,
                "shape": [
                    -1 if isinstance(dimension, str) else dimension
                    for dimension in shape
                ],
            }
            for name, (dtype, shape) in outputs.items()
        )
        profile = None
        if route == "tail_1_8":
            profile = {
                "codec_tokens": {
                    "min": [1, 8, 1],
                    "opt": [1, 8, 4],
                    "max": [1, 8, 8],
                }
            }
        return {
            "num_io_tensors": len(tensors),
            "num_optimization_profiles": 1,
            "tensors": tensors,
            "profile": profile,
        }

    def test_accepts_exact_tail_profile(self) -> None:
        MODULE.verify_inspected_plan_contract(
            self.metadata("tail_1_8"),
            route="tail_1_8",
            state_contracts=self.states,
        )

    def test_rejects_missing_state_output(self) -> None:
        metadata = self.metadata("steady_8")
        metadata["tensors"].pop()
        metadata["num_io_tensors"] -= 1
        with self.assertRaisesRegex(RuntimeError, "bindings differ"):
            MODULE.verify_inspected_plan_contract(
                metadata,
                route="steady_8",
                state_contracts=self.states,
            )

    def test_rejects_widened_tail_profile(self) -> None:
        metadata = self.metadata("tail_1_8")
        metadata["profile"]["codec_tokens"]["max"] = [1, 8, 12]
        with self.assertRaisesRegex(RuntimeError, "profile mismatch"):
            MODULE.verify_inspected_plan_contract(
                metadata,
                route="tail_1_8",
                state_contracts=self.states,
            )


class AtomicPublishTests(unittest.TestCase):
    def test_does_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            (staging / "value").write_text("new", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            (output / "value").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                MODULE.publish_directory_no_replace(staging, output)
            self.assertEqual(
                (output / "value").read_text(encoding="utf-8"),
                "old",
            )
            self.assertTrue(staging.exists())


if __name__ == "__main__":
    unittest.main()

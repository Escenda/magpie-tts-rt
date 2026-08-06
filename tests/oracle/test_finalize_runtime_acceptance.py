from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy


EXPORT_TOOLS = Path(__file__).parents[2] / "tools" / "export"
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))
SCRIPT = EXPORT_TOOLS / "finalize_runtime_acceptance.py"
SPEC = importlib.util.spec_from_file_location(
    "finalize_runtime_acceptance",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class FixtureTensor:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class GoldenFixtureInput:
    tensors: dict[str, FixtureTensor]
    text_tokens: int


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class CanonicalEncodingTests(unittest.TestCase):
    def test_decodes_little_endian_signed_int32(self) -> None:
        payload = b"".join(
            value.to_bytes(4, "little", signed=True)
            for value in (0, 3358, 2**31 - 1)
        )
        self.assertEqual(
            MODULE.little_endian_int32_values(payload),
            (0, 3358, 2**31 - 1),
        )

    def test_rejects_partial_int32_payload(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "INT32"):
            MODULE.little_endian_int32_values(b"\x00\x01")

    def test_chunk_hash_order_differs_from_complete_tensor_order(self) -> None:
        codes = numpy.arange(1 * 8 * 5, dtype="<i8").reshape(1, 8, 5)
        chunked = MODULE.chunked_codec_code_bytes(codes, (2, 3))
        complete = numpy.ascontiguousarray(codes, dtype="<i8").tobytes()
        expected = b"".join(
            (
                numpy.ascontiguousarray(codes[:, :, :2], dtype="<i8").tobytes(),
                numpy.ascontiguousarray(codes[:, :, 2:], dtype="<i8").tobytes(),
            )
        )
        self.assertEqual(chunked, expected)
        self.assertNotEqual(chunked, complete)

    def test_timestamp_is_canonical_utc(self) -> None:
        timestamp = MODULE.utc_timestamp()
        self.assertTrue(timestamp.endswith("Z"))
        self.assertNotIn("+00:00", timestamp)


class GoldenDocumentTests(unittest.TestCase):
    def test_builds_exact_fixture_and_receipt_key_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_payload = b"".join(
                token.to_bytes(4, "little", signed=True)
                for token in (842, 3358)
            )
            code_values = numpy.arange(1 * 8 * 3, dtype="<i8").reshape(1, 8, 3)
            code_payload = code_values.tobytes()
            token_path = root / "tokens.bin"
            code_path = root / "codes.bin"
            baked_path = root / "baked.bin"
            token_path.write_bytes(token_payload)
            code_path.write_bytes(code_payload)
            baked_path.write_bytes(b"baked")
            fixture = GoldenFixtureInput(
                tensors={
                    "input.text_token_ids": FixtureTensor(
                        token_path,
                        "int32",
                        (1, 2),
                        sha256(token_payload),
                    ),
                    "cfg.sofia_prefix": FixtureTensor(
                        baked_path,
                        "bf16",
                        (1, 217, 768),
                        sha256(b"baked"),
                    ),
                    "generation.codes": FixtureTensor(
                        code_path,
                        "int64",
                        (1, 8, 3),
                        sha256(code_payload),
                    ),
                },
                text_tokens=2,
            )
            manifest = {
                "fixture_id": "fixture-v1",
                "text": "こんにちは。",
                "local_ar_seed": 7,
            }

            golden_fixture, receipt, hashes = MODULE.build_golden_documents(
                fixture=fixture,
                frame_schedule=(1, 2),
                valid_codec_frames=3,
                manifest=manifest,
                tokenizer_identity_sha256="1" * 64,
                lock_sha256="2" * 64,
                pcm_sha256="3" * 64,
                pcm_sample_count=3072,
                created_at_utc="2026-07-30T00:00:00.000000Z",
            )

            self.assertEqual(
                set(golden_fixture),
                {
                    "schema_version",
                    "fixture_id",
                    "prepared_token_ids",
                    "seed",
                    "tokenizer_identity_sha256",
                    "oracle_lock_sha256",
                    "normalized_text_sha256",
                    "token_ids_sha256",
                    "baked_context_sha256",
                    "expected",
                },
            )
            self.assertEqual(
                set(receipt),
                {
                    "receipt_version",
                    "created_at_utc",
                    "normalized_text_sha256",
                    "token_ids_sha256",
                    "baked_context_sha256",
                    "seed",
                    "decoder_tokens_sha256",
                    "codec_codes_sha256",
                    "codec_frame_count",
                    "pcm_f32le_sha256",
                    "sample_count",
                    "initial_frames",
                    "steady_frames",
                    "tail_min_frames",
                    "tail_max_frames",
                    "eos_frame_is_audio",
                    "zero_frame_finalization",
                },
            )
            self.assertEqual(hashes.token_ids_sha256, sha256(token_payload))
            self.assertEqual(hashes.decoder_tokens_sha256, sha256(code_payload))
            self.assertEqual(golden_fixture["prepared_token_ids"], [842, 3358])
            self.assertEqual(receipt["codec_frame_count"], 3)
            self.assertIs(receipt["eos_frame_is_audio"], False)
            self.assertEqual(
                receipt["zero_frame_finalization"],
                "control_marker_without_codec_invocation",
            )


class PluginProvenanceTests(unittest.TestCase):
    def test_accepts_one_plugin_digest_across_all_generation_plans(self) -> None:
        MODULE.require_common_plugin_sha256(
            text_encoder="1" * 64,
            main_decoder="1" * 64,
            local_ar="1" * 64,
        )

    def test_rejects_main_decoder_plugin_digest_mismatch(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Text Encoder, Main Decoder, and Local AR",
        ):
            MODULE.require_common_plugin_sha256(
                text_encoder="1" * 64,
                main_decoder="2" * 64,
                local_ar="1" * 64,
            )

    def test_main_decoder_promotion_receipt_binds_required_plugin(self) -> None:
        engine = MODULE.AuthenticatedArtifact(
            path=Path("/unopened/main-decoder-step.plan"),
            relative_path="main-decoder-step.plan",
            size_bytes=101,
            sha256="1" * 64,
        )
        plugin = MODULE.AuthenticatedArtifact(
            path=Path("/unopened/libmagpie_tts_rt_plugins.so.0"),
            relative_path="libmagpie_tts_rt_plugins.so.0",
            size_bytes=202,
            sha256="2" * 64,
        )

        receipt = MODULE.promotion_receipt(
            artifact_role="main_decoder",
            candidate_receipt_sha256="3" * 64,
            sequence_receipt_sha256="4" * 64,
            lock_sha256="5" * 64,
            canonical_fixture_manifest_sha256="6" * 64,
            engines=[("main_decoder_step", engine)],
            created_at_utc="2026-07-30T00:00:00.000000Z",
            plugin=plugin,
        )

        self.assertEqual(
            receipt["plugin"],
            {"sha256": "2" * 64, "size_bytes": 202},
        )


class ReceiptAuthenticationTests(unittest.TestCase):
    def write_directory(self, root: Path) -> None:
        artifact = root / "artifact.bin"
        artifact.write_bytes(b"artifact")
        receipt = {
            "schema_version": 1,
            "artifact_role": "test_role",
            "status": "accepted",
            "artifacts": [
                {
                    "path": "artifact.bin",
                    "size_bytes": 8,
                    "sha256": sha256(b"artifact"),
                }
            ],
        }
        payload = (
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        (root / "receipt.json").write_bytes(payload)
        (root / "receipt.json.sha256").write_text(
            f"{sha256(payload)}  receipt.json\n",
            encoding="ascii",
        )

    def test_authenticates_exact_entry_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_directory(root)
            result = MODULE.authenticate_receipt_directory(
                root,
                receipt_name="receipt.json",
                checksum_name="receipt.json.sha256",
                artifact_role="test_role",
                status="accepted",
            )
            self.assertEqual(result.artifacts["artifact.bin"].sha256, sha256(b"artifact"))

    def test_rejects_unlisted_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_directory(root)
            (root / "unlisted.bin").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "entry set mismatch"):
                MODULE.authenticate_receipt_directory(
                    root,
                    receipt_name="receipt.json",
                    checksum_name="receipt.json.sha256",
                    artifact_role="test_role",
                    status="accepted",
                )

    def test_final_reauthentication_rejects_same_size_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-model-receipt.json"
            payload = b'{"accepted":true}\n'
            path.write_bytes(payload)
            lock = {
                "acceptance": {
                    "receipt_name": path.name,
                    "receipt_sha256": sha256(payload),
                    "receipt_size_bytes": len(payload),
                }
            }
            path.write_bytes(b'{"accepted":null}\n')
            self.assertEqual(path.stat().st_size, len(payload))
            with self.assertRaisesRegex(
                RuntimeError,
                "differs from oracle lock",
            ):
                MODULE.require_locked_source_receipt(path, lock)


if __name__ == "__main__":
    unittest.main()

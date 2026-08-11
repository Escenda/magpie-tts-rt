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


def cublas_identity() -> dict[str, MODULE.JsonValue]:
    return {
        "api_version_integer": 130400,
        "library": {
            "soname": "libcublas.so.13",
            "size_bytes": 67_751_616,
            "sha256": (
                "826486b8869144621e3a477cddcd28f56733c7c80c6f998b"
                "898384fc09e10f91"
            ),
        },
        "lt_library": {
            "soname": "libcublasLt.so.13",
            "size_bytes": 606_744_240,
            "sha256": (
                "b7aa42c190c2e7490abd6ea987883e05678e26222b7f9f1c9"
                "b96374fcbddbf04"
            ),
        },
    }


def cuda_identity() -> dict[str, MODULE.JsonValue]:
    return {
        "cuda_driver_api_version_integer": 13020,
        "cuda_runtime_version_integer": 13020,
        "nvidia_driver_version": "595.78",
    }


def codec_restore() -> MODULE.AuthenticatedCodecRestore:
    return MODULE.AuthenticatedCodecRestore(
        embedded_codec_model_id="nvidia/locked-codec",
        codec_model_sha256="9" * 64,
        codec_model_size_bytes=123,
    )


def codec_restore_json() -> dict[str, MODULE.JsonValue]:
    return {
        "embedded_codec_model_id": "nvidia/locked-codec",
        "codec_model_sha256": "9" * 64,
        "codec_model_size_bytes": 123,
        "codec_resolution": "authenticated_local_file",
        "use_scl_loss": False,
        "network_resolution": False,
    }


def write_sequence_receipt(
    root: Path,
    *,
    include_cuda: bool,
    include_cublas: bool,
) -> tuple[
    MODULE.AuthenticatedTextEncoderExport,
    MODULE.AuthenticatedPlanExport,
    MODULE.AuthenticatedLocalARExport,
]:
    restore_sha256 = MODULE.sha256_file(
        (EXPORT_TOOLS / "locked_magpie_restore.py").resolve(strict=True)
    )
    text = MODULE.AuthenticatedTextEncoderExport(
        root=root,
        receipt_sha256="1" * 64,
        oracle_lock_sha256="a" * 64,
        locked_magpie_restore_sha256=restore_sha256,
        codec_restore=codec_restore(),
        source_fixture_manifest_sha256="b" * 64,
        required_plugin_sha256="2" * 64,
        plan=root / "text.plan",
        plan_sha256="3" * 64,
        tensorrt_version="10.16.2.10",
        torch_cuda_build="13.0",
        gpu_name="NVIDIA Thor",
        gpu_compute_capability=(11, 0),
    )
    main = MODULE.AuthenticatedPlanExport(
        root=root,
        receipt_sha256="4" * 64,
        oracle_lock_sha256="a" * 64,
        locked_magpie_restore_sha256=restore_sha256,
        codec_restore=codec_restore(),
        prefill_plan=root / "prefill.plan",
        prefill_plan_sha256="5" * 64,
        step_plan=root / "step.plan",
        step_plan_sha256="6" * 64,
        source_fixture_manifest_sha256="b" * 64,
        required_plugin_sha256="2" * 64,
        mode8_validation_receipt_sha256="e" * 64,
        mode8_class_table_sha256="f" * 64,
        cuda_identity=MODULE.parse_cuda_runtime_identity(
            cuda_identity(), "test.cuda"
        ),
        cublas_identity=MODULE.parse_cublas_runtime_identity(
            cublas_identity(), "test.cublas"
        ),
        tensorrt_version="10.16.2.10",
        torch_cuda_build="13.0",
        gpu_name="NVIDIA Thor",
        gpu_compute_capability=(11, 0),
    )
    local = MODULE.AuthenticatedLocalARExport(
        root=root,
        receipt_sha256="7" * 64,
        oracle_lock_sha256="a" * 64,
        locked_magpie_restore_sha256=restore_sha256,
        codec_restore=codec_restore(),
        source_fixture_manifest_sha256s=("b" * 64,),
        plan=root / "local.plan",
        plan_sha256="8" * 64,
        plugin=root / "plugin.so",
        plugin_sha256="2" * 64,
        tensorrt_version="10.16.2.10",
        torch_cuda_build="13.0",
        gpu_name="NVIDIA Thor",
        gpu_compute_capability=(11, 0),
    )
    fixture_sha256s = ("b" * 64, "c" * 64, "d" * 64)
    runtime: dict[str, MODULE.JsonValue] = {
        "tensorrt": "10.16.2.10",
    }
    if include_cublas:
        runtime["cublas"] = cublas_identity()
    if include_cuda:
        runtime["cuda"] = cuda_identity()
    runtime["mode8_class_table_sha256"] = main.mode8_class_table_sha256
    receipt: dict[str, MODULE.JsonValue] = {
        "schema_version": 1,
        "artifact_role": "text_main_local_ar_sequence_validation",
        "status": "accepted",
        "fixture_count": 3,
        "exact_code_case_count": 3,
        "all_codes_exact": True,
        "source": {
            "oracle_lock_sha256": "a" * 64,
            "locked_magpie_restore_sha256": restore_sha256,
            "codec_restore": codec_restore_json(),
            "text_encoder_export_receipt_sha256": text.receipt_sha256,
            "text_encoder_plan_sha256": text.plan_sha256,
            "main_export_receipt_sha256": main.receipt_sha256,
            "main_mode8_validation_receipt_sha256": (
                main.mode8_validation_receipt_sha256
            ),
            "main_prefill_plan_sha256": main.prefill_plan_sha256,
            "main_step_plan_sha256": main.step_plan_sha256,
            "local_ar_export_receipt_sha256": local.receipt_sha256,
            "local_ar_plan_sha256": local.plan_sha256,
            "local_ar_plugin_sha256": local.plugin_sha256,
        },
        "runtime": runtime,
        "session_policy": {
            "main_execution_status_recurrence": (
                "int32-device-scalar-sticky-12-layer"
            ),
            "main_execution_status_checked_before_next_local_ar": True,
        },
        "fixtures": [
            {
                "fixture_manifest_sha256": manifest_sha256,
                "code_exact": True,
                "generated_codes_sha256": f"{index + 9:064x}",
                "expected_codes_sha256": f"{index + 9:064x}",
                "local_ar_invocations": 3,
                "main_execution_status_check_count": 2,
                "main_execution_status_all_zero": True,
            }
            for index, manifest_sha256 in enumerate(fixture_sha256s)
        ],
        "artifacts": [],
    }
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (root / MODULE.SEQUENCE_RECEIPT).write_bytes(payload)
    (root / MODULE.SEQUENCE_RECEIPT_CHECKSUM).write_text(
        f"{sha256(payload)}  {MODULE.SEQUENCE_RECEIPT}\n",
        encoding="ascii",
    )
    return text, main, local


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

    def test_sequence_receipt_requires_exact_cublas_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text, main, local = write_sequence_receipt(
                root,
                include_cuda=True,
                include_cublas=True,
            )
            result = MODULE.authenticate_sequence_receipt(
                root,
                lock_sha256="a" * 64,
                canonical_fixture_manifest_sha256="b" * 64,
                text=text,
                main=main,
                local=local,
            )
            self.assertEqual(result.cublas_identity.to_json(), cublas_identity())
            self.assertEqual(result.cuda_identity.to_json(), cuda_identity())

    def test_sequence_receipt_without_cublas_identity_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text, main, local = write_sequence_receipt(
                root,
                include_cuda=True,
                include_cublas=False,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "sequence.runtime.cublas",
            ):
                MODULE.authenticate_sequence_receipt(
                    root,
                    lock_sha256="a" * 64,
                    canonical_fixture_manifest_sha256="b" * 64,
                    text=text,
                    main=main,
                    local=local,
                )

    def test_sequence_receipt_without_cuda_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text, main, local = write_sequence_receipt(
                root,
                include_cuda=False,
                include_cublas=True,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "sequence.runtime.cuda",
            ):
                MODULE.authenticate_sequence_receipt(
                    root,
                    lock_sha256="a" * 64,
                    canonical_fixture_manifest_sha256="b" * 64,
                    text=text,
                    main=main,
                    local=local,
                )


if __name__ == "__main__":
    unittest.main()

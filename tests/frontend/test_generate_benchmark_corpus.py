from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tools.frontend.generate_benchmark_corpus import (
    OUTPUT_SCHEMA,
    REQUIRED_CATEGORIES,
    CorpusSourceError,
    JsonValue,
    generate_corpus_bytes,
    load_benchmark_source,
    publish_no_replace,
    require_absolute_regular_file,
    require_new_output_path,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / "reference" / "benchmark" / "thor-ja-v1-source.json"
)
IDENTITY = (
    "abfe10eadb6900f68a7bd65f8f63e593d6ad0e967bb3d3532d3674960f0c57ad"
)


@dataclass(frozen=True)
class FakePreparedUtterance:
    global_token_ids: tuple[int, ...]


class FakeJapaneseFrontend:
    def prepare(self, source_text: str) -> FakePreparedUtterance:
        digest = hashlib.sha256(source_text.encode("utf-8")).digest()
        core = tuple(
            300 + int.from_bytes(digest[offset : offset + 2], "big") % 2_900
            for offset in range(0, 8, 2)
        )
        return FakePreparedUtterance((*core, 3_358))


def parse_json_lines(payload: bytes) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for line in payload.decode("utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AssertionError("generated JSONL record is not an object")
        records.append(value)
    return records


class GenerateBenchmarkCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_benchmark_source(SOURCE)
        cls.frontend = FakeJapaneseFrontend()

    def test_tracked_source_has_required_coverage_and_unique_pins(self) -> None:
        self.assertEqual(self.source.corpus_id, "thor-ja-v1")
        self.assertEqual(self.source.tokenizer_identity_sha256, IDENTITY)
        self.assertEqual(len(self.source.normal_cases), 108)
        self.assertEqual(
            Counter(case.category for case in self.source.normal_cases),
            Counter({category: 12 for category in REQUIRED_CATEGORIES}),
        )
        all_cases = (*self.source.normal_cases, self.source.cancel_case)
        self.assertEqual(
            len({case.case_id for case in all_cases}), len(all_cases)
        )
        self.assertEqual(
            len({case.random_seed for case in all_cases}), len(all_cases)
        )
        self.assertEqual(
            len({case.source_text for case in all_cases}), len(all_cases)
        )

    def test_generation_is_byte_deterministic_and_matches_runner_schema(
        self,
    ) -> None:
        first = generate_corpus_bytes(
            self.source, self.frontend, IDENTITY
        )
        second = generate_corpus_bytes(
            self.source, self.frontend, IDENTITY
        )

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotIn(b"\r", first)
        records = parse_json_lines(first)
        self.assertEqual(len(records), 110)
        header = records[0]
        self.assertEqual(
            set(header),
            {
                "record_type",
                "schema_version",
                "corpus_id",
                "tokenizer_identity_sha256",
                "normal_case_count",
            },
        )
        self.assertEqual(header["record_type"], "header")
        self.assertEqual(header["schema_version"], OUTPUT_SCHEMA)
        self.assertEqual(header["corpus_id"], "thor-ja-v1")
        self.assertEqual(header["tokenizer_identity_sha256"], IDENTITY)
        self.assertEqual(header["normal_case_count"], 108)

        expected_cases = (
            *self.source.normal_cases,
            self.source.cancel_case,
        )
        for index, (record, expected) in enumerate(
            zip(records[1:], expected_cases, strict=True)
        ):
            with self.subTest(case=expected.case_id):
                self.assertEqual(
                    set(record),
                    {
                        "record_type",
                        "case_id",
                        "source_text",
                        "source_text_sha256",
                        "prepared_token_ids",
                        "random_seed",
                    },
                )
                self.assertEqual(
                    record["record_type"],
                    "cancel_case" if index == 108 else "case",
                )
                self.assertEqual(record["case_id"], expected.case_id)
                self.assertEqual(record["source_text"], expected.source_text)
                self.assertEqual(
                    record["source_text_sha256"],
                    hashlib.sha256(
                        expected.source_text.encode("utf-8")
                    ).hexdigest(),
                )
                self.assertEqual(
                    record["prepared_token_ids"],
                    list(
                        self.frontend.prepare(
                            expected.source_text
                        ).global_token_ids
                    ),
                )
                self.assertEqual(record["random_seed"], expected.random_seed)

    def test_identity_mismatch_fails_before_frontend_preparation(self) -> None:
        wrong_identity = "1" * 64
        with self.assertRaisesRegex(
            CorpusSourceError, "identity differs"
        ):
            generate_corpus_bytes(
                self.source, self.frontend, wrong_identity
            )

    def test_no_replace_publication_preserves_first_result(self) -> None:
        payload = generate_corpus_bytes(
            self.source, self.frontend, IDENTITY
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "corpus.jsonl"
            publish_no_replace(destination, payload)

            self.assertEqual(destination.read_bytes(), payload)
            with self.assertRaisesRegex(
                CorpusSourceError, "already exists"
            ):
                publish_no_replace(destination, b"different\n")
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.tmp.*")),
                [],
            )

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "duplicate.json"
            source.write_text(
                '{"schema_version":"one","schema_version":"two"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CorpusSourceError, "duplicate JSON key"
            ):
                load_benchmark_source(source)

    def test_unknown_source_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "unknown.json"
            text = SOURCE.read_text(encoding="utf-8")
            source.write_text(
                text.replace(
                    '"schema_version":',
                    '"unknown": true, "schema_version":',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CorpusSourceError, "keys mismatch"
            ):
                load_benchmark_source(source)

    def test_acceptance_case_count_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "wrong-count.json"
            source.write_text(
                SOURCE.read_text(encoding="utf-8").replace(
                    '"normal_case_count": 108',
                    '"normal_case_count": 107',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CorpusSourceError, "exactly 108"
            ):
                load_benchmark_source(source)

    def test_duplicate_case_pin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "duplicate-case.json"
            source.write_text(
                SOURCE.read_text(encoding="utf-8").replace(
                    '"case_id": "cancel-001"',
                    '"case_id": "short-001"',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CorpusSourceError, "duplicate case_id"
            ):
                load_benchmark_source(source)

    def test_relative_and_symlink_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(CorpusSourceError, "absolute"):
            require_absolute_regular_file(Path("source.json"), "source")
        with self.assertRaisesRegex(CorpusSourceError, "absolute"):
            require_new_output_path(Path("corpus.jsonl"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            link = root / "source-link.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(CorpusSourceError, "non-symlink"):
                require_absolute_regular_file(link, "source")


if __name__ == "__main__":
    unittest.main()

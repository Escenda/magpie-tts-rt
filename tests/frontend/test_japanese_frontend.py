from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.frontend.japanese_frontend import (
    AlignmentProof,
    AmbiguousAlignmentError,
    FrontendError,
    LockedJapaneseFrontend,
    PreparedUtterance,
    SourceSegment,
    SourceNormalizationError,
    _certified_segment_ranges,
    load_frontend_spec,
    tokenizer_identity_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "reference" / "oracle-lock.json"
FIXTURE = (
    ROOT
    / "tests"
    / "frontend"
    / "fixtures"
    / "japanese-source-spans-v1.json"
)


class FixtureRunner:
    def __init__(self, prefix_records: Sequence[Mapping[str, object]]):
        rows_by_prefix: dict[str, list[dict[str, str | int]]] = {}
        for record_index, record in enumerate(prefix_records):
            prefix = record.get("normalized_prefix")
            rows = record.get("rows")
            if not isinstance(prefix, str) or not isinstance(rows, list):
                raise AssertionError(
                    f"invalid prefix fixture record {record_index}"
                )
            parsed_rows: list[dict[str, str | int]] = []
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise AssertionError(
                        f"invalid row {row_index} for prefix {prefix!r}"
                    )
                parsed_row: dict[str, str | int] = {}
                for key, value in row.items():
                    if not isinstance(key, str) or not isinstance(
                        value, (str, int)
                    ):
                        raise AssertionError(
                            f"invalid field in prefix {prefix!r}"
                        )
                    parsed_row[key] = value
                parsed_rows.append(parsed_row)
            previous = rows_by_prefix.get(prefix)
            if previous is not None and previous != parsed_rows:
                raise AssertionError(
                    f"conflicting OpenJTalk rows for prefix {prefix!r}"
                )
            rows_by_prefix[prefix] = parsed_rows
        self._rows_by_prefix = rows_by_prefix

    def __call__(self, text: str) -> list[dict[str, str | int]]:
        try:
            return copy.deepcopy(self._rows_by_prefix[text])
        except KeyError as error:
            raise AssertionError(
                f"fixture has no OpenJTalk rows for prefix {text!r}"
            ) from error


def load_fixture() -> dict[str, object]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("fixture root is not an object")
    return value


def records(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise AssertionError(f"{label} is not an array")
    parsed: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise AssertionError(f"{label}[{index}] is not an object")
        parsed.append(item)
    return parsed


class JapaneseFrontendGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture()
        cls.spec = load_frontend_spec(LOCK, FIXTURE)
        cls.cases = records(cls.fixture.get("cases"), "cases")
        cls.negative_cases = records(
            cls.fixture.get("negative_cases"), "negative_cases"
        )

    def test_fixture_is_bound_to_current_oracle_lock(self) -> None:
        oracle = self.fixture.get("oracle")
        self.assertIsInstance(oracle, dict)
        assert isinstance(oracle, dict)
        expected = hashlib.sha256(LOCK.read_bytes()).hexdigest()
        self.assertEqual(oracle.get("oracle_lock_sha256"), expected)
        self.assertEqual(
            oracle.get("token_table_sha256"),
            self.spec.token_table_sha256,
        )

    def test_long_text_split_uses_only_certified_source_boundaries(self) -> None:
        prepared = PreparedUtterance(
            source_text="a。bb。c。",
            normalized_text="a。bb。c。",
            g2p_tokens=(),
            global_token_ids=(),
            tokens=(),
            progress=(),
            proof=AlignmentProof(
                method="fixture",
                prepared_char_count=7,
                core_token_count=3,
                certified_prefix_count=3,
                segments=(
                    SourceSegment(0, 2, 0, 4, 0, 1),
                    SourceSegment(2, 5, 4, 9, 1, 2),
                    SourceSegment(5, 7, 9, 13, 2, 3),
                ),
            ),
        )

        ranges = _certified_segment_ranges(
            prepared,
            lambda start, end: (end - start) * 20,
        )

        self.assertEqual(ranges, ((0, 2), (2, 5), (5, 7)))
        self.assertTrue(
            all(
                end
                in {
                    segment.source_char_end
                    for segment in prepared.proof.segments
                }
                for _, end in ranges
            )
        )
        self.assertEqual(
            tokenizer_identity_sha256(LOCK, FIXTURE),
            "abfe10eadb6900f68a7bd65f8f63e593d6ad0e967bb3d3532d3674960f0c57ad",
        )

    def test_all_golden_cases_match_exactly(self) -> None:
        for case in self.cases:
            case_id = case.get("id")
            source_text = case.get("source_text")
            expected = case.get("expected")
            prefix_rows = case.get("prefix_frontend_rows")
            self.assertIsInstance(source_text, str)
            self.assertIsInstance(expected, dict)
            self.assertIsInstance(prefix_rows, list)
            assert isinstance(source_text, str)
            assert isinstance(expected, dict)
            assert isinstance(prefix_rows, list)
            frontend = LockedJapaneseFrontend(
                self.spec, FixtureRunner(prefix_rows)
            )
            with self.subTest(case=case_id):
                actual = frontend.prepare(source_text).to_json_record()
                self.assertEqual(actual, expected)

    def test_locked_minimal_verifies_g2p_tokenizer_and_offset(self) -> None:
        case = next(
            item for item in self.cases if item.get("id") == "locked_minimal"
        )
        prefix_rows = case["prefix_frontend_rows"]
        self.assertIsInstance(prefix_rows, list)
        assert isinstance(prefix_rows, list)
        frontend = LockedJapaneseFrontend(
            self.spec, FixtureRunner(prefix_rows)
        )
        frontend._verify_golden()

    def test_proofs_cover_tokens_and_source_in_order(self) -> None:
        for case in self.cases:
            expected = case["expected"]
            self.assertIsInstance(expected, dict)
            assert isinstance(expected, dict)
            source_text = expected["source_text"]
            tokens = expected["tokens"]
            progress = expected["progress"]
            proof = expected["proof"]
            self.assertIsInstance(source_text, str)
            self.assertIsInstance(tokens, list)
            self.assertIsInstance(progress, list)
            self.assertIsInstance(proof, dict)
            assert isinstance(source_text, str)
            assert isinstance(tokens, list)
            assert isinstance(progress, list)
            assert isinstance(proof, dict)
            segments = proof["segments"]
            self.assertIsInstance(segments, list)
            assert isinstance(segments, list)

            previous_source_end = 0
            previous_core_end = 0
            for segment in segments:
                self.assertIsInstance(segment, dict)
                assert isinstance(segment, dict)
                self.assertEqual(
                    segment["source_char_start"], previous_source_end
                )
                self.assertEqual(
                    segment["core_token_start"], previous_core_end
                )
                self.assertGreater(
                    segment["source_char_end"], previous_source_end
                )
                self.assertGreater(
                    segment["core_token_end"], previous_core_end
                )
                previous_source_end = segment["source_char_end"]
                previous_core_end = segment["core_token_end"]
            self.assertEqual(previous_core_end, proof["core_token_count"])

            self.assertEqual(len(progress), len(tokens) + 1)
            previous_char = 0
            previous_utf8 = 0
            for token_position, item in enumerate(progress):
                self.assertIsInstance(item, dict)
                assert isinstance(item, dict)
                self.assertEqual(
                    item["committed_text_tokens"], token_position
                )
                self.assertGreaterEqual(item["source_char_end"], previous_char)
                self.assertGreaterEqual(item["source_utf8_end"], previous_utf8)
                previous_char = item["source_char_end"]
                previous_utf8 = item["source_utf8_end"]
            self.assertEqual(previous_char, len(source_text))
            self.assertEqual(
                previous_utf8, len(source_text.encode("utf-8"))
            )

            source_bytes = source_text.encode("utf-8")
            for token in tokens:
                self.assertIsInstance(token, dict)
                assert isinstance(token, dict)
                start = token["source_utf8_start"]
                end = token["source_utf8_end"]
                self.assertIsInstance(start, int)
                self.assertIsInstance(end, int)
                assert isinstance(start, int)
                assert isinstance(end, int)
                source_bytes[start:end].decode("utf-8")

    def test_pitch_marker_and_mora_share_an_atomic_unit(self) -> None:
        for case in self.cases:
            expected = case["expected"]
            self.assertIsInstance(expected, dict)
            assert isinstance(expected, dict)
            tokens = expected["tokens"]
            self.assertIsInstance(tokens, list)
            assert isinstance(tokens, list)
            units: dict[int, list[dict[str, object]]] = {}
            for token in tokens:
                self.assertIsInstance(token, dict)
                assert isinstance(token, dict)
                unit = token.get("unit_index")
                if isinstance(unit, int):
                    units.setdefault(unit, []).append(token)
            for unit_tokens in units.values():
                kinds = [token.get("kind") for token in unit_tokens]
                if "pitch" in kinds:
                    self.assertEqual(kinds[0], "pitch")
                    self.assertTrue(all(kind == "mora" for kind in kinds[1:]))
                    self.assertGreaterEqual(len(kinds), 2)

    def test_numeric_expansions_do_not_commit_unspoken_suffixes(self) -> None:
        date = next(
            item for item in self.cases if item.get("id") == "digits_date"
        )
        expected = date["expected"]
        self.assertIsInstance(expected, dict)
        assert isinstance(expected, dict)
        tokens = expected["tokens"]
        self.assertIsInstance(tokens, list)
        assert isinstance(tokens, list)
        first_segment_tokens = [
            token
            for token in tokens
            if isinstance(token, dict)
            and token.get("source_char_start") == 0
            and token.get("source_char_end") == 4
        ]
        self.assertGreater(len(first_segment_tokens), 1)
        self.assertTrue(
            all(
                token["commit_char_end"] == 0
                for token in first_segment_tokens[:-1]
            )
        )
        self.assertEqual(first_segment_tokens[-1]["commit_char_end"], 4)

        progress = expected["progress"]
        self.assertIsInstance(progress, list)
        assert isinstance(progress, list)
        committed_boundaries = {
            item["source_char_end"]
            for item in progress
            if isinstance(item, dict)
        }
        self.assertTrue({1, 2, 3}.isdisjoint(committed_boundaries))

    def test_eos_maps_to_source_end_in_chars_and_utf8(self) -> None:
        for case in self.cases:
            source_text = case["source_text"]
            expected = case["expected"]
            self.assertIsInstance(source_text, str)
            self.assertIsInstance(expected, dict)
            assert isinstance(source_text, str)
            assert isinstance(expected, dict)
            tokens = expected["tokens"]
            self.assertIsInstance(tokens, list)
            assert isinstance(tokens, list)
            eos = tokens[-1]
            self.assertIsInstance(eos, dict)
            assert isinstance(eos, dict)
            self.assertEqual(eos["kind"], "eos")
            self.assertEqual(eos["source_char_start"], len(source_text))
            self.assertEqual(eos["source_char_end"], len(source_text))
            self.assertEqual(eos["commit_char_end"], len(source_text))
            self.assertEqual(
                eos["source_utf8_end"], len(source_text.encode("utf-8"))
            )

    def test_machine_fixture_negative_cases_fail_closed(self) -> None:
        error_types: dict[str, type[FrontendError]] = {
            "SourceNormalizationError": SourceNormalizationError,
            "AmbiguousAlignmentError": AmbiguousAlignmentError,
        }
        for case in self.negative_cases:
            source_text = case["source_text"]
            error_name = case["expected_error"]
            expected_message = case["expected_message"]
            prefix_rows = case["prefix_frontend_rows"]
            self.assertIsInstance(source_text, str)
            self.assertIsInstance(error_name, str)
            self.assertIsInstance(expected_message, str)
            self.assertIsInstance(prefix_rows, list)
            assert isinstance(source_text, str)
            assert isinstance(error_name, str)
            assert isinstance(expected_message, str)
            assert isinstance(prefix_rows, list)
            frontend = LockedJapaneseFrontend(
                self.spec, FixtureRunner(prefix_rows)
            )
            with self.subTest(case=case["id"]):
                with self.assertRaisesRegex(
                    error_types[error_name], expected_message
                ):
                    frontend.prepare(source_text)

    def test_token_table_tampering_is_rejected(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        frontend_contract = fixture["frontend_contract"]
        self.assertIsInstance(frontend_contract, dict)
        assert isinstance(frontend_contract, dict)
        japanese = frontend_contract["japanese"]
        self.assertIsInstance(japanese, dict)
        assert isinstance(japanese, dict)
        token_table = japanese["token_table"]
        self.assertIsInstance(token_table, list)
        assert isinstance(token_table, list)
        row = token_table[1]
        self.assertIsInstance(row, dict)
        assert isinstance(row, dict)
        row["token"] = "tampered"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(
                json.dumps(fixture, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                FrontendError, "effective_local_id mismatch|SHA-256 mismatch"
            ):
                load_frontend_spec(LOCK, path)


if __name__ == "__main__":
    unittest.main()

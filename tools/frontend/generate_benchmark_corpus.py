#!/usr/bin/env python3
"""Generate the immutable Japanese acceptance corpus for mtt-runtime-benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from tools.frontend.japanese_frontend import (
    LockedJapaneseFrontend,
    tokenizer_identity_sha256,
)


SOURCE_SCHEMA = "magpie-tts-rt.benchmark-source.v1"
OUTPUT_SCHEMA = "magpie-tts-rt.benchmark-corpus.v1"
ACCEPTANCE_NORMAL_CASES = 108
MAXIMUM_SOURCE_BYTES = 4_096
MAXIMUM_SOURCE_FILE_BYTES = 4 * 1024 * 1024
MAXIMUM_OUTPUT_BYTES = 16 * 1024 * 1024
MAXIMUM_PREPARED_TOKENS = 512
UINT32_MAXIMUM = (1 << 32) - 1
INT32_MAXIMUM = (1 << 31) - 1
REQUIRED_CATEGORIES = frozenset(
    {
        "short",
        "filler",
        "conversation",
        "punctuation",
        "number_unit",
        "latin_abbreviation",
        "spatial",
        "robot_instruction",
        "long",
    }
)

type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


class CorpusSourceError(RuntimeError):
    """The tracked corpus source violates its exact schema."""


class PreparedUtteranceLike(Protocol):
    @property
    def global_token_ids(self) -> Sequence[int]:
        """Return the exact aggregate token IDs, including EOS."""


class JapaneseFrontendLike(Protocol):
    def prepare(self, source_text: str) -> PreparedUtteranceLike:
        """Prepare one source text without guessing or fallback."""


@dataclass(frozen=True)
class BenchmarkSourceCase:
    case_id: str
    category: str
    source_text: str
    random_seed: int


@dataclass(frozen=True)
class BenchmarkSource:
    corpus_id: str
    tokenizer_identity_sha256: str
    normal_cases: tuple[BenchmarkSourceCase, ...]
    cancel_case: BenchmarkSourceCase


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusSourceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise CorpusSourceError(f"non-finite JSON number is forbidden: {value}")


def _require_exact_keys(
    value: JsonValue,
    expected: frozenset[str],
    label: str,
) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CorpusSourceError(
            f"{label} keys mismatch: expected={sorted(expected)}, actual={actual}"
        )
    return value


def _require_string(
    value: JsonValue,
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise CorpusSourceError(f"{label} must be a non-empty string")
    return value


def _require_uint32(value: JsonValue, label: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT32_MAXIMUM:
        raise CorpusSourceError(f"{label} must be a uint32")
    return value


def _valid_identifier(value: str) -> bool:
    if not 1 <= len(value) <= 64 or not value[0].isalnum():
        return False
    return all(
        character.isascii()
        and (character.isalnum() or character in "_-")
        for character in value
    )


def _require_sha256(value: JsonValue, label: str) -> str:
    digest = _require_string(value, label)
    if (
        len(digest) != 64
        or digest == "0" * 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CorpusSourceError(f"{label} must be a nonzero lowercase SHA-256")
    return digest


def _contains_japanese(source_text: str) -> bool:
    return any(
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u9fff"
        for character in source_text
    )


def _parse_case(
    value: JsonValue,
    label: str,
    *,
    require_category: bool,
) -> BenchmarkSourceCase:
    expected = frozenset(
        {"case_id", "category", "source_text", "random_seed"}
        if require_category
        else {"case_id", "source_text", "random_seed"}
    )
    record = _require_exact_keys(value, expected, label)
    case_id = _require_string(record["case_id"], f"{label}.case_id")
    if not _valid_identifier(case_id):
        raise CorpusSourceError(f"{label}.case_id is not a valid identifier")
    source_text = _require_string(
        record["source_text"], f"{label}.source_text"
    )
    encoded = source_text.encode("utf-8")
    if (
        len(encoded) > MAXIMUM_SOURCE_BYTES
        or not _contains_japanese(source_text)
    ):
        raise CorpusSourceError(
            f"{label}.source_text must contain Japanese and fit 4096 UTF-8 bytes"
        )
    category = (
        _require_string(record["category"], f"{label}.category")
        if require_category
        else "cancel"
    )
    if require_category and category not in REQUIRED_CATEGORIES:
        raise CorpusSourceError(f"{label}.category is unsupported: {category}")
    return BenchmarkSourceCase(
        case_id=case_id,
        category=category,
        source_text=source_text,
        random_seed=_require_uint32(
            record["random_seed"], f"{label}.random_seed"
        ),
    )


def require_absolute_regular_file(path: Path, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size == 0
    ):
        raise CorpusSourceError(
            f"{label} must be an absolute non-symlink regular file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise CorpusSourceError(f"{label} must not traverse a symlink: {path}")
    return resolved


def require_new_output_path(path: Path) -> Path:
    if not path.is_absolute() or not path.name:
        raise CorpusSourceError("--output must be an absolute file path")
    if path.is_symlink():
        raise CorpusSourceError("--output must not be a symbolic link")
    if path.exists():
        raise CorpusSourceError(f"output already exists: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CorpusSourceError(
            "--output parent must be an existing non-symlink directory"
        )
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != parent:
        raise CorpusSourceError("--output parent must not traverse a symlink")
    return path


def load_benchmark_source(path: Path) -> BenchmarkSource:
    source_path = require_absolute_regular_file(path, "--source")
    if source_path.stat().st_size > MAXIMUM_SOURCE_FILE_BYTES:
        raise CorpusSourceError("benchmark source exceeds the 4 MiB limit")
    try:
        document = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusSourceError(
            f"cannot read strict benchmark source JSON: {error}"
        ) from error

    root = _require_exact_keys(
        document,
        frozenset(
            {
                "schema_version",
                "corpus_id",
                "tokenizer_identity_sha256",
                "normal_case_count",
                "cases",
                "cancel_case",
            }
        ),
        "source",
    )
    if root["schema_version"] != SOURCE_SCHEMA:
        raise CorpusSourceError("unsupported benchmark source schema")
    corpus_id = _require_string(root["corpus_id"], "source.corpus_id")
    if not _valid_identifier(corpus_id):
        raise CorpusSourceError("source.corpus_id is not a valid identifier")
    tokenizer_identity = _require_sha256(
        root["tokenizer_identity_sha256"],
        "source.tokenizer_identity_sha256",
    )
    declared_count = root["normal_case_count"]
    if (
        type(declared_count) is not int
        or declared_count != ACCEPTANCE_NORMAL_CASES
    ):
        raise CorpusSourceError(
            "source.normal_case_count must be exactly 108"
        )
    raw_cases = root["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != declared_count:
        raise CorpusSourceError(
            "source.cases length differs from normal_case_count"
        )
    normal_cases = tuple(
        _parse_case(value, f"source.cases[{index}]", require_category=True)
        for index, value in enumerate(raw_cases)
    )
    cancel_case = _parse_case(
        root["cancel_case"], "source.cancel_case", require_category=False
    )

    all_cases = (*normal_cases, cancel_case)
    case_ids = [case.case_id for case in all_cases]
    if len(set(case_ids)) != len(case_ids):
        raise CorpusSourceError("benchmark source contains duplicate case_id")
    seeds = [case.random_seed for case in all_cases]
    if len(set(seeds)) != len(seeds):
        raise CorpusSourceError("benchmark source contains duplicate random_seed")
    source_texts = [case.source_text for case in all_cases]
    if len(set(source_texts)) != len(source_texts):
        raise CorpusSourceError("benchmark source contains duplicate source_text")
    actual_categories = {case.category for case in normal_cases}
    if actual_categories != REQUIRED_CATEGORIES:
        raise CorpusSourceError(
            "benchmark source category coverage mismatch: "
            f"expected={sorted(REQUIRED_CATEGORIES)}, "
            f"actual={sorted(actual_categories)}"
        )
    return BenchmarkSource(
        corpus_id=corpus_id,
        tokenizer_identity_sha256=tokenizer_identity,
        normal_cases=normal_cases,
        cancel_case=cancel_case,
    )


def _prepared_token_ids(
    frontend: JapaneseFrontendLike,
    source_case: BenchmarkSourceCase,
) -> list[int]:
    prepared = frontend.prepare(source_case.source_text)
    token_ids = list(prepared.global_token_ids)
    if not 1 <= len(token_ids) <= MAXIMUM_PREPARED_TOKENS:
        raise CorpusSourceError(
            f"{source_case.case_id}: prepared token count is outside [1, 512]"
        )
    for token_index, token_id in enumerate(token_ids):
        if (
            type(token_id) is not int
            or not 0 <= token_id <= INT32_MAXIMUM
        ):
            raise CorpusSourceError(
                f"{source_case.case_id}: token {token_index} is not INT32"
            )
    return token_ids


def _case_record(
    frontend: JapaneseFrontendLike,
    source_case: BenchmarkSourceCase,
    record_type: str,
) -> dict[str, JsonValue]:
    return {
        "record_type": record_type,
        "case_id": source_case.case_id,
        "source_text": source_case.source_text,
        "source_text_sha256": sha256_bytes(
            source_case.source_text.encode("utf-8")
        ),
        "prepared_token_ids": _prepared_token_ids(frontend, source_case),
        "random_seed": source_case.random_seed,
    }


def generate_corpus_bytes(
    source: BenchmarkSource,
    frontend: JapaneseFrontendLike,
    actual_tokenizer_identity_sha256: str,
) -> bytes:
    actual_identity = _require_sha256(
        actual_tokenizer_identity_sha256,
        "actual tokenizer identity",
    )
    if actual_identity != source.tokenizer_identity_sha256:
        raise CorpusSourceError(
            "frontend tokenizer identity differs from the tracked source pin"
        )
    records: list[dict[str, JsonValue]] = [
        {
            "record_type": "header",
            "schema_version": OUTPUT_SCHEMA,
            "corpus_id": source.corpus_id,
            "tokenizer_identity_sha256": actual_identity,
            "normal_case_count": len(source.normal_cases),
        }
    ]
    records.extend(
        _case_record(frontend, source_case, "case")
        for source_case in source.normal_cases
    )
    records.append(_case_record(frontend, source.cancel_case, "cancel_case"))
    payload = (
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for record in records
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAXIMUM_OUTPUT_BYTES:
        raise CorpusSourceError("generated corpus exceeds the 16 MiB limit")
    return payload


def publish_no_replace(path: Path, payload: bytes) -> None:
    destination = require_new_output_path(path)
    temporary: Path | None = None
    descriptor = -1
    target_linked = False
    try:
        for attempt in range(32):
            candidate = destination.parent / (
                f".{destination.name}.tmp.{os.getpid()}.{attempt}"
            )
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    0o644,
                )
            except FileExistsError:
                continue
            temporary = candidate
            break
        if descriptor < 0 or temporary is None:
            raise CorpusSourceError(
                "cannot allocate an exclusive output temporary"
            )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise CorpusSourceError("short write while publishing corpus")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise CorpusSourceError(
                f"output already exists: {destination}"
            ) from error
        target_linked = True
        temporary.unlink()
        temporary = None
        directory = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None and temporary.exists():
            temporary.unlink()
        if target_linked and destination.exists():
            destination.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--oracle-lock", type=Path, required=True)
    parser.add_argument("--frontend-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = require_absolute_regular_file(args.source, "--source")
    lock_path = require_absolute_regular_file(args.oracle_lock, "--oracle-lock")
    contract_path = require_absolute_regular_file(
        args.frontend_contract, "--frontend-contract"
    )
    output_path = require_new_output_path(args.output)
    source = load_benchmark_source(source_path)
    identity = tokenizer_identity_sha256(lock_path, contract_path)
    frontend = LockedJapaneseFrontend.from_files(lock_path, contract_path)
    payload = generate_corpus_bytes(source, frontend, identity)
    publish_no_replace(output_path, payload)
    print(f"corpus={output_path}")
    print(f"corpus_sha256={sha256_bytes(payload)}")
    print(f"normal_case_count={len(source.normal_cases)}")
    print(f"tokenizer_identity_sha256={identity}")
    print("status=generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

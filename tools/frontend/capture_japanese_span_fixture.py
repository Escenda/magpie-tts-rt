#!/usr/bin/env python3
"""Capture the machine-readable Japanese source-span golden fixture.

This command must run in the locked NeMo environment.  It compares the
standalone frontend against NeMo's JapaneseKatakanaAccentG2p and
JapanesePhonemeTokenizer before writing any fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

from japanese_frontend import (
    AmbiguousAlignmentError,
    LockedJapaneseFrontend,
    SourceNormalizationError,
)


POSITIVE_CASES = (
    ("locked_minimal", "こんにちは。"),
    ("hiragana_kanji_punctuation", "今日は晴れです。"),
    ("digits_date", "2026年7月30日"),
    ("decimal_unit", "3.14kg"),
    ("fullwidth_digits_unit", "１２３ｋｇ"),
    ("latin_nfkc", "ＡＳＰＡ Robot"),
    ("halfwidth_katakana_nfkc", "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ"),
)
SOURCE_FILES = (
    "nemo/collections/tts/g2p/models/ja_jp_ipa.py",
    "nemo/collections/common/tokenizers/text_to_speech/tts_tokenizers.py",
    "nemo/collections/common/tokenizers/text_to_speech/tokenizer_utils.py",
    "nemo/collections/common/tokenizers/text_to_speech/ipa_lexicon.py",
)
SOURCE_MODULES = {
    "nemo.collections.tts.g2p.models.ja_jp_ipa": SOURCE_FILES[0],
    (
        "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers"
    ): SOURCE_FILES[1],
    (
        "nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils"
    ): SOURCE_FILES[2],
    (
        "nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon"
    ): SOURCE_FILES[3],
}
NJD_FIELDS = (
    "string",
    "pron",
    "pos",
    "pos_group1",
    "chain_flag",
    "mora_size",
    "acc",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--frontend-contract", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compact_lower(source_text: str) -> str:
    return "".join(
        character for character in source_text if not character.isspace()
    ).lower()


def selected_rows(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    selected: list[dict[str, str | int]] = []
    for row_index, row in enumerate(rows):
        record: dict[str, str | int] = {}
        for field in NJD_FIELDS:
            value = row.get(field)
            if field in ("chain_flag", "mora_size", "acc"):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise RuntimeError(
                        f"OpenJTalk row {row_index}.{field} is not an integer"
                    )
            elif not isinstance(value, str):
                raise RuntimeError(
                    f"OpenJTalk row {row_index}.{field} is not a string"
                )
            record[field] = value
        selected.append(record)
    return selected


def fixture_rows_for_text(pyopenjtalk: object, text: str) -> list[dict[str, object]]:
    prefixes: list[dict[str, object]] = []
    for end in range(1, len(text) + 1):
        prefix = text[:end]
        rows = pyopenjtalk.run_frontend(prefix)
        prefixes.append(
            {
                "normalized_prefix": prefix,
                "rows": selected_rows(rows),
            }
        )
    return prefixes


def require_source_root(speech_root: Path, lock: dict[str, object]) -> dict[str, str]:
    root = speech_root.resolve(strict=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    oracle_source = lock.get("oracle_source")
    if not isinstance(oracle_source, dict):
        raise RuntimeError("oracle lock has no oracle_source object")
    expected_revision = oracle_source.get("base_revision")
    if revision != expected_revision:
        raise RuntimeError(
            f"NeMo source revision mismatch: expected {expected_revision}, "
            f"got {revision}"
        )
    digests: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise RuntimeError(f"source path escapes checkout: {relative}")
        digests[relative] = sha256_file(path)
    return digests


def require_imported_sources(speech_root: Path) -> dict[str, object]:
    root = speech_root.resolve(strict=True)
    modules: dict[str, object] = {}
    for module_name, relative_path in SOURCE_MODULES.items():
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"imported module has no file: {module_name}")
        actual_path = Path(module_file).resolve(strict=True)
        expected_path = (root / relative_path).resolve(strict=True)
        if actual_path != expected_path:
            raise RuntimeError(
                f"imported frontend source mismatch for {module_name}: "
                f"expected {expected_path}, got {actual_path}"
            )
        modules[module_name] = module
    return modules


def synthetic_backward_prefix_case() -> dict[str, object]:
    def word(string_value: str, pron: str) -> dict[str, str | int]:
        return {
            "string": string_value,
            "pron": pron,
            "pos": "名詞",
            "pos_group1": "一般",
            "chain_flag": 0,
            "mora_size": len(pron),
            "acc": 0,
        }

    return {
        "id": "synthetic_backward_certified_prefix",
        "source_text": "甲乙丙",
        "expected_error": "AmbiguousAlignmentError",
        "expected_message": (
            "certified OpenJTalk prefixes move token progress backward"
        ),
        "prefix_frontend_rows": [
            {"normalized_prefix": "甲", "rows": [word("甲", "アイ")]},
            {"normalized_prefix": "甲乙", "rows": [word("甲乙", "ア")]},
            {"normalized_prefix": "甲乙丙", "rows": [word("甲乙丙", "アイ")]},
        ],
    }


def main() -> int:
    args = parse_args()
    lock_path = args.lock.resolve(strict=True)
    contract_path = args.frontend_contract.resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    contract_root = json.loads(contract_path.read_text(encoding="utf-8"))
    if "frontend_contract" in contract_root:
        frontend_contract = contract_root["frontend_contract"]
    else:
        frontend_contract = contract_root
    source_digests = require_source_root(args.speech_root, lock)

    frontend = LockedJapaneseFrontend.from_files(lock_path, contract_path)
    pyopenjtalk = importlib.import_module("pyopenjtalk")
    source_modules = require_imported_sources(args.speech_root)
    g2p_module = source_modules[
        "nemo.collections.tts.g2p.models.ja_jp_ipa"
    ]
    tokenizer_module = source_modules[
        "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers"
    ]
    g2p = g2p_module.JapaneseKatakanaAccentG2p(
        ascii_letter_prefix="", ascii_letter_case="lower"
    )
    tokenizer = tokenizer_module.JapanesePhonemeTokenizer(
        g2p=g2p,
        punct=True,
        apostrophe=False,
        pad_with_space=True,
    )

    cases: list[dict[str, object]] = []
    for case_id, source_text in POSITIVE_CASES:
        normalized = compact_lower(source_text)
        prepared = frontend.prepare(source_text)
        oracle_g2p = g2p(normalized)
        oracle_local_ids = tokenizer.encode(normalized)
        oracle_global_ids = [
            frontend.spec.global_offset + token_id
            for token_id in oracle_local_ids
        ] + [frontend.spec.eos_token_id]
        if list(prepared.g2p_tokens) != oracle_g2p:
            raise RuntimeError(f"{case_id}: standalone G2P differs from NeMo")
        if list(prepared.global_token_ids) != oracle_global_ids:
            raise RuntimeError(
                f"{case_id}: standalone token IDs differ from NeMo"
            )
        cases.append(
            {
                "id": case_id,
                "source_text": source_text,
                "prefix_frontend_rows": fixture_rows_for_text(
                    pyopenjtalk, normalized
                ),
                "expected": prepared.to_json_record(),
            }
        )

    negative_cases: list[dict[str, object]] = []
    for case_id, source_text, error_type in (
        ("lowercase_scalar_expansion", "İです。", SourceNormalizationError),
        ("non_whitespace_becomes_silent", "㍑", AmbiguousAlignmentError),
        ("whitespace_only", " \t\n", SourceNormalizationError),
    ):
        try:
            frontend.prepare(source_text)
        except error_type as error:
            negative_cases.append(
                {
                    "id": case_id,
                    "source_text": source_text,
                    "expected_error": error_type.__name__,
                    "expected_message": str(error),
                    "prefix_frontend_rows": (
                        fixture_rows_for_text(
                            pyopenjtalk, compact_lower(source_text)
                        )
                        if compact_lower(source_text)
                        and error_type is not SourceNormalizationError
                        else []
                    ),
                }
            )
        else:
            raise RuntimeError(f"{case_id}: expected {error_type.__name__}")
    negative_cases.append(synthetic_backward_prefix_case())

    payload = {
        "schema_version": 1,
        "fixture_id": "magpie-v2607-japanese-source-spans-v1",
        "oracle": {
            "oracle_lock_sha256": sha256_file(lock_path),
            "speech_revision": lock["oracle_source"]["base_revision"],
            "source_files": source_digests,
            "pyopenjtalk_version": frontend.spec.pyopenjtalk.version,
            "token_table_sha256": frontend.spec.token_table_sha256,
        },
        "frontend_contract": frontend_contract,
        "cases": cases,
        "negative_cases": negative_cases,
    }
    output_parent = args.output.parent.resolve(strict=True)
    output = output_parent / args.output.name
    if output.is_symlink():
        raise RuntimeError(f"output must not be a symbolic link: {output}")
    with output.open("x", encoding="utf-8") as target:
        json.dump(
            payload,
            target,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        target.write("\n")
    print(f"Japanese span fixture written: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        KeyError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Japanese span fixture capture failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

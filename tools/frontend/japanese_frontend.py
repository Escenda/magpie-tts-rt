#!/usr/bin/env python3
"""Prepare locked Japanese MagpieTTS tokens with exact source lineage.

The runtime alignment controller reports an end-exclusive position in the
prepared token sequence.  This frontend is the only component that translates
that position back to the caller's UTF-8 text.  It therefore refuses to guess
when OpenJTalk normalization cannot be proven by prefix replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import string
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


TOKEN_TABLE_CANONICALIZATION = (
    "utf8_json_sorted_keys_compact_rows_in_local_id_order"
)
DICTIONARY_MANIFEST_CANONICALIZATION = (
    "utf8_json_sorted_keys_compact_rows_in_path_order"
)
JAPANESE_TOKENIZER_NAME = "japanese_phoneme"
JAPANESE_PUNCTUATION = frozenset(
    [
        "!",
        '"',
        "(",
        ")",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
        "[",
        "]",
        "{",
        "}",
        "«",
        "»",
        "•",
        "‥",
        "…",
        "‹",
        "›",
        "※",
        "◦",
        "、",
        "。",
        "〃",
        "〈",
        "〉",
        "《",
        "》",
        "「",
        "」",
        "『",
        "』",
        "【",
        "】",
        "〒",
        "〓",
        "〔",
        "〕",
        "〖",
        "〗",
        "〘",
        "〙",
        "〚",
        "〛",
        "〜",
        "〽",
        "・",
        "・・・",
        "ー",
        "﹅",
        "﹆",
        "！",
        "＊",
        "？",
        "｟",
        "｠",
    ]
)
ASCII_LOWER = frozenset(string.ascii_lowercase)
MORA_PATTERN = (
    r"[ア-ンヴ][ャュョァィゥェォヮ]?|"
    r"[ァィゥェォヵヶッャュョヮ]|ー"
)


class FrontendError(RuntimeError):
    """Base class for a fail-closed frontend error."""


class ContractError(FrontendError):
    """The runtime frontend assets do not match the accepted lock."""


class SourceNormalizationError(FrontendError):
    """Source preprocessing cannot retain a one-scalar lineage."""


class AmbiguousAlignmentError(FrontendError):
    """OpenJTalk prefix replay cannot prove an ordered source alignment."""


@dataclass(frozen=True)
class TokenTableRow:
    effective_local_id: int
    global_id: int
    local_id: int
    token: str


@dataclass(frozen=True)
class DictionaryFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DictionarySpec:
    directory_name: str
    manifest_canonicalization: str
    manifest_sha256: str
    files: tuple[DictionaryFile, ...]


@dataclass(frozen=True)
class PyOpenJTalkSpec:
    version: str
    dictionary: DictionarySpec


@dataclass(frozen=True)
class FrontendSpec:
    aggregate_vocabulary_size: int
    text_embedding_rows: int
    bos_token_id: int
    eos_token_id: int
    global_offset: int
    global_pad_token_id: int
    local_vocabulary_size: int
    long_vowel_mark_effective_local_token_id: int
    pyopenjtalk: PyOpenJTalkSpec
    token_table: tuple[TokenTableRow, ...]
    token_table_sha256: str
    golden_raw_text: str
    golden_g2p_tokens: tuple[str, ...]
    golden_local_token_ids: tuple[int, ...]
    golden_global_token_ids_with_eos: tuple[int, ...]

    @property
    def effective_local_ids(self) -> dict[str, int]:
        return {row.token: row.effective_local_id for row in self.token_table}


@dataclass(frozen=True)
class NjdWord:
    string: str
    pron: str
    pos: str
    pos_group1: str
    chain_flag: int
    mora_size: int
    acc: int


@dataclass(frozen=True)
class G2PToken:
    symbol: str
    kind: str
    unit_index: int


@dataclass(frozen=True)
class EncodedCoreToken:
    symbol: str
    kind: str
    unit_index: int
    local_token_id: int
    global_token_id: int


@dataclass(frozen=True)
class SourceSegment:
    """One proven monotonic prefix-replay step.

    The source interval and token interval are both half-open.  Every core
    token belongs to exactly one segment and the source intervals are
    contiguous.
    """

    source_char_start: int
    source_char_end: int
    source_utf8_start: int
    source_utf8_end: int
    core_token_start: int
    core_token_end: int


@dataclass(frozen=True)
class PreparedToken:
    """One token consumed by the runtime, including padding and EOS."""

    token_index: int
    symbol: str
    kind: str
    unit_index: int | None
    local_token_id: int | None
    global_token_id: int
    source_char_start: int
    source_char_end: int
    source_utf8_start: int
    source_utf8_end: int
    commit_char_end: int
    commit_utf8_end: int


@dataclass(frozen=True)
class SourceProgress:
    """Mapping for one end-exclusive prepared-token position."""

    committed_text_tokens: int
    source_char_end: int
    source_utf8_end: int


@dataclass(frozen=True)
class AlignmentProof:
    method: str
    prepared_char_count: int
    core_token_count: int
    certified_prefix_count: int
    segments: tuple[SourceSegment, ...]


@dataclass(frozen=True)
class PreparedUtterance:
    source_text: str
    normalized_text: str
    g2p_tokens: tuple[str, ...]
    global_token_ids: tuple[int, ...]
    tokens: tuple[PreparedToken, ...]
    progress: tuple[SourceProgress, ...]
    proof: AlignmentProof

    def source_progress(self, committed_text_tokens: int) -> SourceProgress:
        if committed_text_tokens < 0 or committed_text_tokens >= len(self.progress):
            raise IndexError(
                "committed_text_tokens must be within the prepared token sequence"
            )
        return self.progress[committed_text_tokens]

    def to_json_record(self) -> dict[str, str | int | list[object] | dict[str, object]]:
        return {
            "schema_version": 1,
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "g2p_tokens": list(self.g2p_tokens),
            "global_token_ids": list(self.global_token_ids),
            "tokens": [asdict(token) for token in self.tokens],
            "progress": [asdict(progress) for progress in self.progress],
            "proof": {
                "method": self.proof.method,
                "prepared_char_count": self.proof.prepared_char_count,
                "core_token_count": self.proof.core_token_count,
                "certified_prefix_count": self.proof.certified_prefix_count,
                "segments": [asdict(segment) for segment in self.proof.segments],
            },
        }


FrontendRows = Sequence[Mapping[str, str | int]]
FrontendRunner = Callable[[str], FrontendRows]
MAXIMUM_NJD_WORDS_PER_SEGMENT = 80
INDEPENDENT_SEGMENTATION_MODE = "independent_sentence_segments_v1"


@dataclass(frozen=True)
class PreparedSegment:
    """One independently synthesized segment with global source offsets."""

    segment_index: int
    source_char_start: int
    source_char_end: int
    source_utf8_start: int
    source_utf8_end: int
    prepared: PreparedUtterance


@dataclass(frozen=True)
class PreparedSegments:
    """Source-proven independent synthesis segments for one logical utterance.

    These segments do not preserve the upstream Magpie chunk-history state.
    """

    source_text: str
    segmentation_mode: str
    segments: tuple[PreparedSegment, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(
    value: object, label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise ContractError(f"{label} contains a non-string key")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a JSON array")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    return value


def _member(mapping: Mapping[str, object], key: str, label: str) -> object:
    if key not in mapping:
        raise ContractError(f"{label} is missing {key!r}")
    return mapping[key]


def _json_file(path: Path, label: str) -> Mapping[str, object]:
    if path.is_symlink():
        raise ContractError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {label}: {error}") from error
    return _require_mapping(value, label)


def _canonical_token_table_sha256(rows: Sequence[TokenTableRow]) -> str:
    payload = json.dumps(
        [asdict(row) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_token_table(
    value: object,
    *,
    global_offset: int,
    expected_size: int,
) -> tuple[TokenTableRow, ...]:
    raw_rows = _require_list(value, "frontend_contract.japanese.token_table")
    rows: list[TokenTableRow] = []
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, f"token_table[{index}]")
        parsed = TokenTableRow(
            effective_local_id=_require_int(
                _member(row, "effective_local_id", f"token_table[{index}]"),
                f"token_table[{index}].effective_local_id",
            ),
            global_id=_require_int(
                _member(row, "global_id", f"token_table[{index}]"),
                f"token_table[{index}].global_id",
            ),
            local_id=_require_int(
                _member(row, "local_id", f"token_table[{index}]"),
                f"token_table[{index}].local_id",
            ),
            token=_require_str(
                _member(row, "token", f"token_table[{index}]"),
                f"token_table[{index}].token",
            ),
        )
        if parsed.local_id != index:
            raise ContractError(
                f"token_table[{index}].local_id must equal its row position"
            )
        if parsed.global_id != global_offset + index:
            raise ContractError(
                f"token_table[{index}].global_id does not use the locked offset"
            )
        rows.append(parsed)
    if len(rows) != expected_size:
        raise ContractError(
            f"Japanese token table size mismatch: expected {expected_size}, "
            f"got {len(rows)}"
        )
    effective_ids = {row.token: row.local_id for row in rows}
    for index, row in enumerate(rows):
        expected_effective_id = effective_ids[row.token]
        if row.effective_local_id != expected_effective_id:
            raise ContractError(
                f"token_table[{index}].effective_local_id mismatch: "
                f"expected {expected_effective_id}, got {row.effective_local_id}"
            )
    return tuple(rows)


def load_frontend_spec(lock_path: Path, contract_path: Path) -> FrontendSpec:
    """Load and cross-check the accepted lock and a captured frontend contract."""

    lock_root = _json_file(lock_path, "oracle lock")
    frontend_lock = _require_mapping(
        _member(lock_root, "frontend", "oracle lock"), "oracle lock.frontend"
    )
    japanese_lock = _require_mapping(
        _member(frontend_lock, "japanese", "oracle lock.frontend"),
        "oracle lock.frontend.japanese",
    )

    contract_root = _json_file(contract_path, "frontend contract")
    if "frontend_contract" in contract_root:
        frontend_contract = _require_mapping(
            contract_root["frontend_contract"], "frontend_contract"
        )
    else:
        frontend_contract = contract_root
    japanese_contract = _require_mapping(
        _member(frontend_contract, "japanese", "frontend_contract"),
        "frontend_contract.japanese",
    )

    scalar_fields = (
        "aggregate_vocabulary_size",
        "text_embedding_rows",
        "bos_token_id",
        "eos_token_id",
        "text_token_dtype",
    )
    for field in scalar_fields:
        locked_value = _member(frontend_lock, field, "oracle lock.frontend")
        contract_value = _member(frontend_contract, field, "frontend_contract")
        if contract_value != locked_value:
            raise ContractError(
                f"frontend contract {field} mismatch: "
                f"expected {locked_value!r}, got {contract_value!r}"
            )
    locked_dependencies = _require_mapping(
        _member(frontend_lock, "dependencies", "oracle lock.frontend"),
        "oracle lock.frontend.dependencies",
    )
    contract_dependencies = _require_mapping(
        _member(frontend_contract, "dependencies", "frontend_contract"),
        "frontend_contract.dependencies",
    )
    required_runtime_dependencies = {
        "nemo",
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "omegaconf",
    }
    if set(contract_dependencies) != required_runtime_dependencies:
        raise ContractError(
            "frontend dependency contract has missing or extra runtime packages"
        )
    for dependency_name, contract_value in contract_dependencies.items():
        if locked_dependencies.get(dependency_name) != contract_value:
            raise ContractError(
                f"frontend dependency {dependency_name} does not match lock"
            )

    japanese_scalar_fields = (
        "tokenizer_name",
        "global_offset",
        "local_vocabulary_size",
        "global_pad_token_id",
        "token_table_canonicalization",
        "token_table_sha256",
    )
    for field in japanese_scalar_fields:
        locked_value = _member(japanese_lock, field, "oracle lock.frontend.japanese")
        contract_value = _member(
            japanese_contract, field, "frontend_contract.japanese"
        )
        if contract_value != locked_value:
            raise ContractError(
                f"Japanese frontend contract {field} mismatch: "
                f"expected {locked_value!r}, got {contract_value!r}"
            )
    locked_pyopenjtalk = _require_mapping(
        _member(japanese_lock, "pyopenjtalk", "oracle lock.frontend.japanese"),
        "oracle lock.frontend.japanese.pyopenjtalk",
    )
    contract_pyopenjtalk = _require_mapping(
        _member(
            japanese_contract,
            "pyopenjtalk",
            "frontend_contract.japanese",
        ),
        "frontend_contract.japanese.pyopenjtalk",
    )
    if contract_pyopenjtalk != locked_pyopenjtalk:
        raise ContractError("OpenJTalk frontend contract does not match lock")
    if (
        _require_str(
            _member(
                japanese_lock,
                "token_table_canonicalization",
                "oracle lock.frontend.japanese",
            ),
            "oracle lock.frontend.japanese.token_table_canonicalization",
        )
        != TOKEN_TABLE_CANONICALIZATION
    ):
        raise ContractError("unsupported Japanese token table canonicalization")
    tokenizer_name = _require_str(
        _member(japanese_lock, "tokenizer_name", "oracle lock.frontend.japanese"),
        "oracle lock.frontend.japanese.tokenizer_name",
    )
    if tokenizer_name != JAPANESE_TOKENIZER_NAME:
        raise ContractError(
            f"unsupported Japanese tokenizer: {tokenizer_name!r}"
        )

    global_offset = _require_int(
        _member(japanese_lock, "global_offset", "oracle lock.frontend.japanese"),
        "oracle lock.frontend.japanese.global_offset",
    )
    local_vocabulary_size = _require_int(
        _member(
            japanese_lock,
            "local_vocabulary_size",
            "oracle lock.frontend.japanese",
        ),
        "oracle lock.frontend.japanese.local_vocabulary_size",
    )
    token_table = _parse_token_table(
        _member(japanese_contract, "token_table", "frontend_contract.japanese"),
        global_offset=global_offset,
        expected_size=local_vocabulary_size,
    )
    token_table_sha256 = _canonical_token_table_sha256(token_table)
    expected_table_sha256 = _require_str(
        _member(
            japanese_lock, "token_table_sha256", "oracle lock.frontend.japanese"
        ),
        "oracle lock.frontend.japanese.token_table_sha256",
    )
    if token_table_sha256 != expected_table_sha256:
        raise ContractError(
            "Japanese token table SHA-256 mismatch: "
            f"expected {expected_table_sha256}, got {token_table_sha256}"
        )

    pyopenjtalk_lock = _require_mapping(
        _member(japanese_lock, "pyopenjtalk", "oracle lock.frontend.japanese"),
        "oracle lock.frontend.japanese.pyopenjtalk",
    )
    dictionary_lock = _require_mapping(
        _member(pyopenjtalk_lock, "dictionary", "pyopenjtalk lock"),
        "pyopenjtalk lock.dictionary",
    )
    dictionary_files: list[DictionaryFile] = []
    for index, raw_file in enumerate(
        _require_list(
            _member(dictionary_lock, "files", "pyopenjtalk lock.dictionary"),
            "pyopenjtalk lock.dictionary.files",
        )
    ):
        file_record = _require_mapping(
            raw_file, f"pyopenjtalk lock.dictionary.files[{index}]"
        )
        dictionary_files.append(
            DictionaryFile(
                path=_require_str(
                    _member(file_record, "path", f"dictionary file {index}"),
                    f"dictionary file {index}.path",
                ),
                size_bytes=_require_int(
                    _member(file_record, "size_bytes", f"dictionary file {index}"),
                    f"dictionary file {index}.size_bytes",
                ),
                sha256=_require_str(
                    _member(file_record, "sha256", f"dictionary file {index}"),
                    f"dictionary file {index}.sha256",
                ),
            )
        )
    dictionary = DictionarySpec(
        directory_name=_require_str(
            _member(
                dictionary_lock,
                "directory_name",
                "pyopenjtalk lock.dictionary",
            ),
            "pyopenjtalk lock.dictionary.directory_name",
        ),
        manifest_canonicalization=_require_str(
            _member(
                dictionary_lock,
                "manifest_canonicalization",
                "pyopenjtalk lock.dictionary",
            ),
            "pyopenjtalk lock.dictionary.manifest_canonicalization",
        ),
        manifest_sha256=_require_str(
            _member(
                dictionary_lock,
                "manifest_sha256",
                "pyopenjtalk lock.dictionary",
            ),
            "pyopenjtalk lock.dictionary.manifest_sha256",
        ),
        files=tuple(dictionary_files),
    )
    if dictionary.manifest_canonicalization != DICTIONARY_MANIFEST_CANONICALIZATION:
        raise ContractError("unsupported OpenJTalk dictionary canonicalization")

    golden = _require_mapping(
        _member(japanese_lock, "golden", "oracle lock.frontend.japanese"),
        "oracle lock.frontend.japanese.golden",
    )
    contract_golden = _require_mapping(
        _member(japanese_contract, "golden", "frontend_contract.japanese"),
        "frontend_contract.japanese.golden",
    )
    for field in (
        "raw_text",
        "g2p_tokens",
        "local_token_ids",
        "global_token_ids_with_eos",
    ):
        if _member(contract_golden, field, "frontend_contract.japanese.golden") != (
            _member(golden, field, "oracle lock.frontend.japanese.golden")
        ):
            raise ContractError(
                f"Japanese golden frontend contract {field} does not match lock"
            )

    def integer_tuple(value: object, label: str) -> tuple[int, ...]:
        return tuple(
            _require_int(item, f"{label}[{index}]")
            for index, item in enumerate(_require_list(value, label))
        )

    def string_tuple(value: object, label: str) -> tuple[str, ...]:
        return tuple(
            _require_str(item, f"{label}[{index}]")
            for index, item in enumerate(_require_list(value, label))
        )

    spec = FrontendSpec(
        aggregate_vocabulary_size=_require_int(
            _member(
                frontend_lock,
                "aggregate_vocabulary_size",
                "oracle lock.frontend",
            ),
            "oracle lock.frontend.aggregate_vocabulary_size",
        ),
        text_embedding_rows=_require_int(
            _member(
                frontend_lock,
                "text_embedding_rows",
                "oracle lock.frontend",
            ),
            "oracle lock.frontend.text_embedding_rows",
        ),
        bos_token_id=_require_int(
            _member(frontend_lock, "bos_token_id", "oracle lock.frontend"),
            "oracle lock.frontend.bos_token_id",
        ),
        eos_token_id=_require_int(
            _member(frontend_lock, "eos_token_id", "oracle lock.frontend"),
            "oracle lock.frontend.eos_token_id",
        ),
        global_offset=global_offset,
        global_pad_token_id=_require_int(
            _member(
                japanese_lock,
                "global_pad_token_id",
                "oracle lock.frontend.japanese",
            ),
            "oracle lock.frontend.japanese.global_pad_token_id",
        ),
        local_vocabulary_size=local_vocabulary_size,
        long_vowel_mark_effective_local_token_id=_require_int(
            _member(
                japanese_lock,
                "long_vowel_mark_effective_local_token_id",
                "oracle lock.frontend.japanese",
            ),
            "oracle lock.frontend.japanese.long_vowel_mark_effective_local_token_id",
        ),
        pyopenjtalk=PyOpenJTalkSpec(
            version=_require_str(
                _member(pyopenjtalk_lock, "version", "pyopenjtalk lock"),
                "pyopenjtalk lock.version",
            ),
            dictionary=dictionary,
        ),
        token_table=token_table,
        token_table_sha256=token_table_sha256,
        golden_raw_text=_require_str(
            _member(golden, "raw_text", "Japanese golden"),
            "Japanese golden.raw_text",
        ),
        golden_g2p_tokens=string_tuple(
            _member(golden, "g2p_tokens", "Japanese golden"),
            "Japanese golden.g2p_tokens",
        ),
        golden_local_token_ids=integer_tuple(
            _member(golden, "local_token_ids", "Japanese golden"),
            "Japanese golden.local_token_ids",
        ),
        golden_global_token_ids_with_eos=integer_tuple(
            _member(golden, "global_token_ids_with_eos", "Japanese golden"),
            "Japanese golden.global_token_ids_with_eos",
        ),
    )
    effective_ids = spec.effective_local_ids
    if effective_ids.get("ー") != spec.long_vowel_mark_effective_local_token_id:
        raise ContractError("locked long-vowel token resolution does not match table")
    pad_local_id = spec.global_pad_token_id - spec.global_offset
    if effective_ids.get("<pad>") != pad_local_id:
        raise ContractError("locked Japanese pad token does not match table")
    if effective_ids.get(" ") != 0:
        raise ContractError("locked Japanese boundary-space token must be local ID 0")
    return spec


def tokenizer_identity_sha256(lock_path: Path, contract_path: Path) -> str:
    """Return the bundle identity for the fully validated frontend assets."""

    spec = load_frontend_spec(lock_path, contract_path)
    contract_root = _json_file(contract_path, "frontend contract")
    frontend_contract = (
        _require_mapping(contract_root["frontend_contract"], "frontend_contract")
        if "frontend_contract" in contract_root
        else contract_root
    )
    canonical_frontend = json.dumps(
        frontend_contract,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    projection = {
        "kind": JAPANESE_TOKENIZER_NAME,
        "vocabulary_size": spec.aggregate_vocabulary_size,
        "frontend_contract_sha256": hashlib.sha256(canonical_frontend).hexdigest(),
        "vocabulary_sha256": spec.token_table_sha256,
        "special_tokens": {
            "bos_token_id": spec.bos_token_id,
            "eos_token_id": spec.eos_token_id,
            "japanese_global_pad_token_id": spec.global_pad_token_id,
        },
    }
    canonical_projection = json.dumps(
        projection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_projection).hexdigest()


def _verify_dictionary(module: object, spec: DictionarySpec) -> None:
    if not hasattr(module, "OPEN_JTALK_DICT_DIR"):
        raise ContractError("pyopenjtalk has no OPEN_JTALK_DICT_DIR")
    unresolved = Path(os.fsdecode(module.OPEN_JTALK_DICT_DIR))
    if unresolved.is_symlink():
        raise ContractError(
            f"OpenJTalk dictionary must not be a symbolic link: {unresolved}"
        )
    root = unresolved.resolve(strict=True)
    if not root.is_dir():
        raise ContractError(f"OpenJTalk dictionary is not a directory: {root}")
    if root.name != spec.directory_name:
        raise ContractError(
            f"OpenJTalk dictionary directory mismatch: expected "
            f"{spec.directory_name!r}, got {root.name!r}"
        )
    expected_names = {record.path for record in spec.files}
    actual_names = {entry.name for entry in root.iterdir()}
    if actual_names != expected_names:
        raise ContractError(
            "OpenJTalk dictionary entries mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for record in spec.files:
        if (
            not record.path
            or Path(record.path).name != record.path
            or "/" in record.path
            or "\\" in record.path
        ):
            raise ContractError(
                f"unsafe OpenJTalk dictionary path: {record.path!r}"
            )
        path = root / record.path
        if path.is_symlink():
            raise ContractError(
                f"OpenJTalk dictionary file must not be a symbolic link: "
                f"{record.path}"
            )
        resolved = path.resolve(strict=True)
        if resolved.stat().st_size != record.size_bytes:
            raise ContractError(
                f"OpenJTalk dictionary size mismatch for {record.path}"
            )
        digest = _sha256_file(resolved)
        if digest != record.sha256:
            raise ContractError(
                f"OpenJTalk dictionary SHA-256 mismatch for {record.path}: "
                f"expected {record.sha256}, got {digest}"
            )


def _load_locked_runner(spec: FrontendSpec) -> FrontendRunner:
    try:
        version = importlib.metadata.version("pyopenjtalk")
        module = importlib.import_module("pyopenjtalk")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise ContractError("locked pyopenjtalk is not installed") from error
    if version != spec.pyopenjtalk.version:
        raise ContractError(
            f"pyopenjtalk version mismatch: expected {spec.pyopenjtalk.version}, "
            f"got {version}"
        )
    _verify_dictionary(module, spec.pyopenjtalk.dictionary)
    run_frontend = getattr(module, "run_frontend", None)
    if not callable(run_frontend):
        raise ContractError("pyopenjtalk.run_frontend is not callable")
    return run_frontend


def _njd_words(rows: FrontendRows) -> tuple[NjdWord, ...]:
    words: list[NjdWord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FrontendError(f"OpenJTalk row {index} is not a mapping")

        def text_field(name: str) -> str:
            value = row.get(name)
            if not isinstance(value, str):
                raise FrontendError(
                    f"OpenJTalk row {index}.{name} is not a string"
                )
            return value

        def integer_field(name: str) -> int:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise FrontendError(
                    f"OpenJTalk row {index}.{name} is not an integer"
                )
            return value

        words.append(
            NjdWord(
                string=text_field("string"),
                pron=text_field("pron"),
                pos=text_field("pos"),
                pos_group1=text_field("pos_group1"),
                chain_flag=integer_field("chain_flag"),
                mora_size=integer_field("mora_size"),
                acc=integer_field("acc"),
            )
        )
    return tuple(words)


def _split_moras(katakana: str) -> tuple[str, ...]:
    import re

    return tuple(re.findall(MORA_PATTERN, katakana))


def _pitch_pattern(acc: int, total_mora: int) -> tuple[int, ...]:
    if total_mora == 0:
        return ()
    if acc == 0:
        return (0, *(1 for _ in range(total_mora - 1)))
    if acc == 1:
        return (1, *(0 for _ in range(total_mora - 1)))
    if acc >= total_mora:
        return (0, *(1 for _ in range(total_mora - 1)))
    return (
        0,
        *(1 for _ in range(acc - 1)),
        *(0 for _ in range(total_mora - acc)),
    )


def _g2p_tokens(rows: FrontendRows) -> tuple[G2PToken, ...]:
    words = _njd_words(rows)
    output: list[G2PToken] = []
    current_chain: list[NjdWord] = []
    next_unit = 0

    def append_unit(symbols: Sequence[tuple[str, str]]) -> None:
        nonlocal next_unit
        for symbol, kind in symbols:
            output.append(
                G2PToken(symbol=symbol, kind=kind, unit_index=next_unit)
            )
        next_unit += 1

    def flush_chain() -> None:
        nonlocal current_chain
        if not current_chain:
            return
        chain_starter = next(
            (word for word in current_chain if word.chain_flag != 1),
            current_chain[0],
        )
        moras = tuple(
            mora
            for word in current_chain
            for mora in _split_moras(word.pron)
        )
        pitches = _pitch_pattern(chain_starter.acc, len(moras))
        for mora, pitch in zip(moras, pitches, strict=True):
            append_unit(
                [(str(pitch), "pitch")]
                + [(character, "mora") for character in mora]
            )
        current_chain = []

    for index, word in enumerate(words):
        normalized_string = unicodedata.normalize("NFKC", word.string)
        if (
            normalized_string
            and all(character in ASCII_LOWER for character in normalized_string)
            and word.pos == "フィラー"
        ):
            flush_chain()
            for character in normalized_string:
                append_unit([(character, "ascii")])
            continue

        if word.pos in ("記号", "補助記号") and word.pos_group1 != "アルファベット":
            flush_chain()
            if normalized_string.isspace():
                append_unit([(" ", "space")])
            elif normalized_string in JAPANESE_PUNCTUATION:
                append_unit([(normalized_string, "punctuation")])
            else:
                append_unit([(" ", "normalized_space")])
            continue

        if not word.pron or word.mora_size == 0:
            if normalized_string and not normalized_string.isspace():
                append_unit([(" ", "normalized_space")])
            continue

        current_chain.append(word)
        next_has_chain = (
            index + 1 < len(words) and words[index + 1].chain_flag == 1
        )
        if not next_has_chain:
            flush_chain()
    flush_chain()
    return tuple(output)


def _tokenize_core(
    g2p_tokens: Sequence[G2PToken], spec: FrontendSpec
) -> tuple[EncodedCoreToken, ...]:
    effective_ids = spec.effective_local_ids
    retained: list[G2PToken] = []
    for token in g2p_tokens:
        symbol = token.symbol
        if symbol == " ":
            if retained and retained[-1].symbol != " ":
                retained.append(token)
        elif symbol in effective_ids:
            retained.append(token)
    while retained and retained[-1].symbol == " ":
        retained.pop()
    return tuple(
        EncodedCoreToken(
            symbol=token.symbol,
            kind=token.kind,
            unit_index=token.unit_index,
            local_token_id=effective_ids[token.symbol],
            global_token_id=spec.global_offset + effective_ids[token.symbol],
        )
        for token in retained
    )


def _unit_end_positions(tokens: Sequence[EncodedCoreToken]) -> frozenset[int]:
    ends: set[int] = {0}
    for index, token in enumerate(tokens, start=1):
        if index == len(tokens) or tokens[index].unit_index != token.unit_index:
            ends.add(index)
    return frozenset(ends)


@dataclass(frozen=True)
class _SourceProjection:
    normalized_text: str
    prepared_source_char_ends: tuple[int, ...]
    utf8_boundaries: tuple[int, ...]
    last_non_whitespace_source_end: int


def _project_source(source_text: str) -> _SourceProjection:
    compact_characters: list[str] = []
    source_char_ends: list[int] = []
    for index, character in enumerate(source_text):
        if character.isspace():
            continue
        compact_characters.append(character)
        source_char_ends.append(index + 1)
    compact = "".join(compact_characters)
    normalized = compact.lower()
    independently_lowered = [character.lower() for character in compact_characters]
    if any(len(value) != 1 for value in independently_lowered):
        raise SourceNormalizationError(
            "lowercase preprocessing expands a source Unicode scalar"
        )
    if normalized != "".join(independently_lowered):
        raise SourceNormalizationError(
            "context-sensitive lowercase preprocessing has no one-scalar lineage"
        )
    if not normalized:
        raise SourceNormalizationError(
            "source text contains no non-whitespace Unicode scalar"
        )
    utf8_boundaries = [0]
    for character in source_text:
        utf8_boundaries.append(
            utf8_boundaries[-1] + len(character.encode("utf-8"))
        )
    return _SourceProjection(
        normalized_text=normalized,
        prepared_source_char_ends=tuple(source_char_ends),
        utf8_boundaries=tuple(utf8_boundaries),
        last_non_whitespace_source_end=source_char_ends[-1],
    )


class LockedJapaneseFrontend:
    """The accepted Japanese v2607 frontend plus source-span proof."""

    def __init__(self, spec: FrontendSpec, runner: FrontendRunner):
        self._spec = spec
        self._runner = runner

    @classmethod
    def from_files(
        cls, lock_path: Path, frontend_contract_path: Path
    ) -> "LockedJapaneseFrontend":
        spec = load_frontend_spec(lock_path, frontend_contract_path)
        frontend = cls(spec, _load_locked_runner(spec))
        frontend._verify_golden()
        return frontend

    @property
    def spec(self) -> FrontendSpec:
        return self._spec

    def _core(self, normalized_text: str) -> tuple[
        tuple[G2PToken, ...], tuple[EncodedCoreToken, ...]
    ]:
        rows = self._runner(normalized_text)
        g2p = _g2p_tokens(rows)
        return g2p, _tokenize_core(g2p, self._spec)

    def _njd_word_count(self, source_text: str) -> int:
        projection = _project_source(source_text)
        return len(_njd_words(self._runner(projection.normalized_text)))

    def _verify_golden(self) -> None:
        normalized = self._spec.golden_raw_text.lower()
        g2p, core = self._core(normalized)
        g2p_symbols = tuple(token.symbol for token in g2p)
        if g2p_symbols != self._spec.golden_g2p_tokens:
            raise ContractError(
                "locked Japanese golden G2P output does not match runtime"
            )
        boundary_space_local = self._spec.effective_local_ids[" "]
        local_ids = (
            boundary_space_local,
            *(token.local_token_id for token in core),
            boundary_space_local,
        )
        if local_ids != self._spec.golden_local_token_ids:
            raise ContractError(
                "locked Japanese golden local IDs do not match runtime"
            )
        global_ids = (
            self._spec.global_offset + boundary_space_local,
            *(token.global_token_id for token in core),
            self._spec.global_offset + boundary_space_local,
            self._spec.eos_token_id,
        )
        if global_ids != self._spec.golden_global_token_ids_with_eos:
            raise ContractError(
                "locked Japanese golden global IDs do not match runtime"
            )

    def prepare(self, source_text: str) -> PreparedUtterance:
        if not isinstance(source_text, str):
            raise TypeError("source_text must be str")
        projection = _project_source(source_text)
        full_g2p, full_core = self._core(projection.normalized_text)
        if not full_core:
            raise AmbiguousAlignmentError(
                "non-whitespace source produced no prepared speech token"
            )
        full_identity = tuple(
            (token.symbol, token.local_token_id) for token in full_core
        )
        unit_ends = _unit_end_positions(full_core)

        certified: list[tuple[int, int]] = [(0, 0)]
        maximum_certified_token_end = 0
        for prepared_end in range(1, len(projection.normalized_text) + 1):
            _, prefix_core = self._core(
                projection.normalized_text[:prepared_end]
            )
            prefix_identity = tuple(
                (token.symbol, token.local_token_id) for token in prefix_core
            )
            token_end = len(prefix_identity)
            if (
                token_end <= len(full_identity)
                and prefix_identity == full_identity[:token_end]
                and token_end in unit_ends
            ):
                if token_end < maximum_certified_token_end:
                    raise AmbiguousAlignmentError(
                        "certified OpenJTalk prefixes move token progress backward"
                    )
                certified.append((prepared_end, token_end))
                maximum_certified_token_end = token_end
        if certified[-1] != (
            len(projection.normalized_text),
            len(full_core),
        ):
            raise AmbiguousAlignmentError(
                "full OpenJTalk output is not a certified terminal prefix"
            )

        segments: list[SourceSegment] = []
        source_start = 0
        core_token_start = 0
        for prepared_end, core_token_end in certified[1:]:
            if core_token_end == core_token_start:
                continue
            if core_token_end < core_token_start:
                raise AmbiguousAlignmentError(
                    "OpenJTalk prefix alignment is not monotonic"
                )
            source_end = projection.prepared_source_char_ends[prepared_end - 1]
            if source_end <= source_start:
                raise AmbiguousAlignmentError(
                    "OpenJTalk token growth has no forward source interval"
                )
            segments.append(
                SourceSegment(
                    source_char_start=source_start,
                    source_char_end=source_end,
                    source_utf8_start=projection.utf8_boundaries[source_start],
                    source_utf8_end=projection.utf8_boundaries[source_end],
                    core_token_start=core_token_start,
                    core_token_end=core_token_end,
                )
            )
            source_start = source_end
            core_token_start = core_token_end
        if core_token_start != len(full_core):
            raise AmbiguousAlignmentError(
                "source alignment does not cover every prepared core token"
            )
        if source_start != projection.last_non_whitespace_source_end:
            raise AmbiguousAlignmentError(
                "non-whitespace source suffix has no spoken token"
            )

        source_length = len(source_text)
        source_utf8_length = projection.utf8_boundaries[-1]
        prepared_tokens: list[PreparedToken] = []

        def append_token(
            *,
            symbol: str,
            kind: str,
            unit_index: int | None,
            local_token_id: int | None,
            global_token_id: int,
            source_char_start: int,
            source_char_end: int,
            commit_char_end: int,
        ) -> None:
            prepared_tokens.append(
                PreparedToken(
                    token_index=len(prepared_tokens),
                    symbol=symbol,
                    kind=kind,
                    unit_index=unit_index,
                    local_token_id=local_token_id,
                    global_token_id=global_token_id,
                    source_char_start=source_char_start,
                    source_char_end=source_char_end,
                    source_utf8_start=projection.utf8_boundaries[
                        source_char_start
                    ],
                    source_utf8_end=projection.utf8_boundaries[source_char_end],
                    commit_char_end=commit_char_end,
                    commit_utf8_end=projection.utf8_boundaries[commit_char_end],
                )
            )

        boundary_space_local_id = self._spec.effective_local_ids[" "]
        boundary_space_global_id = (
            self._spec.global_offset + boundary_space_local_id
        )
        append_token(
            symbol=" ",
            kind="boundary_space",
            unit_index=None,
            local_token_id=boundary_space_local_id,
            global_token_id=boundary_space_global_id,
            source_char_start=0,
            source_char_end=0,
            commit_char_end=0,
        )
        for segment in segments:
            for core_index in range(
                segment.core_token_start, segment.core_token_end
            ):
                core_token = full_core[core_index]
                final_in_segment = core_index + 1 == segment.core_token_end
                append_token(
                    symbol=core_token.symbol,
                    kind=core_token.kind,
                    unit_index=core_token.unit_index,
                    local_token_id=core_token.local_token_id,
                    global_token_id=core_token.global_token_id,
                    source_char_start=segment.source_char_start,
                    source_char_end=segment.source_char_end,
                    commit_char_end=(
                        segment.source_char_end
                        if final_in_segment
                        else segment.source_char_start
                    ),
                )
        append_token(
            symbol=" ",
            kind="boundary_space",
            unit_index=None,
            local_token_id=boundary_space_local_id,
            global_token_id=boundary_space_global_id,
            source_char_start=source_start,
            source_char_end=source_length,
            commit_char_end=source_length,
        )
        append_token(
            symbol="<eos>",
            kind="eos",
            unit_index=None,
            local_token_id=None,
            global_token_id=self._spec.eos_token_id,
            source_char_start=source_length,
            source_char_end=source_length,
            commit_char_end=source_length,
        )

        progress = [SourceProgress(0, 0, 0)]
        for token in prepared_tokens:
            progress.append(
                SourceProgress(
                    committed_text_tokens=token.token_index + 1,
                    source_char_end=token.commit_char_end,
                    source_utf8_end=token.commit_utf8_end,
                )
            )
        previous_char = 0
        previous_utf8 = 0
        for item in progress:
            if (
                item.source_char_end < previous_char
                or item.source_utf8_end < previous_utf8
            ):
                raise AmbiguousAlignmentError(
                    "prepared token progress is not monotonic"
                )
            previous_char = item.source_char_end
            previous_utf8 = item.source_utf8_end
        if progress[-1] != SourceProgress(
            committed_text_tokens=len(prepared_tokens),
            source_char_end=source_length,
            source_utf8_end=source_utf8_length,
        ):
            raise AmbiguousAlignmentError(
                "EOS does not map to the exact source end"
            )

        return PreparedUtterance(
            source_text=source_text,
            normalized_text=projection.normalized_text,
            g2p_tokens=tuple(token.symbol for token in full_g2p),
            global_token_ids=tuple(
                token.global_token_id for token in prepared_tokens
            ),
            tokens=tuple(prepared_tokens),
            progress=tuple(progress),
            proof=AlignmentProof(
                method="locked_openjtalk_complete_prefix_replay_v1",
                prepared_char_count=len(projection.normalized_text),
                core_token_count=len(full_core),
                certified_prefix_count=len(certified),
                segments=tuple(segments),
            ),
        )

    def prepare_segments(self, source_text: str) -> PreparedSegments:
        """Split one logical utterance only at certified source boundaries."""

        prepared = self.prepare(source_text)
        ranges = _certified_segment_ranges(
            prepared,
            lambda start, end: self._njd_word_count(source_text[start:end]),
        )
        utf8_boundaries = [0]
        for character in source_text:
            utf8_boundaries.append(
                utf8_boundaries[-1] + len(character.encode("utf-8"))
            )
        segments: list[PreparedSegment] = []
        for segment_index, (source_start, source_end) in enumerate(ranges):
            segment_source = source_text[source_start:source_end]
            segment_prepared = (
                prepared
                if source_start == 0 and source_end == len(source_text)
                else self.prepare(segment_source)
            )
            if self._njd_word_count(segment_source) > MAXIMUM_NJD_WORDS_PER_SEGMENT:
                raise AmbiguousAlignmentError(
                    "certified source segment exceeds the 80-NJD-word runtime limit"
                )
            segments.append(
                PreparedSegment(
                    segment_index=segment_index,
                    source_char_start=source_start,
                    source_char_end=source_end,
                    source_utf8_start=utf8_boundaries[source_start],
                    source_utf8_end=utf8_boundaries[source_end],
                    prepared=segment_prepared,
                )
            )
        return PreparedSegments(
            source_text=source_text,
            segmentation_mode=INDEPENDENT_SEGMENTATION_MODE,
            segments=tuple(segments),
        )


def _certified_segment_ranges(
    prepared: PreparedUtterance,
    njd_word_count: Callable[[int, int], int],
) -> tuple[tuple[int, int], ...]:
    """Greedily select the longest <=80-word certified source interval."""

    source_length = len(prepared.source_text)
    sentence_ends = {
        byte_index + len(character)
        for byte_index, character in enumerate(prepared.source_text)
        if character in {"。", "！", "？", "\n"}
    }
    certified_ends = sorted(
        {
            segment.source_char_end
            for segment in prepared.proof.segments
            if 0 < segment.source_char_end < source_length
            and segment.source_char_end in sentence_ends
        }
        | {source_length}
    )
    ranges: list[tuple[int, int]] = []
    source_start = 0
    while source_start < source_length:
        selected_end: int | None = None
        for source_end in certified_ends:
            if source_end <= source_start:
                continue
            count = njd_word_count(source_start, source_end)
            if count <= 0:
                raise AmbiguousAlignmentError(
                    "certified source interval produced no OpenJTalk NJD word"
                )
            if count <= MAXIMUM_NJD_WORDS_PER_SEGMENT:
                selected_end = source_end
        if selected_end is None:
            raise AmbiguousAlignmentError(
                "no certified sentence boundary satisfies the 80-NJD-word runtime limit"
            )
        ranges.append((source_start, selected_end))
        source_start = selected_end
    if not ranges or ranges[-1][1] != source_length:
        raise AmbiguousAlignmentError(
            "certified long-text segmentation did not cover the exact source"
        )
    return tuple(ranges)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare locked Japanese MagpieTTS tokens and exact UTF-8/source "
            "character spans"
        )
    )
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--frontend-contract", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def _read_cli_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.text_file.is_symlink():
        raise FrontendError(
            f"text file must not be a symbolic link: {args.text_file}"
        )
    path = args.text_file.resolve(strict=True)
    return path.read_text(encoding="utf-8")


def main() -> int:
    args = _parse_args()
    frontend = LockedJapaneseFrontend.from_files(
        args.lock, args.frontend_contract
    )
    prepared = frontend.prepare(_read_cli_text(args))
    indent = 2 if args.pretty else None
    separators = None if args.pretty else (",", ":")
    payload = (
        json.dumps(
            prepared.to_json_record(),
            ensure_ascii=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output is None:
        sys.stdout.write(payload)
    else:
        resolved_parent = args.output.parent.resolve(strict=True)
        output = resolved_parent / args.output.name
        if output.is_symlink():
            raise FrontendError(f"output must not be a symbolic link: {output}")
        with output.open("x", encoding="utf-8") as target:
            target.write(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FrontendError, OSError, UnicodeError) as error:
        print(f"Japanese frontend failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

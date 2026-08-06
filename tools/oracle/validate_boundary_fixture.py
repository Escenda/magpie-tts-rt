#!/usr/bin/env python3
"""Validate every byte and structural invariant of an oracle boundary fixture.

The validator never repairs, ignores, or substitutes fixture content. A
manifest inconsistency, unsafe path, unlisted file, or lock mismatch is fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list[JsonValue]
    | dict[str, JsonValue]
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_PATTERN = re.compile(r"^[a-z0-9._-]+$")
DTYPE_WIDTH_BYTES = {
    "bf16": 2,
    "fp32": 4,
    "fp16": 2,
    "int64": 8,
    "int32": 4,
    "uint8": 1,
    "bool": 1,
}
MANIFEST_FILE = "manifest.json"
MANIFEST_CHECKSUM_FILE = "manifest.json.sha256"


@dataclass(frozen=True)
class TensorRecord:
    name: str
    path: str
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    sha256: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string")
    return value


def require_integer(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_boolean(value: JsonValue, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} must be a boolean")
    return value


def require_sha256(value: JsonValue, label: str) -> str:
    digest = require_string(value, label)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def require_safe_relative_path(value: JsonValue, label: str) -> str:
    relative = require_string(value, label)
    if not relative or "\\" in relative or "\0" in relative:
        raise RuntimeError(f"{label} is not a safe POSIX relative path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"{label} is not a safe POSIX relative path: {relative!r}")
    if path.as_posix() != relative:
        raise RuntimeError(f"{label} is not normalized: {relative!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in relative):
        raise RuntimeError(f"{label} contains a control character")
    return relative


def require_exact_keys(
    value: dict[str, JsonValue],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def reject_duplicate_object_pairs(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def parse_tensor_record(value: JsonValue, index: int) -> TensorRecord:
    label = f"tensors[{index}]"
    record = require_mapping(value, label)
    require_exact_keys(
        record,
        frozenset(("name", "path", "dtype", "shape", "size_bytes", "sha256")),
        label,
    )
    name = require_string(record["name"], f"{label}.name")
    if SAFE_NAME_PATTERN.fullmatch(name) is None:
        raise RuntimeError(f"{label}.name is unsafe: {name!r}")
    relative_path = require_safe_relative_path(record["path"], f"{label}.path")
    expected_path = f"tensors/{name}.bin"
    if relative_path != expected_path:
        raise RuntimeError(
            f"{label}.path must be derived from its name: "
            f"expected {expected_path!r}, got {relative_path!r}"
        )
    dtype = require_string(record["dtype"], f"{label}.dtype")
    width = DTYPE_WIDTH_BYTES.get(dtype)
    if width is None:
        raise RuntimeError(f"{label}.dtype is unsupported: {dtype!r}")
    dimensions = require_list(record["shape"], f"{label}.shape")
    shape: list[int] = []
    element_count = 1
    for dimension_index, dimension_value in enumerate(dimensions):
        dimension = require_integer(
            dimension_value,
            f"{label}.shape[{dimension_index}]",
        )
        if dimension < 0:
            raise RuntimeError(
                f"{label}.shape[{dimension_index}] must be non-negative"
            )
        shape.append(dimension)
        element_count *= dimension
    size_bytes = require_integer(record["size_bytes"], f"{label}.size_bytes")
    if size_bytes < 0:
        raise RuntimeError(f"{label}.size_bytes must be non-negative")
    expected_size = element_count * width
    if size_bytes != expected_size:
        raise RuntimeError(
            f"{label} dtype/shape byte count mismatch: "
            f"expected {expected_size}, got {size_bytes}"
        )
    return TensorRecord(
        name=name,
        path=relative_path,
        dtype=dtype,
        shape=tuple(shape),
        size_bytes=size_bytes,
        sha256=require_sha256(record["sha256"], f"{label}.sha256"),
    )


def load_manifest(root: Path) -> tuple[dict[str, JsonValue], bytes]:
    manifest_path = root / MANIFEST_FILE
    checksum_path = root / MANIFEST_CHECKSUM_FILE
    if manifest_path.is_symlink() or checksum_path.is_symlink():
        raise RuntimeError("manifest files must not be symbolic links")
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_digest = sha256_bytes(manifest_bytes)
    expected_checksum_bytes = (
        f"{actual_manifest_digest}  {MANIFEST_FILE}\n".encode("ascii")
    )
    checksum_bytes = checksum_path.read_bytes()
    if checksum_bytes != expected_checksum_bytes:
        raise RuntimeError(
            f"{MANIFEST_CHECKSUM_FILE} does not exactly match {MANIFEST_FILE}"
        )
    try:
        parsed: JsonValue = json.loads(
            manifest_bytes,
            object_pairs_hook=reject_duplicate_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{MANIFEST_FILE} is not valid UTF-8 JSON: {error}") from error
    return require_mapping(parsed, "manifest"), manifest_bytes


def require_lock_match(
    manifest: dict[str, JsonValue],
    lock_path: Path | None,
) -> None:
    manifest_digest = require_sha256(
        manifest.get("oracle_lock_sha256"),
        "manifest.oracle_lock_sha256",
    )
    if lock_path is None:
        return
    if lock_path.is_symlink():
        raise RuntimeError(f"oracle lock must not be a symbolic link: {lock_path}")
    resolved_lock = lock_path.resolve(strict=True)
    if not resolved_lock.is_file():
        raise RuntimeError(f"oracle lock must be a regular non-symlink file: {lock_path}")
    actual_digest = sha256_file(resolved_lock)
    if manifest_digest != actual_digest:
        raise RuntimeError(
            "oracle lock SHA-256 mismatch: "
            f"fixture requires {manifest_digest}, supplied lock is {actual_digest}"
        )


def require_ieee_fp32_runtime_policy(
    manifest: dict[str, JsonValue],
) -> None:
    runtime = require_mapping(manifest.get("runtime"), "manifest.runtime")
    precision = require_string(
        runtime.get("float32_matmul_precision"),
        "manifest.runtime.float32_matmul_precision",
    )
    cuda_tf32 = require_boolean(
        runtime.get("cuda_matmul_allow_tf32"),
        "manifest.runtime.cuda_matmul_allow_tf32",
    )
    cudnn_tf32 = require_boolean(
        runtime.get("cudnn_allow_tf32"),
        "manifest.runtime.cudnn_allow_tf32",
    )
    if precision != "highest" or cuda_tf32 or cudnn_tf32:
        raise RuntimeError(
            "boundary fixture runtime precision policy must be "
            "highest/cuda_matmul_allow_tf32=false/cudnn_allow_tf32=false"
        )


def require_local_ar_seed(manifest: dict[str, JsonValue]) -> None:
    seed = require_integer(
        manifest.get("local_ar_seed"),
        "manifest.local_ar_seed",
    )
    if seed < 0 or seed >= 2**32:
        raise RuntimeError(
            "manifest.local_ar_seed must be in [0, 2^32), "
            f"got {seed}"
        )


def require_no_extra_entries(root: Path, expected_files: frozenset[str]) -> None:
    expected_directories = {""}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = {""}
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"fixture contains a symbolic link: {relative}")
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(f"fixture contains a non-file entry: {relative}")

    extra_files = sorted(actual_files - expected_files)
    missing_files = sorted(expected_files - actual_files)
    extra_directories = sorted(actual_directories - expected_directories)
    if extra_files or missing_files or extra_directories:
        raise RuntimeError(
            "fixture entries mismatch: "
            f"missing_files={missing_files}, extra_files={extra_files}, "
            f"extra_directories={extra_directories}"
        )


def validate_boundary_fixture(root_path: Path, lock_path: Path | None = None) -> int:
    if root_path.is_symlink():
        raise RuntimeError(f"fixture root must not be a symbolic link: {root_path}")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"fixture root is not a directory: {root_path}")
    manifest, _ = load_manifest(root)
    schema_version = require_integer(
        manifest.get("schema_version"),
        "manifest.schema_version",
    )
    if schema_version != 1:
        raise RuntimeError(f"unsupported fixture schema_version: {schema_version}")
    require_lock_match(manifest, lock_path)
    require_ieee_fp32_runtime_policy(manifest)
    require_local_ar_seed(manifest)

    tensor_values = require_list(manifest.get("tensors"), "manifest.tensors")
    records = [
        parse_tensor_record(tensor_value, index)
        for index, tensor_value in enumerate(tensor_values)
    ]
    if not records:
        raise RuntimeError("manifest.tensors must not be empty")
    names = [record.name for record in records]
    paths = [record.path for record in records]
    if len(set(names)) != len(names):
        raise RuntimeError("manifest contains duplicate tensor names")
    if len(set(paths)) != len(paths):
        raise RuntimeError("manifest contains duplicate tensor paths")

    expected_files = frozenset(
        (MANIFEST_FILE, MANIFEST_CHECKSUM_FILE, *(record.path for record in records))
    )
    require_no_extra_entries(root, expected_files)

    for record in records:
        tensor_path = root / PurePosixPath(record.path)
        resolved_tensor = tensor_path.resolve(strict=True)
        if not resolved_tensor.is_relative_to(root):
            raise RuntimeError(f"tensor path escapes fixture root: {record.path}")
        if not resolved_tensor.is_file():
            raise RuntimeError(f"tensor is not a regular file: {record.path}")
        actual_size = resolved_tensor.stat().st_size
        if actual_size != record.size_bytes:
            raise RuntimeError(
                f"tensor size mismatch for {record.name}: "
                f"expected {record.size_bytes}, got {actual_size}"
            )
        actual_digest = sha256_file(resolved_tensor)
        if actual_digest != record.sha256:
            raise RuntimeError(
                f"tensor SHA-256 mismatch for {record.name}: "
                f"expected {record.sha256}, got {actual_digest}"
            )
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        help="when supplied, require the fixture to reference these exact lock bytes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = validate_boundary_fixture(args.fixture, args.lock)
    print(f"boundary fixture verified: {count} tensors")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"boundary fixture verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

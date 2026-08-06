#!/usr/bin/env python3
"""Verify every external input used by the accepted Sofia oracle.

This tool deliberately does not search for assets or choose a fallback. Every
path is supplied by the caller and every byte is checked against the lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

ORACLE_TOOLS = Path(__file__).resolve().parent
if str(ORACLE_TOOLS) not in sys.path:
    sys.path.insert(0, str(ORACLE_TOOLS))

from source_acceptance_receipt import (
    SourceAcceptanceError,
    ordered_source_bundle_sha256,
    validate_public_acceptance,
)


@dataclass(frozen=True)
class FileExpectation:
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_file(path: Path, expected: FileExpectation, label: str) -> None:
    resolved = path.resolve(strict=True)
    actual_size = resolved.stat().st_size
    if actual_size != expected.size_bytes:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected.size_bytes}, got {actual_size}"
        )
    actual_digest = sha256_file(resolved)
    if actual_digest != expected.sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected.sha256}, got {actual_digest}"
        )


def require_source_checkout(
    speech_root: Path,
    expected_revision: str,
    expected_files: dict[str, str],
    expected_bundle_sha256: str,
) -> None:
    root = speech_root.resolve(strict=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if revision != expected_revision:
        raise RuntimeError(
            f"NeMo Speech revision mismatch: expected {expected_revision}, got {revision}"
        )
    actual_files: dict[str, str] = {}
    for relative, expected_digest in expected_files.items():
        source_path = (root / relative).resolve(strict=True)
        if not source_path.is_relative_to(root):
            raise RuntimeError(f"oracle source escapes the checkout: {relative}")
        actual_digest = sha256_file(source_path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"oracle source SHA-256 mismatch for {relative}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        actual_files[relative] = actual_digest
    actual_bundle_sha256 = ordered_source_bundle_sha256(actual_files)
    if actual_bundle_sha256 != expected_bundle_sha256:
        raise RuntimeError(
            "ordered oracle source bundle mismatch: "
            f"expected {expected_bundle_sha256}, got {actual_bundle_sha256}"
        )


def require_model_configs(
    model_path: Path,
    expected_members: list[str],
    active_policy: str,
    expected_active: str,
) -> None:
    if active_policy != "last_tar_member":
        raise RuntimeError(f"unsupported model config selection policy: {active_policy}")
    with tarfile.open(model_path.resolve(strict=True), mode="r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.lstrip("./") == "model_config.yaml"
        ]
        member_digests: list[str] = []
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError("model_config.yaml tar member is not a regular file")
            member_digests.append(sha256_bytes(stream.read()))
    if member_digests != expected_members:
        raise RuntimeError(
            "model_config.yaml occurrence mismatch: "
            f"expected {expected_members}, got {member_digests}"
        )
    if not member_digests or member_digests[-1] != expected_active:
        raise RuntimeError(
            f"active model config mismatch: expected {expected_active}, "
            f"got {member_digests[-1] if member_digests else 'no member'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-model", type=Path, required=True)
    parser.add_argument("--acceptance-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = json.loads(args.lock.resolve(strict=True).read_text(encoding="utf-8"))
    model = lock["model"]
    codec = lock["codec"]
    source = lock["oracle_source"]
    acceptance = lock["acceptance"]

    require_file(
        args.model,
        FileExpectation(model["sha256"], model["size_bytes"]),
        "Magpie model",
    )
    require_model_configs(
        args.model,
        model["config_member_sha256"],
        model["active_config_policy"],
        model["active_config_sha256"],
    )
    require_file(
        args.codec_model,
        FileExpectation(codec["sha256"], codec["size_bytes"]),
        "NanoCodec model",
    )
    require_file(
        args.acceptance_receipt,
        FileExpectation(
            acceptance["receipt_sha256"],
            acceptance["receipt_size_bytes"],
        ),
        "acceptance receipt",
    )
    acceptance_receipt = json.loads(
        args.acceptance_receipt.resolve(strict=True).read_text(encoding="utf-8")
    )
    if not isinstance(acceptance_receipt, dict):
        raise RuntimeError("acceptance receipt root must be an object")
    validate_public_acceptance(acceptance_receipt, lock)
    require_source_checkout(
        args.speech_root,
        source["base_revision"],
        source["files"],
        source["optimized_source_bundle_sha256"],
    )
    print("oracle lock verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        SourceAcceptanceError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"oracle lock verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

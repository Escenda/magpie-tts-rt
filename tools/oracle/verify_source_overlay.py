#!/usr/bin/env python3
"""Verify the committed NeMo oracle overlay and its ordered bundle digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = json.loads(args.lock.resolve(strict=True).read_text(encoding="utf-8"))
    source = lock["oracle_source"]
    overlay = args.overlay_root.resolve(strict=True)
    bundle = hashlib.sha256()
    for relative, expected_digest in source["files"].items():
        path = (overlay / relative).resolve(strict=True)
        if not path.is_relative_to(overlay):
            raise RuntimeError(f"overlay path escapes its root: {relative}")
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"overlay SHA-256 mismatch for {relative}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        bundle.update(relative.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(actual_digest.encode("ascii"))
        bundle.update(b"\n")
    actual_bundle = bundle.hexdigest()
    expected_bundle = source["optimized_source_bundle_sha256"]
    if actual_bundle != expected_bundle:
        raise RuntimeError(
            f"ordered source bundle mismatch: expected {expected_bundle}, got {actual_bundle}"
        )
    print(f"oracle source overlay verified: {len(source['files'])} files")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"oracle source overlay verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

#!/usr/bin/env python3
"""Persistent strict-JSONL process for the locked Japanese frontend."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, Protocol, TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.frontend.japanese_frontend import (
    FrontendError,
    FrontendSpec,
    LockedJapaneseFrontend,
    PreparedSegments,
    tokenizer_identity_sha256,
)


SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 1024 * 1024


class Frontend(Protocol):
    @property
    def spec(self) -> FrontendSpec: ...

    def prepare_segments(self, source_text: str) -> PreparedSegments: ...


class ProtocolError(RuntimeError):
    """The parent process violated the frontend transport contract."""


def _request(line: bytes) -> tuple[str, str]:
    if not line.endswith(b"\n"):
        raise ProtocolError("request line must end with LF")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("request must be one strict UTF-8 JSON object") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "request_id",
        "text",
    }:
        raise ProtocolError(
            "request fields must be schema_version, request_id, and text"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ProtocolError("request schema_version must be 1")
    request_id = value["request_id"]
    text = value["text"]
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("request_id must be a non-empty string")
    if "\x00" in request_id or len(request_id.encode("utf-8")) > 256:
        raise ProtocolError("request_id exceeds the transport contract")
    if not isinstance(text, str):
        raise ProtocolError("text must be a string")
    return request_id, text


def _write(output: TextIO, value: dict) -> None:
    output.write(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    output.write("\n")
    output.flush()


def prepared_response(
    frontend: Frontend,
    identity_sha256: str,
    request_id: str,
    text: str,
) -> dict:
    prepared = frontend.prepare_segments(text)
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "prepared_segments",
        "request_id": request_id,
        "tokenizer_identity_sha256": identity_sha256,
        "source_text": prepared.source_text,
        "segmentation_mode": prepared.segmentation_mode,
        "segments": [
            {
                "segment_index": segment.segment_index,
                "source_char_start": segment.source_char_start,
                "source_char_end": segment.source_char_end,
                "source_utf8_start": segment.source_utf8_start,
                "source_utf8_end": segment.source_utf8_end,
                "normalized_text": segment.prepared.normalized_text,
                "global_token_ids": list(
                    segment.prepared.global_token_ids
                ),
                "progress": [
                    asdict(item) for item in segment.prepared.progress
                ],
            }
            for segment in prepared.segments
        ],
    }


def serve(
    frontend: Frontend,
    identity_sha256: str,
    source: BinaryIO,
    output: TextIO,
) -> None:
    _write(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "type": "ready",
            "tokenizer_identity_sha256": identity_sha256,
            "tokenizer_vocabulary_size": (
                frontend.spec.aggregate_vocabulary_size
            ),
            "text_embedding_rows": frontend.spec.text_embedding_rows,
            "bos_token_id": frontend.spec.bos_token_id,
            "eos_token_id": frontend.spec.eos_token_id,
            "japanese_global_pad_token_id": (
                frontend.spec.global_pad_token_id
            ),
        },
    )
    while True:
        line = source.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST_BYTES:
            raise ProtocolError("request exceeds the one-MiB transport limit")
        request_id, text = _request(line)
        try:
            response = prepared_response(
                frontend, identity_sha256, request_id, text
            )
        except (FrontendError, TypeError, ValueError) as error:
            response = {
                "schema_version": SCHEMA_VERSION,
                "type": "error",
                "request_id": request_id,
                "error_code": type(error).__name__,
                "message": str(error),
            }
        _write(output, response)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--frontend-contract", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    frontend = LockedJapaneseFrontend.from_files(
        arguments.lock, arguments.frontend_contract
    )
    identity = tokenizer_identity_sha256(
        arguments.lock, arguments.frontend_contract
    )
    serve(frontend, identity, sys.stdin.buffer, sys.stdout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FrontendError, ProtocolError, OSError) as error:
        print(f"Japanese frontend server failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

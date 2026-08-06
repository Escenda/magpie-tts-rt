from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pytest

from tools.frontend.japanese_frontend import (
    AlignmentProof,
    PreparedSegment,
    PreparedSegments,
    PreparedToken,
    PreparedUtterance,
    SourceProgress,
)
from tools.frontend.japanese_frontend_server import (
    ProtocolError,
    _request,
    serve,
)


@dataclass(frozen=True)
class _Spec:
    aggregate_vocabulary_size: int = 3_357
    text_embedding_rows: int = 3_359
    bos_token_id: int = 3_357
    eos_token_id: int = 3_358
    global_pad_token_id: int = 1_015


class _Frontend:
    spec = _Spec()

    def prepare_segments(self, source_text: str) -> PreparedSegments:
        if source_text == "失敗":
            raise ValueError("fixture rejection")
        source_utf8_length = len(source_text.encode("utf-8"))
        token = PreparedToken(
            token_index=0,
            symbol="<eos>",
            kind="eos",
            unit_index=None,
            local_token_id=None,
            global_token_id=3_358,
            source_char_start=0,
            source_char_end=len(source_text),
            source_utf8_start=0,
            source_utf8_end=source_utf8_length,
            commit_char_end=len(source_text),
            commit_utf8_end=source_utf8_length,
        )
        prepared = PreparedUtterance(
            source_text=source_text,
            normalized_text=source_text,
            g2p_tokens=("<eos>",),
            global_token_ids=(3_358,),
            tokens=(token,),
            progress=(
                SourceProgress(0, 0, 0),
                SourceProgress(1, len(source_text), source_utf8_length),
            ),
            proof=AlignmentProof(
                method="fixture",
                prepared_char_count=len(source_text),
                core_token_count=1,
                certified_prefix_count=1,
                segments=(),
            ),
        )
        return PreparedSegments(
            source_text=source_text,
            segmentation_mode="independent_sentence_segments_v1",
            segments=(
                PreparedSegment(
                    segment_index=0,
                    source_char_start=0,
                    source_char_end=len(source_text),
                    source_utf8_start=0,
                    source_utf8_end=source_utf8_length,
                    prepared=prepared,
                ),
            ),
        )


def _records(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_server_stays_loaded_and_returns_exact_progress() -> None:
    source = io.BytesIO(
        b'{"schema_version":1,"request_id":"a","text":"\\u3042"}\n'
        b'{"schema_version":1,"request_id":"b","text":"\\u5931\\u6557"}\n'
    )
    output = io.StringIO()
    serve(_Frontend(), "1" * 64, source, output)
    records = _records(output)
    assert records[0] == {
        "schema_version": 1,
        "type": "ready",
        "tokenizer_identity_sha256": "1" * 64,
        "tokenizer_vocabulary_size": 3_357,
        "text_embedding_rows": 3_359,
        "bos_token_id": 3_357,
        "eos_token_id": 3_358,
        "japanese_global_pad_token_id": 1_015,
    }
    assert records[1]["type"] == "prepared_segments"
    assert (
        records[1]["segmentation_mode"]
        == "independent_sentence_segments_v1"
    )
    assert records[1]["source_text"] == "あ"
    assert records[1]["segments"][0]["global_token_ids"] == [3_358]
    assert records[1]["segments"][0]["progress"][-1] == {
        "committed_text_tokens": 1,
        "source_char_end": 1,
        "source_utf8_end": 3,
    }
    assert records[2]["type"] == "error"
    assert records[2]["request_id"] == "b"
    assert records[2]["error_code"] == "ValueError"


@pytest.mark.parametrize(
    "line",
    [
        b"{}\n",
        b'{"schema_version":2,"request_id":"a","text":"x"}\n',
        b'{"schema_version":1,"request_id":"","text":"x"}\n',
        b'{"schema_version":1,"request_id":"a","text":"x"}',
        b"\xff\n",
    ],
)
def test_protocol_errors_are_process_fatal(line: bytes) -> None:
    with pytest.raises(ProtocolError):
        _request(line)

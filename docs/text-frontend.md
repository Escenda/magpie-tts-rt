# Japanese text frontend contract

MagpieTTS-RT accepts global token IDs. It does not accept raw text and does not
guess a reading. The application frontend must reproduce the tokenizer used to
build the verified engine bundle.

For Japanese v2607, the accepted path is:

1. remove every whitespace character while retaining its source position;
2. apply the locked lowercase preprocessing;
3. run OpenJTalk full-context analysis;
4. derive Katakana plus the `0`/`1` pitch marker placed before each mora;
5. tokenize with NeMo `JapanesePhonemeTokenizer`;
6. add the `japanese_phoneme` offset from the aggregate multilingual
   tokenizer;
7. append the aggregate EOS token to every text chunk; and
8. retain a proven mapping from every prepared token position to the original
   source string.

OpenJTalk is required. The upstream character-count path used when
`pyopenjtalk` is unavailable changes chunking and therefore speech output; it
is not an accepted fallback.

## Offline frontend release assets

The realtime runtime must not build `pyopenjtalk` or let it download a
dictionary. PyPI publishes `pyopenjtalk 0.4.1` as a source distribution for
Linux aarch64, so release preparation and runtime installation are separate
operations:

1. A release builder uses
   `tools/frontend/build_pyopenjtalk_release_wheel` with a pre-populated input
   wheelhouse. The source archive and every PEP-517 build requirement must
   match
   `requirements-pyopenjtalk-wheel-build-aarch64-cp312.lock`.
2. The builder tests the new wheel in a second copied venv with dependencies
   from
   `requirements-frontend-runtime-dependencies-aarch64-cp312.lock`.
3. The builder publishes a new output directory with Linux
   `RENAME_NOREPLACE`. The directory contains the wheel, its checksum sidecar,
   and a machine-readable build receipt. It never replaces an existing
   release asset.
4. A deployment release pins that exact wheel and build receipt outside the
   runtime directory. Runtime installation uses `--no-index`,
   `--only-binary=:all:`, and `--require-hashes` with
   `requirements-aarch64-cp312.lock`; it never falls back to the source
   archive.
5. The deployment supplies `open_jtalk_dic_utf_8-1.11.tar.gz` as a separate
   pinned input, verifies its filename, size, and SHA-256, and safely extracts
   only regular files and directories. The runtime sets
   `OPEN_JTALK_DICT_DIR` to that accepted tree before importing or calling
   `pyopenjtalk`.

Example release-wheel build:

```bash
tools/frontend/build_pyopenjtalk_release_wheel \
  --input-wheelhouse /absolute/pinned/pyopenjtalk-build-inputs \
  --output /absolute/new/pyopenjtalk-release-asset
```

The build receipt records the exact source archive, build/runtime lock hashes,
Python, CMake, compiler, and the resulting wheel SHA-256. This records how a
candidate was produced; the deployment's tracked external pin remains the
authority for whether that candidate is accepted.

## Locked values

| Item | Value |
| --- | --- |
| aggregate vocabulary | `3,357` |
| text embedding rows | `3,359` |
| Japanese global offset | `842` |
| Japanese local vocabulary | `175` |
| global Japanese batch-pad ID | `1,015` |
| global boundary-space ID | `842` |
| global BOS / EOS | `3,357` / `3,358` |

`pad_with_space=True` inserts local token `0` (global `842`) at the beginning
and end of an utterance. This is different from the tokenizer's batch-padding
token, local `173` (global `1,015`).

The aggregate vocabulary count covers normal rows `0..3356`; it is not the
upper bound of the embedding table. Every prepared native request consists of
one or more normal rows followed by exactly one global EOS `3358`. Global BOS
`3357` is authenticated but is not inserted into a prepared request. The
frontend server, Rust consumer, native request boundary, startup fixture, and
bundle packager all enforce this same rule.

The Japanese token list contains a duplicate `ー`. The actual mapping uses the
last occurrence, local ID `165`; rebuilding the map with a first-index search
changes token IDs.

The minimal locked fixture is:

```text
raw:       こんにちは。
G2P:       0 コ 1 ン 1 ニ 1 チ 1 ワ 。
local IDs: 0 1 21 2 83 2 45 2 35 2 81 139 0
global:    842 843 863 844 925 844 887 844 877 844 923 981 842 3358
```

## Public frontend API

The correctness-first Python API is
`tools.frontend.japanese_frontend.LockedJapaneseFrontend`:

```python
from pathlib import Path

from tools.frontend.japanese_frontend import LockedJapaneseFrontend

frontend = LockedJapaneseFrontend.from_files(
    Path("reference/oracle-lock.json"),
    Path("/path/to/verified-boundary-fixture/manifest.json"),
)
utterance = frontend.prepare("2026年7月30日")
```

`from_files()` verifies all of the following before accepting text:

- frontend constants and the full 175-row token table against
  `oracle-lock.json`;
- the token-table canonical SHA-256, including duplicate-token resolution;
- the installed `pyopenjtalk` version;
- every file, byte size, and SHA-256 in the OpenJTalk dictionary; and
- the locked `こんにちは。` G2P, local-ID, aggregate-offset, and EOS fixture.

There is no environment-check bypass in the public constructor.

The equivalent machine-readable CLI is:

```bash
python3 tools/frontend/japanese_frontend.py \
  --lock reference/oracle-lock.json \
  --frontend-contract /path/to/verified-boundary-fixture/manifest.json \
  --text '2026年7月30日' \
  --pretty
```

`--text-file` accepts strict UTF-8 instead of `--text`. `--output` creates a
new file and refuses to overwrite an existing file.

For a realtime process, load the same assets once and keep the frontend alive:

```bash
python3 tools/frontend/japanese_frontend_server.py \
  --lock reference/oracle-lock.json \
  --frontend-contract /path/to/tokenizer-identity-receipt.json
```

The child emits one strict JSONL `ready` record containing
`tokenizer_identity_sha256`, then accepts exact
`schema_version/request_id/text` request objects. A successful response carries
the prepared global token IDs and complete source-progress table. A text that
cannot be proven returns a typed per-request error. Malformed JSON, an unknown
field, invalid UTF-8, an overlong line, or an asset mismatch terminates the
process; the parent must not silently restart with another frontend.

## Runtime-to-source mapping

`PreparedUtterance.global_token_ids` is the exact sequence passed to the C ABI.
Each `PreparedToken` contains:

| Field | Meaning |
| --- | --- |
| `token_index` | zero-based prepared token index |
| `symbol`, `kind`, `unit_index` | token identity; a pitch marker and its mora characters share one unit |
| `local_token_id`, `global_token_id` | locked tokenizer IDs; EOS has no local ID |
| `source_char_start/end` | half-open source span in Unicode scalar indices |
| `source_utf8_start/end` | the same half-open span in UTF-8 byte offsets |
| `commit_char_end`, `commit_utf8_end` | source boundary safe to expose after this token |

“Character” here means a Unicode scalar as indexed by Python, not a grapheme
cluster and not a UTF-16 code unit. UTF-8 consumers should use the byte offsets
directly.

`PreparedUtterance.progress` has `token_count + 1` entries. Its index is the
C ABI's end-exclusive `committed_text_tokens` value. Rust and ROS consumers
must perform this lookup:

```text
C ABI committed_text_tokens
        -> progress[committed_text_tokens]
        -> source_char_end / source_utf8_end
```

They must not divide token counts, count Katakana, or estimate source
characters. EOS always maps to the exact source character and UTF-8 byte end.

## Why numeric expansions do not advance early

OpenJTalk can turn `2026` into multiple NJD words such as `二`, `千`, `二`,
`十`, `六`; the NJD strings do not retain source offsets. Matching those
normalized strings back to the input would be a guess.

The frontend instead replays OpenJTalk for every input prefix. A boundary is
certified only if its complete token output is an exact prefix of the final
token output and ends at a pitch-plus-mora unit boundary. Certified token and
source positions must never move backward. Newly proven tokens and newly
consumed source characters form one `SourceSegment`.

Every token in a segment retains that segment's exact source lineage. The
segment's source end becomes committed only on its final token. For example,
the reading expanded from `2026` cannot expose source positions `1`, `2`, or
`3`; it moves from `0` to `4` only after the whole certified expansion has
been played. The same rule applies to decimals, dates, and units.

The proof is accepted only when:

- source and core-token segments are strictly ordered and contiguous;
- every core token belongs to exactly one segment;
- every non-whitespace source suffix has a spoken token;
- all prepared-token progress values are monotonic; and
- EOS reaches the exact source end.

Whitespace removed before G2P remains in the source spans. Leading and
interstitial whitespace is committed with the next proven spoken segment;
trailing whitespace is committed by the final boundary-space token and EOS.

## Fail-closed cases

The frontend returns an error instead of a guessed map when:

- lowercase preprocessing expands one source Unicode scalar or is
  context-sensitive;
- a non-whitespace input becomes an empty prepared speech sequence;
- a certified OpenJTalk prefix would move token progress backward;
- a token-growth boundary has no forward source interval;
- a non-whitespace suffix has no spoken token; or
- the final proof does not cover all tokens and the exact source end.

NFKC forms that retain a provable prefix lineage, including full-width Latin,
full-width digits, and half-width Katakana, remain accepted. No NJD
`string`-to-source heuristic is used.

## Golden span fixture

`tests/frontend/fixtures/japanese-source-spans-v1.json` is machine-readable
and records the locked OpenJTalk rows for every prefix. It covers:

- Hiragana, Kanji, and punctuation;
- digits and a date;
- a decimal and unit;
- full-width digits and unit;
- full-width/ASCII Latin; and
- half-width Katakana/NFKC behavior.

The capture tool independently compares every case with locked NeMo before it
writes the fixture:

```bash
python3 tools/frontend/capture_japanese_span_fixture.py \
  --lock reference/oracle-lock.json \
  --frontend-contract /path/to/verified-boundary-fixture/manifest.json \
  --speech-root /path/to/locked/NeMo-Speech \
  --output tests/frontend/fixtures/japanese-source-spans-v1.json
```

Negative fixtures cover Unicode lowercase expansion, non-whitespace text that
OpenJTalk turns into silence, whitespace-only input, and a synthetic certified
prefix regression.

## Chunking boundary

The long-text split threshold is 80 OpenJTalk NJD words. It is not 80 Unicode
characters. `pyopenjtalk.run_frontend` is the oracle for that count.
`prepare_segments()` first proves the complete source lineage, then selects
only boundaries that are both a certified `SourceSegment` end and a sentence
end (`。`, `！`, `？`, or newline). Every returned segment carries its exact
source character and UTF-8 base offsets. A single sentence over the limit has
no legal split and fails closed.

The public JSONL response is `prepared_segments` with
`segmentation_mode=independent_sentence_segments_v1`. Each segment is a fresh
native synthesis request. This preserves exact source lineage and one logical
application utterance, but it does **not** reproduce NeMo's training-time
chunk-history path (`history_text`, `history_context_tensor`, attention
offsets, and codec emission state). Matching that path requires a future C ABI
and runtime state-continuation contract; consumers must not describe these
independent segments as model-state-continuous long-form synthesis.

The model bundle must identify the tokenizer vocabulary, every tokenizer
offset, special token IDs, OpenJTalk dictionary/version, normalization rules,
and the frontend golden fixture. It does so with the receipt-derived
`identity_sha256` defined in
[Runtime bundle packaging](runtime-bundle-packaging.md), not with a raw
tokenizer file. The ROS consumer may report source-character progress only
through this retained token-to-source map.

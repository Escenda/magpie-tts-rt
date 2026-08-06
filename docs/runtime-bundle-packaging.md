# Runtime bundle packaging

`tools/bundle/package_runtime_bundle.py` is the only supported P2 bundle
constructor. It packages an immutable directory and emits the external trust
anchor in `runtime-bundle-manifest.json.sha256`. The checksum file is a
transport convenience; a caller still has to authenticate that digest through
its release or deployment mechanism.

The tool does not package the upstream `.nemo` model. TensorRT execution does
not read those weights, and copying them into a runtime bundle would add an
unused 1.47 GB artifact and a separate redistribution obligation. The manifest
instead records:

- the upstream model ID, display version, immutable 40-hex revision, and exact
  source `.nemo` SHA-256;
- the accepted source-model receipt that proves which bytes were used;
- one consolidated export receipt;
- a Japanese frontend/tokenizer identity receipt;
- the authenticated plugin and its deterministic build-provenance receipt;
- exactly seven TensorRT plans;
- one self-contained startup golden input fixture;
- one complete-generation golden receipt;
- exactly eight authenticated license/notice artifacts: MagpieTTS-RT,
  PyTorch, CUTLASS, the redistributed CUDA static runtime components, and the
  NVIDIA model.

There is no raw-model or raw-tokenizer compatibility path.

## Generated versus specified fields

The package specification contains non-engine semantics such as CFG order,
alignment policy, RNG rules, codec schedule, limits, and destination paths. It
does not contain an `engines` manifest or any tensor shape.

The reviewed Sofia/Thor v1 values and their source evidence are recorded in
[`reference/runtime-bundle-package-spec-v1.json`](../reference/runtime-bundle-package-spec-v1.json)
and its
[`review note`](../reference/runtime-bundle-package-spec-v1.md). Candidate
receipts and mock artifacts are never substituted into that specification.

For each plan, the packager deserializes the exact bytes copied into the
staging bundle and reads every I/O name, dtype, declared shape, and every
dynamic input's min/opt/max range through TensorRT. It also derives:

- Main Decoder layer bindings and KV dimensions from prefill and one-step
  bindings;
- all 97 NanoCodec persistent-state records from the initial, steady, and tail
  plans.

The C++ manifest parser then validates the generated result against the v1
semantic contract. A hand-written shape, renamed binding, omitted profile,
extra state, or mismatched connected dimension cannot enter a published
bundle.

Manifest engine names and profile labels are canonical runtime role labels
and are not freely configurable. TensorRT does not preserve useful semantic
engine/profile names in these plans (`trtexec` reports an unnamed network);
the binding names, dtypes, declared shapes, and every profile range are the
fields obtained from plan introspection.

## Tokenizer identity

The tokenizer artifact is an identity receipt, not a tokenizer model file.
The accepted receipt contains the complete `frontend_contract` object and its
ordered
`frontend_contract.japanese.token_table`.

All JSON hashes below use UTF-8 JSON with keys sorted, separators `,` and `:`,
no insignificant whitespace, `ensure_ascii=false`, and no trailing newline.
Array order is retained.

1. `frontend_contract_sha256` hashes the complete `frontend_contract` object.
2. `vocabulary_sha256` hashes the ordered Japanese `token_table` array. It
   must also equal the receipt's `token_table_sha256`.
3. `special_tokens` is exactly:
   `bos_token_id`, `eos_token_id`, and
   `japanese_global_pad_token_id`.
4. `identity_sha256` hashes this projection:

```json
{
  "frontend_contract_sha256": "<sha256>",
  "kind": "japanese_phoneme",
  "special_tokens": {
    "bos_token_id": 3357,
    "eos_token_id": 3358,
    "japanese_global_pad_token_id": 1015
  },
  "vocabulary_sha256": "<sha256>",
  "vocabulary_size": 3357
}
```

The packager recomputes all three hashes. A changed frontend dependency,
normalization contract, vocabulary row, row order, global offset, or special
token fails closed.

The identity projection keeps the historical key `vocabulary_size`; it means
the 3,357 normal tokenizer rows. The emitted runtime manifest names that same
value `tokenizer_vocabulary_size` and separately authenticates
`text_embedding_rows=3359`, `bos_token_id=3357`, `eos_token_id=3358`, and
`japanese_global_pad_token_id=1015`. A prepared request is nonempty, ends in
exactly one EOS row, and contains only normal tokenizer rows before EOS. BOS is
an authenticated embedding row but is not a valid prepared-request token.

The consolidated export receipt must declare status `accepted`; a
`measured-not-accepted` receipt is not a bundle input even when its files and
shapes are otherwise valid.

Schema version 1 accepts exactly the export format identifier
`magpie_tts_rt_export_v1`. The packager, JSON schema, and native manifest
parser reject alternate labels so a bundle cannot claim a different export
contract while loading the v1 engines.

Sofia v1 sampling has one canonical token partition: codec tokens are
`0..2015`, `AUDIO_EOS` is `2017`, and the static forbidden set is exactly
`[2016,2018,2019,2020,2021,2022,2023]` in that order. The schema, packager,
and manifest parser reject any other EOS ID or forbidden set. This metadata
therefore cannot disagree with the authenticated Local AR plugin.

The codec stream boundary is also fixed metadata, not an implementation
choice. `eos_frame_is_audio` is `false` and `zero_frame_finalization` is
`control_marker_without_codec_invocation` in the reviewed package spec, the
accepted consolidated export receipt, the complete-generation golden receipt,
and the emitted bundle manifest. The tail plan remains `F=1..8`. If EOS leaves
zero audio frames, the runtime must skip NanoCodec and emit a zero-frame
`FINAL` control marker after preceding PCM has been published. The playback
consumer remains responsible for physical-device drain. The packager rejects
a receipt that counts EOS as audio or claims a zero-frame
tail-engine invocation.

## Startup golden fixture

Session creation is gated by a real complete generation. The bundle therefore
contains a `golden_fixture` JSON artifact in addition to its accepted golden
receipt. Its exact format is
[`schemas/golden-fixture.schema.json`](../schemas/golden-fixture.schema.json).

The fixture carries the prepared token IDs and uint32 seed needed to run the
model. It does not carry expected PCM bytes. The runtime hashes the actual
complete decoder codes, scheduled NanoCodec inputs, and valid FP32 PCM, then
compares those hashes and counts with the fixture before the session becomes
ready. A mismatch fails session creation; there is no alternate fixture or
skip path.

The token hash is over contiguous little-endian signed INT32 values in row
order. Decoder/code hashes are over the documented contiguous little-endian
INT64 sequences. PCM hashes cover concatenated valid little-endian FP32
samples with no padding. The packager recomputes the token hash and requires
the fixture's tokenizer identity, oracle lock, seed, all expected hashes,
codec frame count, and PCM sample count to match the accepted receipt and
package specification.

The consolidated export receipt is also checked against the exact staged
plugin, plugin build receipt, all seven plans, source/tokenizer receipts,
golden fixture, and golden receipt by SHA-256 and size. Its component and
engine arrays have fixed canonical order, so a valid receipt for one set of
plans cannot authorize a different compatible-looking set.

## Plugin and runtime identity

Before any plan is deserialized, the copied plugin must expose only the v1
query contract expected by the runtime. The packager queries
`mtt_plugin_get_api_v1`, checks the exact five creator identities
(`MagpieLocalARSampling`, `MagpieLocalAREos`, `MagpieLayerNorm`,
`MagpieGeluTanh`, and `MagpieSoftmax`), validates all ABI sizes and reserved
fields, and explicitly registers the creators.

The plugin artifact's logical library name is `magpie_tts_rt_plugins`.
`local_ar.sampling_plugin_name` is instead the TensorRT creator identity
`MagpieLocalARSampling`. The packager and native loader authenticate both
fields independently; treating the library name as a creator name fails
closed before plan deserialization.

Before that first `dlopen`, the packager also inspects the staged ELF dynamic
section. It requires SONAME `libmagpie_tts_rt_plugins.so.0`, the exact fixed
dependency set (`libcublas.so.13`, `libcudart.so.13`, `libnvinfer.so.10`,
`libstdc++.so.6`, `libgcc_s.so.1`, `libc.so.6`, and
`ld-linux-aarch64.so.1`), and rejects `RPATH`, `RUNPATH`, loader-audit,
filter, and auxiliary-object tags. A changed dependency closure is a
packaging failure rather than an accepted host fallback.

The required plugin build receipt is validated before packaging. It binds the
exact plugin SHA-256 and size to the hashed source inventory, CUDA `sm_110`,
Release/TF32 policy, pinned CUTLASS archive, compiler/linker identities, and
the exact compile and link commands reported by Ninja. Source and build roots
are replaced by literal `${SOURCE_ROOT}` and `${BUILD_ROOT}` tokens so two
independent clean directories can produce the same receipt. The compile
command must contain the deterministic nvcc
`--frandom-seed=magpie_tts_rt_plugins_v1` flag, the complete `sm_110`,
Release, PIC, and warning policy; the link command must contain the version
script, SONAME, no-undefined, and strip policy. The source inventory is exactly
the five plugin build inputs and is rehashed from the repository again by the
final acceptance command. The consolidated export receipt authenticates the
receipt's SHA-256 and size, and the receipt itself is copied into the
immutable bundle. A matching plugin without matching build provenance is
rejected.

The runtime fingerprint is collected live rather than copied from the
specification:

- Ubuntu release and architecture;
- endianness;
- CUDA runtime and TensorRT versions;
- NVIDIA driver version;
- CUDA device name and compute capability;
- authenticated plugin ABI.

P2 v1 packaging is restricted to AGX Thor (`NVIDIA Thor`, `sm_110`). The same
fingerprint is checked again when a bundle is loaded.

## Filesystem and publication rules

- Every input must be a nonempty regular file. Symlinks in any input path
  component and lexical `..` components are rejected.
- Destination paths are normalized relative POSIX paths. Absolute paths,
  backslashes, control bytes, `.`/`..`, duplicate paths, and path-prefix
  collisions are rejected.
- Each input is copied through `O_NOFOLLOW`; source identity, size, ctime, and
  mtime must remain unchanged for the complete copy.
- Two artifact roles may not refer to the same device/inode.
- Receipts are parsed and authenticated from the staged copies. The plugin and
  plans are likewise loaded from staged copies, not reopened from their
  original paths.
- JSON duplicate keys are rejected; receipt and specification reads are
  bounded and require stable file identity, size, ctime, and mtime.
- `mtt-validate-manifest` and `mtt-validate-bundle` must both pass before
  publication.
- After both validators return, the packager enumerates the staging tree
  again, streams and rehashes every artifact, the manifest, and the manifest
  digest record, and then enumerates the tree a second time. Any byte,
  identity, timestamp, entry-type, or entry-set change prevents publication.
- The native bundle verifier accepts exactly one manifest, every artifact
  declared by that manifest, and only the parent directories required by
  those files. The only non-manifest artifact exception is the optional
  sibling `<manifest>.sha256` transport file. When present, its bytes must be
  exactly `<trusted-sha256>  <manifest-relative-path>\n`. Extra regular files,
  empty directories, symbolic links, FIFOs, sockets, and other special entries
  fail closed.
- The final directory is published with
  `renameat2(RENAME_NOREPLACE)`. Existing output is never overwritten.

## Command

The seven engine roles and eight legal artifacts are intentionally explicit:

```bash
python3 tools/bundle/package_runtime_bundle.py \
  --spec /path/to/runtime-bundle-package-spec-v1.json \
  --source-model-acceptance-receipt /path/to/model-acceptance.json \
  --export-receipt /path/to/runtime-export-receipt.json \
  --tokenizer-identity-receipt /path/to/japanese-source-spans-v1.json \
  --plugin-build-receipt /path/to/plugin-build-receipt.json \
  --plugin /path/to/libmagpie_tts_rt_plugins.so \
  --text-encoder-plan /path/to/text_encoder.plan \
  --main-decoder-prefill-plan /path/to/main-decoder-prefill.plan \
  --main-decoder-step-plan /path/to/main-decoder-step.plan \
  --local-ar-plan /path/to/local-ar.plan \
  --nanocodec-initial-4-plan /path/to/nanocodec-initial-4.plan \
  --nanocodec-steady-8-plan /path/to/nanocodec-steady-8.plan \
  --nanocodec-tail-1-8-plan /path/to/nanocodec-tail-1-8.plan \
  --golden-fixture /path/to/golden-fixture.json \
  --golden-receipt /path/to/golden-receipt.json \
  --project-license LICENSE \
  --project-notice NOTICE \
  --pytorch-license third_party/pytorch/LICENSE \
  --cutlass-license third_party/cutlass/LICENSE.txt \
  --cuda-eula third_party/cuda/EULA.txt \
  --cuda-notice third_party/cuda/NOTICE \
  --nvidia-open-model-license third_party/nvidia-open-model-license/NVIDIA-Open-Model-License-Agreement-2025-10-24.pdf \
  --nvidia-model-notice third_party/nvidia-open-model-license/NOTICE \
  --manifest-validator build/mtt-validate-manifest \
  --bundle-validator build/mtt-validate-bundle \
  --output /path/to/new-bundle
```

Add `--check-inputs` to report every missing, symlinked, empty, nonregular, or
nonexecutable input in one pass without creating a staging directory.

The final accepted bundle must not be generated until the seed contract,
locked fixtures, seven accepted plans, consolidated export receipt, and golden
receipt are all final. A measured-but-not-accepted plan remains a missing P2
input rather than a degraded bundle.

After publishing the immutable bundle, validate the exact installed runtime
library and GPU path with `mtt-runtime-smoke` as described in
[Acceptance gates](acceptance.md). `mtt-validate-bundle` proves the bundle
snapshot and runtime fingerprint; `mtt-runtime-smoke` additionally executes
session golden generation and drains a normal streaming request. Neither check
substitutes a different engine, backend, or model when it fails.

For an installation receipt, pass
`--receipt-json /absolute/new/runtime-smoke-receipt.json`. The destination
must not already exist. The resulting
`magpie-tts-rt.runtime-smoke.v1` JSON records the authenticated manifest,
tokenizer identity, canonical 4/8/tail stream contract, final counters, TTFA,
RTF, and clean native teardown. Persist this JSON itself (or its digest);
do not scrape the human-readable `key=value` summary as an acceptance
contract.

After that deployment gate passes, run the same installed library and bundle
through `mtt-runtime-benchmark` with the pinned prepared-token Japanese
corpus. Its immutable `magpie-tts-rt.acceptance-benchmark.v2` receipt is the
native Gate 4 evidence for cold/warm prepared-token TTFA, generation and total
RTF, chunk cadence, three cancellation injection points, and the separate
108-case quality and 1000-turn durability gates. It reports benchmark-process
RSS/CPU and device-wide CUDA-memory trends without mislabeling either as
process-isolated VRAM or physical playback. Only process RSS is memory-gated:
growth from the post-warm baseline is capped at 64 MiB and the 1000-sample
linear slope at 64 KiB/iteration; device-wide CUDA memory remains
observational. The Thor workflow also retains
checksummed nsys and tegrastats companion evidence. The exact command, corpus
contract, metric definitions, scope exclusions, and release thresholds are
specified in
[Acceptance gates](acceptance.md). A smoke receipt does not substitute for
the benchmark receipt, and a benchmark against a different library or bundle
path is not evidence for the packaged deployment.

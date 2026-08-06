# Sofia Thor runtime package specification

[`runtime-bundle-package-spec-v1.json`](runtime-bundle-package-spec-v1.json)
is the reviewed, non-engine specification for the first Sofia AGX Thor bundle.
It is a package input, not an acceptance receipt. The packager still requires
the accepted consolidated export receipt, seven authenticated plans, plugin,
frontend receipt, complete-generation golden evidence, and the canonical
eight-file license/notice inventory.

The values are tied to the following evidence:

| Package field | Reviewed source |
|---|---|
| model ID, version, immutable revision, and model SHA-256 | [`oracle-lock.json`](oracle-lock.json), version `v2607`, revision `5023df68bd3f5b5ce6d666a50979bc501af145cc`, SHA-256 `ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953` |
| export source revision | NeMo oracle source base revision `9ae3e66b7314b0358c96bce47fbac56d78728bcd` in [`oracle-lock.json`](oracle-lock.json) |
| Sofia context | canonical fixture `magpie-v2607-sofia-ja-boundaries-v1`, manifest SHA-256 `0ca05d9b613aa4b3923ded357a812260da91cb2f1d60f04de6e57ba3f6b8004c`; tensor `cfg.sofia_prefix`, `[1,217,768]` BF16, SHA-256 `aee360ccdeeb96a145efbce176ffe5d444215bdd5bc1a7bc8af8744df53d7b8d` |
| tokenizer identity | [`japanese-source-spans-v1.json`](../tests/frontend/fixtures/japanese-source-spans-v1.json), canonical identity SHA-256 `abfe10eadb6900f68a7bd65f8f63e593d6ad0e967bb3d3532d3674960f0c57ad` |
| CFG and Main Decoder state | accepted source config plus the exact Text→Main→Local sequence oracle: conditional row first, Sofia prefix 217, 250 generated steps, absolute step positions `[218,467)` |
| alignment controller | [`alignment_controller.py`](../tools/export/alignment_controller.py): layers `[4,5,8,9]`, epsilon `0.1`, initial position `1`, terminal tokens `3`, short-text boundary `5`, lookahead `6`, sink threshold `4` |
| sampling | [`local_ar_wrapper.py`](../tools/export/local_ar_wrapper.py) and the authenticated plugin constants: top-k `80`, temperature `0.6`, codebook IDs `0..2015`, `AUDIO_EOS=2017`, special IDs `2016` and `2018..2023` forbidden |
| Local AR position table | accepted Local AR receipt: source shape `[18,768]` BF16, rows `0..15`, complete table SHA-256 `1db63ebd4ceffba52e03cf67c9d186f3b7e38bb0c6eb9056a93ceeb55a4a695e` |
| codec | accepted NanoCodec contract: mono FP32 at 22,050 Hz, 1,024 samples/frame, stateful `4/8/tail 1..8` schedule; EOS is not audio, and a zero-frame finish skips NanoCodec and emits a `FINAL` control marker |
| request limits | v1 runtime contract: 512 text tokens, 250 decoder steps, 500 audio frames, one session, one active request, eight-frame PCM ring |
| memory limits | release admission policy: 4 GiB workspace and 16 GiB total per-session device allocation; actual engine requirements are introspected and must fit before a session is admitted |
| legal inventory | repository `LICENSE`/`NOTICE`, PyTorch BSD, CUTLASS BSD, CUDA EULA plus static-runtime notice, and NVIDIA Open Model License plus required model notice |

The tokenizer has 3,357 normal rows and 3,359 text-embedding rows. BOS is
3,357, EOS is 3,358, and the Japanese batch-pad row is 1,015. Prepared input
contains only normal rows before one terminal EOS; BOS is never a prepared
input row. The receipt identity projection retains the key
`vocabulary_size=3357`, while the runtime manifest spells the two independent
dimensions `tokenizer_vocabulary_size=3357` and
`text_embedding_rows=3359`.

The plugin artifact's logical name is `magpie_tts_rt_plugins`; the Local AR
sampling creator selected by the engine is `MagpieLocalARSampling`. They are
different authenticated fields and must not be equated.

The `sampling.eos_token_id` and ordered
`sampling.forbidden_token_ids` are canonical constants. They are not
descriptive labels that a package author may change. The JSON schema, Python
packager, and C++ manifest parser all reject any other values. This keeps the
manifest consistent with the plugin path, which permits all codec IDs
`0..2015`, reserves `2017` for EOS, and masks exactly the remaining seven
special IDs.

The codec fields `eos_frame_is_audio=false` and
`zero_frame_finalization=control_marker_without_codec_invocation` close the
boundary between Local AR and NanoCodec. The tail plan accepts only one through
eight real audio frames. If EOS is the first position of a generated pair,
there is no tail invocation; the runtime emits a zero-frame `FINAL` control
marker after preceding PCM has been published. The downstream playback layer
then waits for physical-device drain. No fake frame or EOS codec row is
permitted.

`sampling.rng.seed_bits=64` describes the TensorRT/CUDA INT64 seed binding.
The public C ABI separately validates the caller's seed as a uint32 value and
rejects values outside `[0,2^32)` without truncation.

The package output remains unpublished until the consolidated receipt has
status `accepted` and the packager verifies the exact staged artifacts. The
specification contains no candidate-plan digest and cannot promote a
measured-only export. A missing, reordered, aliased, or byte-modified legal
artifact is treated exactly like a missing or modified engine: publication
and native bundle loading both fail closed.

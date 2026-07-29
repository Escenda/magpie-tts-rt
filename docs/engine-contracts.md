# TensorRT engine contracts

This document fixes the initial v2607/Sofia engine partition. Exact tensor
names in serialized plans must match the bundle manifest; positional binding
is not accepted.

## Fixed model values

| Value | Contract |
| --- | --- |
| application batch | `B = 1` |
| CFG decoder batch | `Be = 2` |
| model width | `768` |
| Text Encoder | 6 layers, 12 heads |
| Main Decoder | 12 layers, 12 self-attention heads |
| self-attention head width | `64` |
| cross-attention | 1 head of width `128` |
| Local AR | 2 layers, 12 heads |
| audio codebooks | `8` |
| frame stacking | `2` |
| Local AR positions per Main Decoder step | `16` |
| codec tokens | `2016` base + `8` special |
| Local AR vocabulary | `2024` |
| sampling | temperature `0.6`, top-k `80` |
| codec output | mono FP32, 22,050 Hz |
| samples per codec frame | `1,024` |

Text Encoder, Main Decoder, and Local AR execute in BF16. NanoCodec remains
FP32. Token/code tensors are INT64 and masks are BOOL.

## Optimization profiles

Schema version 1 accepts one named profile per engine:

| Engine role | Profile | Dynamic range |
| --- | --- | --- |
| Text Encoder | `text_1_512` | `T`: min 1, opt 64, max 512 |
| Main Decoder prefill | `text_1_512` | the same `T` range |
| Main Decoder one-step | `text_1_512` | alignment and all cross K/V use the same `T` range |
| Local AR | `fixed` | no dynamic input |
| NanoCodec initial 4 | `fixed` | no dynamic input |
| NanoCodec steady 8 | `fixed` | no dynamic input |
| NanoCodec tail | `tail_1_8` | `F`: min 1, opt 4, max 8 |

Different ranges, extra profiles, or a profile that omits one dynamic input
are manifest errors. The text maximum also equals
`limits.maximum_text_tokens=512`.

## Text Encoder

The Text Encoder runs once per input text chunk.

Inputs:

- `text_token_ids`: `[1, T] INT64`
- `text_mask`: `[1, T] BOOL`

Output:

- `text_condition`: `[1, T, 768] BF16`

The C++ runtime constructs the two CFG rows required by Main Decoder. Reading
conversion and tokenization are outside this engine.

## Main Decoder prefill

For the accepted Sofia bundle, the baked speaker prefix has length `P = 217`.
The prefix, its mask, and `AUDIO_BOS` are immutable constants in the
voice-specific prefill plan. Version 1 selects a voice by loading a different
verified model bundle; it does not expose a speaker index.

Classifier-free guidance has exactly two rows. Row 0 is conditional and copies
the Text Encoder output and `text_mask`. Row 1 is unconditional: its condition
is all zero and its mask is all false. The manifest records this order and
source policy; reversing rows or inventing another unconditional input is an
error.

Inputs:

- `condition`: `[2, T, 768] BF16`
- `condition_mask`: `[2, T] BOOL`

Outputs:

- `last_hidden`: `[2, 1, 768] BF16`
- `alignment`: `[2, T] BF16`
- 12 self K and 12 self V cache tensors with capacity
  `[2, P + 250, 12, 64] BF16`
- 12 self key masks with capacity `[2, P + 250] BOOL`
- 12 cross K and 12 cross V tensors `[2, T, 1, 128] BF16`

The initial accepted profile therefore has a self-cache capacity of 467
positions. Cache capacity is a required manifest value and is never inferred
from a plan at request time.

Alignment is the declared reduction of decoder layers 4, 5, 8, and 9.

## Main Decoder one-step

Inputs:

- previous codec tokens: `[1, 8, 2] INT64`
- `position`: scalar INT64 containing the absolute self-cache write index;
- current self K/V caches and key masks;
- prefilled cross K/V tensors;
- dynamic attention prior: `[2, 1, T] BF16`.

Outputs:

- `decoder_hidden`: `[2, 768] BF16`
- `alignment`: `[2, T] BF16`
- successfully updated self K/V and key-mask state.

The engine includes the 16 codec-token embedding gathers, their average, and
CFG-row construction. KV writes occur at the explicit input position. Layers
2 through 10 apply the dynamic prior. Alignment extraction remains limited to
the declared layers.

Prefill consumes the fixed 217-position Sofia prefix plus `AUDIO_BOS` and
therefore leaves 218 occupied cache positions. One-step invocation `n`, with
`n=0` for the first invocation after prefill, writes position `218+n`. Valid
positions are `[218, 467)`. Prefill produces the first generated decoder step;
the 249 one-step positions plus prefill give the declared maximum of 250
generated steps. No relative-position interpretation is accepted.

## Local AR and sampling

The Local AR engine is statically unrolled across `8 × 2 = 16` positions.

Inputs:

- `decoder_hidden`: `[2, 768] BF16`
- `unfinished`: `[1] BOOL`
- `finished`: `[1] BOOL`
- `forbid_eos`: `[1] BOOL`
- `rng_seed`: `[1] INT64`
- `rng_counter`: `[1] INT64`.

Outputs:

- codec tokens: `[1, 8, 2] INT64`
- updated RNG counter: `[1] INT64`
- invalid-row status: `[1] INT32`
- first EOS frame index: `[1] INT32`.

`invalid_rows` is a bitmask over the fixed CFG row order. Bit 0 represents
the conditional row and bit 1 the unconditional row. `0` means both rows
produced valid sampling distributions; any nonzero value fails the request
before codec tokens or embeddings are consumed.

`end_frame_index` is `0` or `1` when EOS occurs in the corresponding generated
frame. `-1` is the only no-EOS sentinel. When `forbid_eos` is true, the output
must be `-1`.

Each position runs the two-layer Local Transformer and its position-specific
`768 → 2024` projection. A TensorRT `IPluginV3` operation performs CFG
combination, forbidden-token handling, top-k selection, temperature scaling,
Gumbel sampling, RNG-counter advancement, and the next embedding gather.

Local K/V scratch has shape `[2, 17, 12, 64] BF16` per layer. It is reset for
every Main Decoder step and is not session history.

### Unresolved deterministic-sampling gate

The engine partition and tensor shapes above are fixed. The bit-exact
sampling algorithm is not yet fixed well enough to implement. Before the
Local AR engine or sampling plugin is accepted, the pinned PyTorch oracle must
provide fixtures that define:

- conversion of the public `uint64_t` seed to the signed INT64 engine
  representation;
- Philox key and counter lane layout;
- counter consumption for every Local AR position and both CFG rows;
- integer-to-uniform conversion, endpoint handling, and the exact Gumbel
  transform;
- tie-breaking for equal logits at the top-k boundary;
- EOS reduction and counter behavior on forbidden, sampled, and invalid
  distributions; and
- every persistent NanoCodec state tensor, including update order at
  initial, steady, and terminal boundaries.

The exporter source snapshot, patch set, and sanitized golden boundary
fixtures are release inputs, not optional documentation. Until they are
committed or published in an authenticated release, model loading and
synthesis remain unavailable.

## EOS

With Local AR enabled, audio codes come from Local AR rather than the Main
Decoder's 32,384-wide projection.

EOS evaluation is skipped while EOS is forbidden. Once enabled, a fused
operation consumes:

- `decoder_hidden`;
- the final projection weights;
- sampled codec tokens `[1, 8, 2]`.

It performs the projection and reduction without exposing a
`[2, 1, 32384]` intermediate outside the engine. The output is the first EOS
frame index in the two generated frames, using the exact `0`, `1`, and `-1`
encoding defined above.

## Alignment controller

The alignment controller is C++ session logic, not another neural engine. It
consumes the selected-layer alignment and maintains:

- the last committed text-token position;
- attention-sink and monotonicity state;
- the look-ahead window;
- the dynamic prior for the next Main Decoder step;
- text-end proximity and EOS permission;
- text-chunk history.

The controller must not invent source-character positions. An application may
map committed token positions to source spans only when it has an exact
frontend span map.

## Stateful NanoCodec

Input:

- codec tokens: `[1, 8, F] INT64`, where an accepted route defines `F`.

Output:

- PCM: `[1, F × 1024] FP32`
- valid sample length: `[1] INT64`.

The initial routes are:

- first: fixed `F = 4`;
- steady: fixed `F = 8`;
- terminal tail: `F = 1..8`.

The session owns every causal Conv1d input history, ConvTranspose1d pending
overlap, residual-branch history, and FP32 work buffer. Standard TensorRT
Conv/ConvTranspose operations perform the main computation; plugins may update
explicit causal histories and overlap state.

The initial 4-frame plan has no causal-state input. It creates the deterministic
initial state inside the verified plan and returns every declared state output.
The steady and tail plans consume and replace those explicit state tensors.
The runtime never guesses or fills a missing state tensor.

A 12-frame rolling re-decode route is not part of this contract.

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
| active attention look-ahead | `6` text tokens |
| active attention-sink threshold | `4` decoder steps |

Text Encoder, Main Decoder, and Local AR execute in BF16. NanoCodec remains
FP32. Text token IDs are INT32, codec/code and RNG tensors are INT64, and masks
are BOOL. The public C ABI accepts INT64 text IDs and the runtime range-checks
every value before converting it to the Text Encoder's INT32 input.

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

- `text_token_ids`: `[1, T] INT32`
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
is all zero, while its mask keeps only text position zero true and sets every
later position false. This is the exact NeMo `prepare_dummy_cond_for_cfg`
contract; an all-false mask is not equivalent. The manifest records this order
and source policy. Reversing rows or inventing another unconditional input is
an error.

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
- `position`: DEVICE scalar INT64 containing the absolute self-cache write
  index;
- `execution_status_in`: DEVICE scalar INT32 carrying the sticky status from
  the previous cache direction;
- `condition_mask`: `[2, T] BOOL`, retained unchanged from prefill;
- current self K/V caches and key masks;
- prefilled cross K/V tensors;
- dynamic attention prior: `[2, 1, T] BF16`.

Outputs:

- `decoder_hidden`: `[2, 768] BF16`
- `alignment`: `[2, T] BF16`
- `execution_status_out`: DEVICE scalar INT32 carrying the first mode-8
  execution error seen by this invocation;
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

`position` is an ordinary TensorRT execution input: its manifest fields are
`location: device` and `shape_inference_io: false`. It has no optimization
profile value range. A plan or bundle that exposes `position` as a HOST shape
input belongs to the superseded contract and is rejected rather than adapted.

`execution_status_in` and `execution_status_out` are ordinary TensorRT
execution I/O with scalar shape, `location: device`, and
`shape_inference_io: false`. A-to-B consumes the preceding B-to-A status and
B-to-A consumes the preceding A-to-B status. The runtime initializes the first
input to zero, then checks each output before starting the following Local AR
invocation. A nonzero value is never cleared by a later layer or cache
direction.

Status bit 31 is zero. Bits 30 through 28 contain the category (`1` invalid
active K, `2` CUDA graph-update failure), bits 27 through 24 contain the Main
Decoder layer index, bits 23 through 22 contain the operation (`0` selector,
`1` QK, `2` PV), and bits 21 through 0 contain the exact detail value. Invalid
K uses detail zero; CUDA failures retain the exact CUDA status. Every one of
the 12 mode-8 plugin instances has an immutable `layer_index` field matching
its decoder-layer order. A status-less plan, a non-scalar status binding, or a
HOST status binding is rejected rather than adapted.

The mode-8 plugin publishes one immutable launch-class table after discovery:
7 QK classes, 14 PV classes, and one QK/PV class-and-grid mapping for every
active K from 219 through 467. Its identity JSON contains ASCII kernel names,
ASCII operation/transport enums, and integers only. The SHA-256 payload is
exactly compact JSON with lexicographically sorted object keys
(`ensure_ascii=true`, separators `,` and `:`), encoded as ASCII with no
trailing newline. Pointer values and opaque cuBLAS parameter bytes are never
part of this identity.

At each request boundary the runtime initializes `position` to `217`. Every
one-step invocation increments that device scalar before TensorRT executes, so
the plan observes `218, 219, ...` without a HOST shape-input ring or a
per-step H2D copy. The increment is the first node in both Main Decoder CUDA
Graphs. The same increment precedes the one eager warmup execution for each
cache direction, which keeps eager, captured, and replayed position semantics
identical.

Main Decoder owns separate A-to-B and B-to-A TensorRT contexts. The first
A-to-B eager result is retained on the first-audio path. After first audio is
published, the first B-to-A eager result is also retained. The next invocation
of each direction is captured and immediately launched; direct enqueue is not
accepted after that direction's single warmup. Because the plan retains
request-length `T` in its mask, prior, and cross K/V tensors, both graphs are
destroyed and recaptured for every request. There is no startup-graph or
same-length reuse branch.

## Local AR and sampling

The Local AR engine is statically unrolled across `8 × 2 = 16` positions.
Its fixed tensor shapes permit one captured CUDA Graph. The runtime binds that
graph to one canonical `unfinished` input, one canonical invalid-row output,
one canonical EOS output, and fixed RNG counter input/output addresses.
Logical-step diagnostics are separate storage and are populated only by the
ordered post-graph commit kernel, so a later Local invocation cannot overwrite
an earlier step before batch diagnostics are copied to the host.

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

`invalid_rows[0]` is zero only when the guided distribution is finite and has
at least one eligible token. Any nonzero value fails the request before codec
tokens or embeddings are consumed. CFG is combined before sampling, so it does
not consume a separate random number for the unconditional row.

`end_frame_index` is `0` or `1` when EOS occurs in the corresponding generated
frame. `-1` is the only no-EOS sentinel. When `forbid_eos` is true, the output
must be `-1`.

An EOS position is a control position, not a codec-audio frame. Therefore
`end_frame_index=0` contributes zero new audio frames, and
`end_frame_index=1` contributes only frame zero. The runtime must never pass
the EOS position to NanoCodec.

Each position runs the two-layer Local Transformer and its position-specific
`768 → 2024` projection. A TensorRT `IPluginV3` operation performs CFG
combination, forbidden-token handling, top-k selection, temperature scaling,
Gumbel sampling, RNG-counter advancement, and the next embedding gather.

The Text Encoder, Main Decoder, and Local Transformer use authenticated
`IPluginV3` operations at BF16 boundaries where TensorRT's native lowering
does not preserve the accepted PyTorch 2.11 reduction and rounding order.
`MagpieLayerNorm`, `MagpieGeluTanh`, and mode 0 of `MagpieSoftmax` cover their
declared Text, Main, and Local tensor shapes. The same `MagpieSoftmax` creator
also has five Main-Decoder-only modes: mode 1 computes masked cross-attention
probabilities, mode 2 computes the prefill or one-step self-attention
probability/value product, mode 3 computes the prefill or one-step cross-attention
probability/value product, and mode 4 performs the one-step dynamic-prior
renormalization division. Mode 5 computes the one-step self-attention
`[2,12,1,64] × [2,12,64,467]` QK product with BF16 inputs and FP32
accumulation. Mode 4 accepts a `[2,1,1,T]` BF16 numerator and
`[2,1,1,1]` BF16 denominator and rounds each quotient directly to BF16; a
TensorRT elementwise division is not equivalent at this boundary. Text length
`T` is always in `[1,512]`. Small BF16 differences are not accepted because
they can alter alignment, later decoder state, and sampled codec tokens. The
plugin ABI v1 therefore authenticates exactly five creators, in order:
`MagpieLocalARSampling`, `MagpieLocalAREos`, `MagpieLayerNorm`,
`MagpieGeluTanh`, and `MagpieSoftmax`, all at version `1` in namespace
`magpie_tts_rt`.

Before the two Local Transformer layers, position `p` adds learned absolute
position-embedding row `p`, for every `p` in `[0, 16)`. The accepted model does
not use a zero, sinusoidal, or relative-position substitute here. The engine
must contain the 16 BF16 rows selected from the model's `[18, 768]` local
position-embedding table, and the export receipt records the complete source
table digest. Omitting this addition changes the locked codec-token output.

Local K/V scratch has shape `[2, 17, 12, 64] BF16` per layer. It is reset for
every Main Decoder step and is not session history.

### Deterministic sampling

The accepted oracle fixes the sampling algorithm:

1. Public seeds are accepted only in `[0, 2^32)`. No truncation or modulo
   conversion is allowed.
2. One signed INT64 counter exists per application batch row. The counter
   starts at zero and advances exactly once at each of the 16 Local AR
   positions, including a position that later causes EOS.
3. The random offset for vocabulary element `v` is
   `counter * 2048 + v`.
4. The Philox seed for row `r` is the low 32 bits of
   `seed + r * 0x9E3779B9`.
5. Uniform values are clamped to `[0.00000006, 0.99999994]`; the Gumbel value
   is `-log(-log(u))`.
6. The top-k boundary is the 80th greatest guided logit. Every value greater
   than or equal to that boundary is eligible. This inclusive rule is
   intentional when logits tie.
7. The chosen token is the left-most argmax of
   `guided_logit / 0.6 + gumbel` over eligible tokens.

The conditional BF16 logits are multiplied by `2.5` and rounded to BF16. The
unconditional logits are then added with scale `-1.5` and rounded to BF16
again before conversion to FP32 for sampling. Replacing those two roundings
with one FP32 expression is not parity.

Codec tokens are exactly `0..2015`. `AUDIO_EOS=2017` is the only eligible
special token; the static forbidden set is exactly
`[2016,2018,2019,2020,2021,2022,2023]`. EOS is additionally forbidden while
the first four codec frames are generated or while alignment marks the text
unfinished. A finished row forces EOS. NaN, infinity, or an all-forbidden
distribution fails the request. The manifest records this exact partition
and rejects any alternate IDs.

The locked boundary fixture records the initial seed, counter zero, the first
16-position output, and counter 16. Plugin tests must additionally cover
top-k ties, the two clamp endpoints, invalid distributions, forced EOS, and
counter overflow.

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

For the active Sofia configuration, the controller starts at text position 1,
searches `[last + sink_advance, min(start + 6, text_length - 3))`, and chooses
the left-most maximum alignment score. A position advances by one before the
search when its counter has reached 4. The next conditional prior starts at
`0.1`, writes `1.0` at one history position, the attended position, and six
look-ahead positions, then suppresses every position through the greatest
attention sink back to `0.1`. Texts of five tokens or fewer receive an all-one
conditional prior. The unconditional CFG row remains `0.1`.

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

The tail engine has no zero-frame profile. When EOS leaves zero undecoded audio
frames, the runtime does not enqueue any NanoCodec engine. It emits a
zero-frame `FINAL` control marker after all preceding PCM has been published.
This marker carries no PCM and does not advance codec state. The downstream
playback layer, not the synthesis runtime, waits for physical audio drain.
Calling
the tail engine with a fabricated frame, treating EOS as audio, or padding the
request to one frame is a contract violation.

The session owns every causal Conv1d input history, ConvTranspose1d pending
overlap, residual-branch history, and FP32 work buffer. Standard TensorRT
Conv/ConvTranspose operations perform the main computation; plugins may update
explicit causal histories and overlap state.

The accepted PyTorch implementation exposes 189 FP32 state and work tensors,
approximately 57.30 MiB per active decoder. Only about 0.713 MiB is persistent
causal session state: exactly 92 convolution input histories and five
pending-overlap tensors. These 97 tensors are individually named in the
canonical registry in `tools/export/nanocodec_contract.py`. The remaining 92
concatenation/work buffers belong to the execution-context workspace and are
not state bindings. Concurrent execution contexts must not share that mutable
workspace.

The binding names are deterministic. Persistent inputs are
`state_in.<logical_name>` and replacement outputs are
`state_out.<logical_name>`. The initial-4 engine therefore has one input and
99 outputs; steady-8 and tail-1-through-8 each have 98 inputs and 99 outputs.
An opaque aggregate, a reordered registry, or a runtime-inferred state is not
accepted.

The decoder stages are
`32 → 864 → 432 → 216 → 108 → 54 → 27` with upsampling rates
`8, 8, 4, 2, 2`. Each of the five residual stages has three kernel branches
(`3, 7, 11`), three dilations (`1, 3, 5`), and two causal convolutions per
residual block. The exporter must enumerate these states by name; the C++
runtime must not infer or omit them.

Each transposed-convolution overlap is added to the beginning of the next PCM
chunk. The stored overlap has that layer's bias subtracted so the next call
does not count the bias twice. This update order is part of codec parity.

The initial 4-frame plan has no causal-state input. It creates the deterministic
initial state inside the verified plan and returns every declared state output.
The steady and tail plans consume and replace those explicit state tensors.
The runtime never guesses or fills a missing state tensor.

The fixed initial-4 execution and both fixed steady-8 directions are mandatory
CUDA Graph routes. Initial-4 writes state A. Steady A-to-B and steady B-to-A
use separate TensorRT execution contexts and separate immutable graph
executables; one context must never be captured into both graphs. The runtime
performs one discarded startup warmup for each route, captures it with its
production tensor addresses, and then admits only graph replay output. A
missing, failed, or unaccounted fixed-route graph closes the startup gate and
has no eager-enqueue fallback.

Tail-1-through-8 remains the authenticated dynamic-shape TensorRT route. That
direct enqueue is valid only for a terminal batch and cannot substitute for a
missing initial or steady graph. Before workspace allocation, the runtime adds
the second steady context's `getDeviceMemorySizeV2()` to explicit session
context memory. CUDA Graph executable memory is measured independently using
current and high-water device graph attributes after startup replay; both
amounts must fit `maximum_device_memory_bytes`.

A 12-frame rolling re-decode route is not part of this contract.

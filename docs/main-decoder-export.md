# Main Decoder export

`tools/export/export_main_decoder.py` exports the Sofia prefill and one-step
graphs from the locked PyTorch oracle. It also builds both TensorRT plans,
checks every plan binding and optimization profile, and measures the plans
against the boundary fixture.

The tool does not declare TensorRT numerical equivalence. Its receipt is
`measured-not-accepted` until tolerances are justified with multiple Japanese
fixtures, closed-loop generation, Local AR token/EOS equality, and sequence
tests. A successful ONNX parse or plan build is not acceptance.

## Prefill

Prefill has two inputs:

- `condition`: `[2,T,768] BF16`;
- `condition_mask`: `[2,T] BOOL`.

CFG row 0 is conditional. CFG row 1 is unconditional. The accepted oracle
does not make the unconditional mask entirely false: position zero is true
and every later position is false. Both fixed 217-position speaker-prefix
masks are entirely true.

The fixed plan constants are the Sofia speaker embedding, its unconditional
zero embedding, both prefix masks, and the averaged `AUDIO_BOS` embedding.
Prefill occupies positions 0 through 217. It returns the hidden state and
selected-layer alignment for position 217, twelve capacity-467 self K/V/mask
sets, and twelve text-length cross K/V sets.

## One-step

One-step has these non-layer inputs:

- `previous_codec_tokens`: `[1,8,2] INT64`;
- `position`: scalar INT64;
- `alignment_prior`: `[2,1,T] BF16`;
- `condition_mask`: `[2,T] BOOL`.

Each layer also receives capacity-467 self K/V/mask and `[2,T,1,128]` cross
K/V. Invocation zero writes absolute position 218. Relative positions are not
accepted.

The graph performs all sixteen embedding gathers and averages them before
duplicating the result into the two CFG rows. It applies the dynamic prior to
layers 2 through 10 and averages alignment from layers 4, 5, 8, and 9. It
returns `[2,768]` hidden state, `[2,T]` alignment, and all twelve updated
self-cache sets.

Dynamic-prior renormalization is an authenticated BF16 boundary. After
cross-attention probabilities are multiplied by the prior, the numerator
`[2,1,1,T]` is divided by its `[2,1,1,1]` sum with
`MagpieSoftmax` mode 4. Each quotient is converted directly to BF16 using the
accepted CUDA rounding path. TensorRT's elementwise division differs by BF16
bits for locked one-step inputs and is therefore not an allowed substitute.

The one-step self-attention probability/value product is also an authenticated
boundary. `MagpieSoftmax` mode 6 executes
`[2,12,1,K] × [2,12,K,64]` after copying the active value-cache prefix
to a linear buffer. The probability tensor is already linear from the active
softmax and therefore retains leading dimension and batch stride `K`.
Restoring a capacity-467 probability stride changes cuBLAS results for some
accepted inputs and is not allowed.

The preceding one-step self-attention QK product is authenticated separately.
`MagpieSoftmax` mode 5 executes
`[2,12,1,64] × [2,12,64,K]` through cuBLAS after the key cache has
been sliced to the active `K = position + 1` prefix. Both inputs are linear,
the leading dimension and batch stride use the active `K`, and the call uses
BF16 inputs, FP32 accumulation, BF16 output, and
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`. The cuBLAS handle belongs to the attached
TensorRT plugin context and is rebound to the enqueue stream for each call.
Keeping the 467-position physical stride changes the selected cuBLAS path at
some active lengths and is not an allowed substitute.

Prefill cross-attention context products use `MagpieSoftmax` mode 3 through
the same per-context cuBLAS execution contract. The one-step shape retains its
accepted fixed CUTLASS kernel. This split preserves the PyTorch 2.11 BF16
boundary for all locked Japanese text lengths; selecting one implementation
for both shapes is not bit-exact.

`condition_mask` must remain an explicit one-step input. Cross K/V does not
encode which text positions are valid. Reconstructing the mask from `T` would
change the meaning of padded inputs.

## Alignment controller boundary

Look-ahead 6 and attention-sink threshold 4 belong to the C++ alignment
controller. The neural one-step plan consumes the controller's prior; it does
not update monotonic position, sink counters, or the next prior. This keeps
the session decision state explicit and testable.

## Manifest contract

The runtime manifest records the verified oracle behavior directly:

- the unconditional condition mask is true only at position zero;
- `condition_mask` is a required Main Decoder one-step input.
- the alignment controller constants are `prior_epsilon=0.1`,
  `initial_attended=1`, `ignored_terminal_tokens=3`,
  `short_text_no_prior_max_tokens=5`, `lookahead=6`, and
  `sink_threshold=4`.

The exporter does not add a compatibility path for the superseded contract.

## Reproducible gate

The exporter requires explicit paths for the oracle lock, NeMo source, model,
codec, acceptance receipt, boundary fixture, `trtexec`, TensorRT Python
package, and output directory. It then:

1. verifies every locked input and all fixture bytes;
2. requires bit-exact PyTorch parity for every declared output and cache;
3. exports standard-domain, inline-weight ONNX;
4. builds strongly typed, no-TF32 plans for `T=1/64/512` with detailed
   profiling metadata;
5. validates plan names, dtypes, shapes, and profiles;
6. runs TensorRT prefill and feeds its own K/V directly into TensorRT step;
7. records max, mean, p99, cosine similarity, and BF16 bit mismatches by
   output group;
8. writes raw prefill and step hidden tensors for Local AR semantic tests;
9. publishes the directory atomically without replacing an existing result.

The plan-parity receipt stays unaccepted even when every mechanical step
succeeds. The remaining acceptance evidence is tracked by the component and
sequence gates in `docs/acceptance.md`.

## Same-plan multi-fixture measurement

`tools/export/validate_main_decoder_plans.py` authenticates an existing export
receipt and every artifact before running the plans. It requires at least two
distinct, locked Japanese fixtures and refuses a GPU, CUDA, or TensorRT
fingerprint different from the export host.

The tool does not rebuild either plan. For every fixture it runs TensorRT
prefill, feeds TensorRT's own K/V into one-step, writes the two hidden tensors,
and records per-output numerical metrics. It also requires the host alignment
controller to reproduce the fixture's prior, attended position, and counter
bit-for-bit. The prior selected from TensorRT's measured alignment must make
the same discrete decision as the oracle alignment. Its receipt also contains
the worst observed metric envelope across fixtures. The result remains
`measured-not-accepted`: the envelope is evidence for a future predeclared
tolerance, not a tolerance inferred and accepted by the same run.

## Closed-loop code and EOS accounting

`tools/export/sequence_contract.py` defines the fail-closed state used by the
closed-loop validator. One Local AR invocation must advance the RNG counter by
exactly 16, return no invalid row, and report `end_frame_index` as `-1`, `0`,
or `1`. EOS is forbidden for decoder steps zero and one because those steps
cover the required first four frames.

An EOS index is the first invalid frame in the two-frame stack. Index zero
therefore retains no frame from that invocation; index one retains only frame
zero. The comparison reports the first differing frame, codebook, and token,
along with complete generated and expected code hashes. It never converts a
token difference into a numerical tolerance.

`tools/export/validate_main_local_ar_sequence.py` applies that state to at
least three independently captured, predeclared Japanese fixtures. It
authenticates every Text Encoder, Main Decoder, and Local AR artifact, requires
one oracle lock and canonical fixture binding, and reuses each TensorRT engine
and execution context for the whole run. Text Encoder output is connected
directly to Main Decoder; the fixture condition is used only for numerical
diagnostics. For every fixture the validator records Text Encoder metrics,
generated codes, the attended-position trace, EOS step/frame, final RNG
counter, and the first code mismatch.

The sequence receipt is `accepted` only when every fixture reproduces the
complete code tensor exactly. Exact tensor shape includes the EOS boundary,
while the fail-closed tracker independently enforces contiguous RNG counters,
valid EOS indices, minimum generated frames, and monotonic alignment. Any code
or EOS difference keeps the receipt `measured-not-accepted`; it is retained as
diagnostic evidence and is never converted into a numerical tolerance.

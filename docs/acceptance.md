# Acceptance gates

Passing compilation, engine build, or one audible sample is not acceptance.
Each gate produces a machine-readable receipt containing source, model,
exporter, engine, plugin, runtime, hardware, and test-corpus hashes.

Release tags must match the CMake and both Cargo package versions, identify a
commit already contained in `main`, and pass the complete AGX Thor workflow.
The workflow emits a checksummed archive only after the contract, schema,
generated-binding, sanitizer, packaging, installed-consumer, native ABI, and
Thor GPU checks pass. Artifact signing, attestation, and automated GitHub
Release publication remain blockers for a production release.

## Gate 0: bundle integrity

- strict manifest parses with no unknown or missing fields;
- every asset SHA-256 and exact byte length match;
- the declared artifact lengths exactly equal the bounded bundle snapshot
  budget;
- operating system, architecture, endianness, CUDA, TensorRT, driver, GPU
  compute capability, and plugin ABI match;
- every deserialized plan reports the exact tensor names, dtypes, shapes, and
  profiles declared by the manifest;
- the bundle's golden startup case passes before the model becomes available.

Any failure prevents session creation.

## Gate 1: component equivalence

For Text Encoder, prefill, one-step, Local AR, EOS, and NanoCodec:

- compare every declared output and persistent state against the pinned
  PyTorch oracle;
- predeclare absolute/relative tolerances by tensor and dtype;
- test minimum, optimum, and maximum text/profile sizes;
- test EOS-forbidden, EOS-enabled, and terminal-tail routes;
- verify RNG counter advancement and sampling against fixed logit fixtures;
- verify that failed execution does not advance persistent state.

Numerical differences are reported; they are not converted into a silent
acceptance fallback.

## Gate 2: sequence equivalence

Run complete generation with fixed seeds across Japanese cases covering:

- short fillers and one-sentence answers;
- punctuation and sentence boundaries;
- numbers, units, Latin abbreviations, and required reading conversion;
- long text and maximum accepted context;
- early, normal, and maximum-length EOS;
- every terminal codec tail length from 1 through 8;
- repeated session creation, cancellation, and restart.

Record generated code hashes, alignment traces, EOS steps, and PCM chunk
boundaries. Any token divergence must be localized to the first differing
engine boundary before audio-quality review.

## Gate 3: audio and alignment

- PCM is finite and has exactly `codec_frames × 1024` samples;
- chunk sequence and absolute sample ranges are contiguous;
- stateful chunks contain no boundary seam relative to the reference;
- ASR character error rate and human listening quality do not regress;
- committed alignment is monotonic;
- token-to-source progress is published only when an exact frontend span map
  exists;
- cancellation publishes no post-cancel PCM.

## Gate 4: performance on AGX Thor

Measure cold and warm runs separately. Report median, p95, and maximum for:

- Text Encoder and prefill latency;
- Main Decoder, Local AR, EOS, and codec latency per step/chunk;
- raw time to first audio;
- generation and total real-time factor;
- positive playback lateness;
- CUDA allocated/reserved memory;
- kernel launches, host submissions, and blocking synchronizations.

The initial PyTorch oracle measured four cases with:

- raw first-audio median `136.03 ms`, p95 `152.39 ms`;
- generation RTF p95 `0.401`;
- total RTF p95 `0.408`;
- maximum positive playback lateness `0 ms`;
- maximum peak CUDA allocation `1,416,798,720 bytes`.

These values are a regression ceiling, not the optimization objective. The
first TensorRT release additionally targets raw first-audio p95 at or below
`100 ms` and generation RTF p95 at or below `0.30` over at least 100 pinned
Japanese cases. If profiling shows the target is infeasible, the target must
be revised explicitly rather than silently weakened.

## Gate 5: C ABI and Rust

- ABI negotiation accepts only the declared major version;
- every public structure validates `struct_size` and `abi_version`;
- null, cross-parent, and double-release handles fail explicitly;
- safe-wrapper ownership prevents use of a destroyed native handle;
- parent destruction with live children returns `BUSY`;
- error data is call-local and thread-safe;
- asynchronous failure status, stage, and message remain available in every
  later terminal snapshot;
- cancellation, consumer backpressure, and lease starvation are bounded;
- lease identifiers are process-wide, strictly increasing, and never reused;
- address/undefined/thread sanitizers pass where supported;
- the Rust safe wrapper cannot outlive its native parent and does not
  implicitly cancel or block in `Drop`.

## Gate 6: application streaming

The application integration must verify:

- incremental PCM is published without waiting for the whole utterance;
- first 4-frame and steady 8-frame chunks reach playback unchanged except for
  the application's declared resampling step;
- reconnect, cancellation, barge-in, priority speech, and output flush work;
- source-position events reflect actually acknowledged playback, not merely
  synthesized PCM;
- a failed or poisoned runtime is visible and cannot be reported ready.

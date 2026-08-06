# Acceptance gates

Passing compilation, engine build, or one audible sample is not acceptance.
Each gate produces a machine-readable receipt containing source, model,
exporter, engine, plugin, runtime, hardware, and test-corpus hashes.

Release tags must match the CMake and both Cargo package versions, identify a
commit already contained in `main`, and pass the complete AGX Thor workflow.
The checked-in workflow covers the contract, schema, generated-binding,
sanitizer, installed-consumer, native ABI, Thor GPU unit-test, authenticated
seven-engine packaging, installed `mtt-runtime-smoke`, and the 108-case plus
1,000-iteration `mtt-runtime-benchmark` paths before emitting a checksummed
install archive. Every external acceptance input is an explicit required
repository variable; a missing path fails closed. Artifact signing,
attestation, and automated GitHub Release publication remain blockers for a
production release.

## Gate 0: bundle integrity

- strict manifest parses with no unknown or missing fields;
- every asset SHA-256 and exact byte length match;
- the declared artifact lengths exactly equal the bounded bundle snapshot
  budget;
- operating system, architecture, endianness, CUDA, TensorRT, driver, GPU
  compute capability, and plugin ABI match;
- every deserialized plan reports the exact tensor names, dtypes, shapes, and
  profiles declared by the manifest;
- source-model, export, and tokenizer identity receipts match their declared
  source and identity hashes; raw `.nemo` and raw-tokenizer fallbacks are not
  accepted;
- the plugin build receipt matches the exact plugin bytes, deterministic
  compiler flags, source hashes, `sm_110` target, and fixed ELF dependency
  contract;
- the plugin artifact name `magpie_tts_rt_plugins` and the TensorRT sampling
  creator `MagpieLocalARSampling` are authenticated as distinct identities;
- tokenizer normal rows, embedding rows, BOS/EOS/Japanese-pad IDs, and the
  nonempty normal-prefix plus terminal-EOS request rule agree across the
  identity receipt, manifest, and golden fixture;
- every session runs the bundle's self-contained prepared-token golden fixture
  through complete generation and matches decoder/code/PCM hashes and counts
  before it becomes ready.

Any failure prevents session creation.

The installed `mtt-runtime-smoke` executable is the deployment-side gate. It
loads the exact shared library supplied by the application, creates the
runtime/model/session through C ABI v1, and therefore executes the mandatory
startup golden. It then submits the authenticated golden input as a normal
streaming request and requires one contiguous FIRST-to-FINAL lease sequence,
finite PCM, monotonic alignment, a successful terminal snapshot, and clean
native teardown:

```bash
timeout 300s mtt-runtime-smoke \
  --library /absolute/prefix/lib/libmagpie_tts_rt.so.0.1.0 \
  --bundle /absolute/bundles/sofia \
  --manifest-sha256 <authenticated-64-lowercase-hex> \
  --cuda-device 0 \
  --receipt-json /absolute/new/runtime-smoke-receipt.json
```

Its `startup_gate_ms`, `request_ttfa_ms`, `request_total_ms`, chunk/sample
counts, and RTF are measurements of that exact deployment. A timeout, cleanup
failure, non-success terminal state, or malformed stream is a failed gate.
The gate validates the exact ABI v1 headers and reserved fields returned for
model information, request snapshots, audio leases, and alignment events. It
also requires canonical 22,050 Hz mono F32 PCM, the 4-frame initial chunk,
8-frame non-terminal chunks, a 1-8 frame decoded terminal tail or a
zero-sample non-FIRST FINAL control marker, and alignment progress that never
exceeds the submitted prepared-token count. EOS is excluded from decoded
frames. The zero marker bypasses NanoCodec, carries no alignment event, and
does not advance sample or token progress. The 1,024-sample value is a
codec-frame size, not a requirement that every audio lease contain only one
frame.

`--receipt-json` is optional and must name an absolute, nonexistent file in an
existing nonsymlink directory. The tool creates that file only after the
normal request, final snapshot, every audio release, all native handle
destruction, and `dlclose` have succeeded. It uses a same-directory durable
temporary and publishes the receipt without overwriting an existing path. A
receipt-write failure makes the command fail.

The receipt is one JSON object with
`schema_version = "magpie-tts-rt.runtime-smoke.v1"` and
`status = "accepted"`. Its required top-level members are
`completed_at_unix_ms`, `abi_version`, `inputs`, `model`,
`stream_contract`, `result`, `latency`, and `verification`.
`inputs` publishes the runtime-library basename and byte size, bundle basename,
and authenticated manifest basename, SHA-256, and byte size. It also records
the CUDA device, request timeout, prepared-token count, and seed. Host-local
absolute paths are accepted only as command-line inputs and are never part of
the public receipt.
`result.terminal_control_marker_seen` is a required boolean. It is `true` only
when the accepted request ended through the zero-sample, non-FIRST `FINAL`
control marker; a normal one-through-eight-frame decoded tail records `false`.
`verification.native_cleanup` is exactly `"passed"` only because receipt
publication occurs after clean teardown. Deployment tooling may embed or hash
this JSON, but must reject unknown schema versions, missing members, a status
other than `accepted`, or any verification value other than `passed`.

`--help` and `-h` print usage and return zero without loading a library or
touching CUDA.

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
- EOS is excluded from codec frames, and frame-zero EOS produces exactly one
  zero-sample FINAL control marker without invoking NanoCodec;
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
`100 ms` and generation RTF p95 at or below `0.30` over exactly 108 pinned
Japanese cases. If profiling shows the target is infeasible, the target must
be revised explicitly rather than silently weakened.

The installed `mtt-runtime-benchmark` executable is the machine-readable Gate
4 runner. It loads the accepted bundle through the exact C ABI shared library,
keeps one session alive for the entire run, consumes and releases real PCM,
and applies the same ABI, snapshot, finite-PCM, 4/8/tail, contiguity, and
alignment checks as `mtt-runtime-smoke`.

```bash
timeout 2h mtt-runtime-benchmark \
  --library /absolute/prefix/lib/libmagpie_tts_rt.so.0.1.0 \
  --bundle /absolute/bundles/sofia \
  --manifest-sha256 <authenticated-64-lowercase-hex> \
  --corpus /absolute/corpora/thor-ja-v1.jsonl \
  --cuda-device 0 \
  --cuda-runtime-library \
    /usr/local/cuda-13.2/targets/sbsa-linux/lib/libcudart.so.13.2.75 \
  --nvml-library /usr/lib/aarch64-linux-gnu/nvidia/libnvidia-ml.so.1 \
  --nvml-device 0 \
  --long-run-iterations 1000 \
  --request-timeout-ms 120000 \
  --receipt-json /absolute/new/thor-benchmark-receipt.json
```

The timing definitions are fixed:

- `startup_gate_ms` starts before `runtime_create` and ends after
  `session_create`; it therefore includes the mandatory per-session startup
  golden.
- `cold` is the first user request after that startup golden. It is not a
  process-before-model-load measurement.
- `warm` submits every normal corpus case sequentially through that same
  session after the cold probe. The first normal case is deliberately measured
  once as the cold probe and again in the warm corpus.
- native prepared-token TTFA starts immediately before `request_start` and
  ends when the first audio lease is successfully acquired. The receipt also
  separates the synchronous `request_start` call from the wait between that
  call returning and first audio.
- generation time ends when the first `COMPLETED` snapshot is observed.
- total time ends only after the FINAL decoded tail or zero-sample FINAL
  control marker has been validated and released and the final drained
  snapshot has been validated.
- generation and total RTF divide those respective durations by emitted audio
  duration. Chunk interval is measured between successful `audio_acquire`
  returns.
- positive playback lateness is computed per lease against the audio timeline:
  `max(0, acquired_at - (first_audio_at + first_sample_index / 22050))`.
  Each case reports its maximum and the warm/long-run sections summarize those
  case maxima.
- percentiles use linear interpolation at position `p × (n - 1)` in the sorted
  sample set.
- cancel latency starts immediately before `request_cancel` and ends when a
  `CANCELLED` snapshot is observed. Any lease acquired after the cancel call
  fails the cancellation gate, even if it is released correctly.

This executable starts at prepared token IDs. It does not execute text
normalization/tokenization, ROS transport, a playback queue, or a physical
audio device. Accordingly, only the native prepared-token stage is measured in
`ttfa_stages`; frontend, ROS, and physical-output values are explicit `null`
values with `measured = false`. These stages must not be reported as zero or
included in a claimed microphone-to-speaker TTFA. Per-engine TensorRT timing is
also unmeasured by the C ABI v1 and is left to the companion nsys evidence.

The corpus is UTF-8 JSON Lines with no blank lines, CRLF, NUL bytes, duplicate
keys, unknown members, or missing members. The first record has exactly:

```json
{
  "record_type": "header",
  "schema_version": "magpie-tts-rt.benchmark-corpus.v1",
  "corpus_id": "thor-ja-v1",
  "tokenizer_identity_sha256": "<64-lowercase-hex>",
  "normal_case_count": 108
}
```

It is followed by exactly `normal_case_count` case records and one final
`cancel_case` record. Both record forms have exactly
`record_type`, `case_id`, `source_text`, `source_text_sha256`,
`prepared_token_ids`, and `random_seed`. `case_id` values are unique,
`source_text` contains Japanese UTF-8 and matches its SHA-256, token IDs are
nonnegative INT32 values within the authenticated model vocabulary, and the
seed is a uint32. The header tokenizer digest must equal the authenticated
model information. An acceptance corpus has exactly 108 normal cases.

The reviewed 108-case source, generation instructions, and strict generator
are tracked in
[`reference/benchmark`](../reference/benchmark/README.md) and
[`generate_benchmark_corpus.py`](../tools/frontend/generate_benchmark_corpus.py).
They are installed under
`share/magpie-tts-rt/reference/benchmark` and
`share/magpie-tts-rt/tools/frontend`; generation from an install tree sets
`PYTHONPATH` to `share/magpie-tts-rt` before invoking the module. The generated
JSONL remains an immutable external acceptance input and is not synthesized by
the benchmark runner.

After the cold and warm measurements, the runner restarts the pinned cancel
case three times and cancels it (1) immediately after `request_start`, (2)
after the first audio chunk, and (3) after the second audio chunk. Every point
must reach terminal `CANCELLED` without any lease becoming visible after the
cancel call. It then runs exactly 1000 complete requests, cycling over the
normal corpus. The 108-case quality gate and 1000-turn durability gate are
separate receipt thresholds; neither can substitute for the other.

One process RSS and one device-wide CUDART memory sample are taken after every
successful long-run request. On AGX Thor,
NVML framebuffer-memory queries are unsupported because the GPU uses shared
system memory. NVML is therefore used only for driver, device name, UUID, and
compute-capability identity. Device total/free/used memory is measured through
the explicitly pinned CUDART library's `cudaMemGetInfo`; process RSS comes from
Linux `VmRSS`. The public receipt names this source with the logical identifier
`linux_procfs_vmrss`; it does not publish the host procfs path. The first
CUDART call is deliberately made only after the startup gate finishes, so CUDA
context initialization by the benchmark cannot make startup look faster. The
receipt records the CUDART library basename, SHA-256, and byte size,
driver/runtime versions, the measurement sources, and whether NVML memory was
`unsupported` or `supported_not_used`.

Process RSS covers the benchmark process and the native TTS runtime loaded into
it. `cudaMemGetInfo` is device-wide and is not labeled as TTS-process VRAM.
Likewise, the in-process receipt does not claim process-isolated GPU
utilization. The Thor workflow profiles the exact benchmark target process
tree with nsys and records concurrent device-wide tegrastats output. After the
benchmark process exits, it writes
`thor-profile-evidence.json`, containing the public basename, byte size, and
SHA-256 of the benchmark receipt, `.nsys-rep`, and tegrastats log. The workflow
resolves each basename inside the evidence directory, rejects symlinks and
directory escapes, and rechecks byte size and SHA-256 before upload. This
post-exit companion is necessary because an nsys report does not exist until
the profiled process has terminated. Its scope is explicit: nsys is target
process-tree evidence, tegrastats is device-wide evidence, and physical audio
is not measured.

Process user and system CPU time come from `getrusage(RUSAGE_SELF)`. The
benchmark records cumulative values before runtime creation and after all
native/CUDART/NVML cleanup, their deltas, the same interval's monotonic wall
time, and `(user + system) / wall`. This ratio may exceed 1 when multiple CPU
cores are busy; it is not a percentage.

The receipt reports first/last/minimum/maximum/delta and linear slope in bytes
per iteration. Device-wide `cudaMemGetInfo` remains observational and is never
used to accept or reject process memory stability. Process RSS has two fixed
provisional gates:

- `after_long_run - after_warm` must be at most `64 MiB`
  (`67,108,864 bytes`). A negative delta passes.
- the least-squares slope over the 1000 per-request RSS samples must be at most
  `64 KiB/iteration` (`65,536 bytes/iteration`). A negative slope passes.

The endpoint limit allows bounded allocator/context settling while preventing
a large retained working-set increase. The slope limit independently detects
a persistent per-request leak; over 1000 turns it is approximately the same
64 MiB order of growth. Both limits are intentionally conservative first
release values and both must pass. The baseline is the sample immediately
after the warm suite and the final sample is after all three cancel probes and
the 1000-turn run, so intervening retained allocations are not hidden.

An accepted result requires all of the following:

- exactly 108 completed warm cases;
- warm TTFA p95 at or below `100 ms`;
- warm generation RTF p95 at or below `0.30`;
- maximum positive playback lateness at or below the oracle regression ceiling
  of `0 ms`;
- terminal `CANCELLED` with zero post-cancel leases at all three fixed
  injection points;
- exactly 1000 long-run iterations completed without a failure;
- long-run TTFA p95 at or below the oracle regression ceiling of `152.39 ms`;
- long-run generation RTF p95 at or below `0.401`;
- long-run maximum positive playback lateness at or below `0 ms`;
- process RSS growth from `after_warm` to `after_long_run` at or below
  `67,108,864 bytes`;
- process RSS linear slope at or below `65,536 bytes/iteration`;
- clean request/session/model/runtime destruction, runtime `dlclose`, CUDART
  `dlclose`, NVML shutdown, and NVML `dlclose`.

The warm suite uses the stricter first-release targets (`100 ms` and `0.30`).
The 1000-turn suite uses the oracle regression ceilings (`152.39 ms` and
`0.401`) to prove that sustained operation has not degraded beyond the
accepted baseline. A completed measurement that misses any performance,
stream, cancellation, completion, or process-RSS threshold writes a rejected
receipt and exits with status 2. Invalid inputs or a failure before a complete
measurement exits with status 1 and does not publish a partial receipt.

`--receipt-json` is mandatory, absolute, and immutable. The parent must be an
existing nonsymlink directory and the target must not exist. The JSON has
`schema_version = "magpie-tts-rt.acceptance-benchmark.v2"` and exact top-level
members `status`, `completed_at_unix_ms`, `abi_version`,
`measurement_scope`, `ttfa_stages`, `component_latency`, `resource_scope`,
`inputs`, `model`, `stream_contract`, `startup`, `cold`, `warm`,
`cancellation`, `long_run`, `memory`, `cpu`, `thresholds`, and
`verification`. It is durably published
with the same no-replace procedure as the smoke receipt. Consumers must
validate this JSON against
[`runtime-benchmark-receipt.schema.json`](../schemas/runtime-benchmark-receipt.schema.json);
`inputs` includes public basenames, SHA-256 identities, and byte sizes for the
exact runtime, CUDART, and NVML shared libraries, authenticated bundle
manifest, and corpus. Host-local absolute paths are not part of the public
contract. The human-readable stdout summary is not an acceptance contract.

The benchmark receipt deliberately does not contain hashes of nsys or
tegrastats outputs: both are finalized after the profiled benchmark exits.
The workflow validates the receipt first, then creates and validates the
companion against
[`thor-profile-evidence.schema.json`](../schemas/thor-profile-evidence.schema.json)
and re-hashes each referenced artifact before upload.

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
- the Release contract suite runs the complete deterministic benchmark,
  including its fixed process-RSS and performance acceptance thresholds;
- ASan/UBSan and TSan run every functional contract except that measured
  benchmark because sanitizer allocator, quarantine, and instrumentation
  overhead make its production RSS/performance receipt meaningless; the
  separate `runtime_benchmark_thresholds` test keeps every pure threshold
  boundary and fail-closed comparison under sanitizers without weakening a
  threshold;
- UBSan is compiled with recovery disabled and runs with `halt_on_error=1`, so
  a reported undefined operation cannot produce a successful CTest result;
- the ASan/UBSan functional pass covers all manifest mutation cases with
  stack-use-after-return disabled, then a second required pass enables
  stack-use-after-return detection for `request_state`,
  `async_pipeline_contract`, and `startup_golden`; this preserves UAR coverage
  for runtime state, concurrency, cancellation, and startup cleanup without
  turning the large aarch64 manifest parser's FakeStack overhead into an
  unbounded CI job;
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

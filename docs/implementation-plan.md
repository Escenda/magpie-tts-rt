# Streaming implementation plan

This is the acceptance and publication dependency order for making
MagpieTTS-RT usable by the live voice dialogue stack. Implementation work may
proceed in parallel, but a later phase cannot be accepted or published until
the preceding phase has a reproducible passing receipt.

A checked item means that the implementation exists and has passed its
component-level verification. It does not by itself close the phase gate.
Each gate remains open until every verification named by that gate has been
run in the stated environment.

## P0 — Locked oracle

- [x] Lock model, NanoCodec, accepted source files, active model config, and
  acceptance receipt by SHA-256.
- [x] Verify the unusual repeated `model_config.yaml` archive contract.
- [x] Implement portable boundary-fixture capture.
- [x] Publish an immutable, license-preserving source overlay for the 11
  optimized NeMo files.
- [x] Capture and publish the first authenticated boundary fixture.
- [x] Freeze the Japanese OpenJTalk/G2P vocabulary, global offsets, special
  tokens, and chunking contract.
- [x] Freeze exact prepared-token-to-source span fixtures.

Gate: a fresh accepted checkout reproduces the fixture manifest and every
tensor hash.

## P1 — Export and engine build

- [x] Export the Text Encoder and build measured TensorRT plan candidates.
- [x] Accept the Text Encoder plan with the same final plugin library used by
  Main Decoder and Local AR.
- [x] Export Main Decoder prefill with the fixed Sofia prefix.
- [x] Export true one-step Main Decoder with explicit self/cross K/V.
- [x] Export the 16-position Local AR network.
- [x] Implement the fused CFG/top-k/Gumbel/embedding plugin.
- [x] Implement the fused EOS reduction.
- [x] Export stateful NanoCodec initial-4, steady-8, and tail-1-through-8.
- [x] Build TensorRT plans only after named-binding and profile introspection.

The initial parity build disables TensorRT TF32 explicitly. The accepted
NanoCodec oracle is FP32 with IEEE FP32 matmul/cuDNN policy; enabling TensorRT's
default TF32 is a later measured optimization, not part of the parity baseline.

Gate: every component matches the locked PyTorch boundary fixture. A plan with
a missing, renamed, extra, or incorrectly profiled binding fails closed.

Current result: the three authenticated Japanese fixtures pass exact
Text Encoder, Main Decoder, Local AR, EOS/RNG, alignment, and NanoCodec
comparison with the final plugin bytes.

## P2 — Verified model bundle

The fail-closed packager, source/tokenizer contracts, seven accepted plans,
consolidated export receipt, and golden receipt are implemented. An immutable
bundle has passed the real-GPU native smoke test. Release publication remains
gated by the later Rust, ROS, physical-audio, quality, and performance gates.

- [x] Package seven plans, plugin library plus deterministic build receipt,
  tokenizer identity, exporter receipt, self-contained golden input fixture,
  accepted golden receipt, and runtime fingerprint.
- [x] Generate the manifest from inspected plans rather than hand-written
  shapes.
- [x] Verify every byte by size and SHA-256 before deserializing any plan.
- [x] Require an external authenticated manifest digest as the trust anchor.

Gate: valid bundle loads; missing, modified, mismatched-runtime, wrong-plugin,
and path-escape bundles are rejected.

## P3 — C++ runtime

- [x] Load and register the exact plugin creator/version/namespace.
- [x] Create one execution context per engine per session.
- [x] Allocate self/cross K/V, RNG, alignment, codec state, workspaces, CUDA
  streams/events, pinned PCM ring, and lease metadata.
- [x] Run Text Encoder and prefill once, then only one-step decoder calls.
- [x] Pipeline generation and codec streams without device-wide
  synchronization.
- [x] Enforce first-4, steady-8, decoded tail-1-through-8 or the zero-sample
  non-FIRST FINAL control marker, sequence, and absolute sample offsets. EOS
  itself is never a codec frame.
- [x] Implement bounded backpressure, cancellation, poison, teardown, and
  session reuse exactly as the C ABI declares.

Gate: the public C ABI completes real synthesis and passes sequence parity,
cancel, backpressure, reuse, failure persistence, and lease lifetime tests.

Current result: the clean Thor Release build passes the native test suite and
the authenticated real-GPU synthesis smoke. Long-running stress and measured
performance remain in P6.

## P4 — Rust runtime consumer

- [x] Regenerate raw bindings for any final ABI change.
- [x] Keep runtime/model/session/request handles on one dedicated inference
  thread; they are intentionally neither `Send` nor `Sync`.
- [x] Bridge commands and audio leases through typed channels.
- [x] Preserve zero-copy lease lifetime until the downstream consumer has
  copied or played the PCM.
- [x] Add native-link, packaging, ownership, cancellation, and injected-fault
  tests.

Gate: Rust drives the real Thor runtime with no shim and all ownership tests
pass under sanitizers.

Current result: the Rust workspace passes its ownership, cancellation,
injected-fault, ABI, and real-library link tests. Full synthesis through the
Rust consumer and the sanitizer run are still required to close this gate.

## P5 — ROS streaming replacement

The existing `SynthesizedSpeech` message and playback controller treat one ROS
message as one complete PCM utterance. They cannot carry Magpie streaming
without changing meaning. Replace the contract rather than adding a legacy
mode. This integration is owned by the external `fluent_vision_ros2`
repository; MagpieTTS-RT owns the C ABI, verified bundle, and Rust consumer
crate it uses. The Rust `fv_tts` replacement is implemented there, but this
phase remains unaccepted until its revision is committed and pinned by the
parent repository and the live ROS/physical-audio gate below passes.

- [ ] Revise dialogue design documents before code.
- [x] Replace whole-utterance PCM with strict per-utterance chunks containing
  sequence number, absolute first sample, first/final/abort flags, fixed
  format, and progressive alignment. The stream contract includes a
  zero-sample non-FIRST final control chunk when native EOS leaves no decoded
  tail.
- [x] Add a request seed and exact frontend token-to-source span map.
- [x] Replace the C++/VOICEVOX `fv_tts` implementation with the Rust
  MagpieTTS-RT consumer.
- [x] Make playback reject gaps, duplicates, reordering, format drift, and
  chunks after final/abort.
- [x] Report physical playback completion only after the final queued sample
  drains.
- [x] Update launch, setup, readiness, restart, status, and soundboard/dialogue
  callers.

These checked P5 items were inspected in the current external
`fluent_vision_ros2` working tree. They are not published integration until
that repository is committed and pushed and its parent submodule pin is
updated.

Gate: live ROS E2E measures first PCM, physical first audio, physical end,
barge-in cancellation, system-speech preemption, and recovery after failure.

## P6 — Thor acceptance

The native prepared-token benchmark supplies only the native subset of this
phase. Its receipt explicitly leaves frontend, per-engine, ROS, and physical
audio stages unmeasured; those values require the frontend/ROS/device harness
and cannot be inferred from native TTFA. The workflow's nsys evidence is
target-process scoped, while tegrastats remains device-wide.

- [ ] Component and complete-audio parity against the authenticated fixture.
- [ ] Japanese listening and ASR-assisted quality suite.
- [ ] TTFA split into frontend, Text Encoder/prefill, first two decoder steps,
  first-4 codec, D2H, ROS transport, queue, and physical device start.
- [ ] Steady RTF, per-component latency, positive playback lateness, CPU, GPU,
  VRAM, and synchronization audit.
- [ ] Repeated short/long synthesis, cancellation at every boundary, queue
  saturation, session reuse, and multi-session stress.

Gate: signed receipt records exact source, model, bundle, engine, plugin,
driver, CUDA, TensorRT, GPU, test inputs, quality, performance, and failures.

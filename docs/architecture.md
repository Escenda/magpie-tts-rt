# Architecture

## Boundary

MagpieTTS-RT owns deterministic GPU inference and the state required to advance
one synthesis session. It does not own text normalization, ROS, dialogue
policy, or an audio device.

The application performs:

1. language-specific normalization and Japanese reading conversion;
2. tokenization using the tokenizer pinned by the engine bundle;
3. source-text to token-span tracking;
4. playback policy, resampling, device output, and barge-in.

The runtime receives validated token IDs and an explicit random seed. Voice,
sampling, and engine semantics are fixed by the verified model bundle. It
returns ordered PCM leases, alignment progress, or a terminal error.

## Ownership

```text
Runtime
└── Model
    └── Session
        └── Request
            └── zero or more AudioLease values
```

- `Runtime` owns the CUDA device binding, TensorRT runtime, worker scheduler,
  and immutable runtime fingerprint. TensorRT's creator registry is
  process-global, so the first runtime also installs one authenticated
  process-global plugin owner. Later runtimes must present the same plugin
  digest, ABI, and creator contract and reuse that exact mapping. A different
  digest fails as an explicit process conflict before plan deserialization.
- `Model` owns a verified bundle manifest, its strictly parsed startup fixture,
  and immutable deserialized engines. The bundle authenticates the upstream
  model ID/display version/immutable revision/source hash and its acceptance
  receipt; it does not copy the unused `.nemo` weights.
- `Session` owns one execution context per engine, KV caches, cross-attention
  state, RNG state, alignment state, codec causal state, CUDA streams/events,
  workspaces, and a bounded PCM emission ring. It becomes externally visible
  only after the mandatory golden generation has passed.
- `Request` owns the state of one utterance, cancellation state, absolute
  sample position, sequence number, and outstanding audio leases.
- An `AudioLease` keeps one pinned-host PCM region alive until the application
  explicitly releases it.

A parent cannot be destroyed while a child exists. Destruction does not
implicitly cancel work or wait for GPU completion.

## Threading and CUDA

Version 1 permits one active request per session. Concurrent utterances require
separate sessions and therefore separate TensorRT execution contexts.

Each session has at least two CUDA streams:

- generation stream: Text Encoder, prefill, Main Decoder step, Local AR,
  sampling, and EOS;
- codec stream: NanoCodec and PCM materialization.

Two fixed generation slots hold codec codes and pinned generation diagnostics.
For batch `n`, the codec stream waits on that slot's `codes-ready` event, packs
the codes into its private input buffer, and records `codes-consumed`. The
generation stream may then produce batch `n+1` in the other slot while
NanoCodec decodes batch `n`. Reusing a slot requires an explicit generation
stream wait on its `codes-consumed` event.

The first four-frame NanoCodec call is the exception: the runtime waits for and
publishes its PCM before enqueueing the first steady generation batch, so GPU
competition cannot lengthen the first-audio path. Cross-stream overlap starts
with the second audio chunk.

The host polls CUDA events only at the two boundaries where it needs host
data: EOS/alignment diagnostics for terminal routing and PCM for publication.
Healthy request execution does not call `cudaEventSynchronize` or
`cudaDeviceSynchronize`. The single pinned PCM slot is not reused until the
bounded ring and audio lease make it available. Cancellation and non-poisoning
request failures drain every armed slot event before the session can be reused.
Successful events advance text progress, codec state, and published audio
state. Failed work never advances those counters.

`cudaStreamSynchronize` is reserved for `SessionResources` teardown: the
explicit teardown call and destructor failure-unwind path wait for both
session streams before destroying TensorRT contexts, workspace storage, CUDA
events, or streams. It is not part of the healthy per-request generation
path.

## Session startup gate

Creating a session performs a complete deterministic golden generation with
the prepared token IDs and uint32 seed stored in the authenticated bundle. It
uses the same execution contexts, CUDA streams, alignment controller, Local
AR sampler, codec state, and bounded backpressure path as a normal request, so
the successful run also warms the exact production path.

The gate compares:

- the complete codebook-major decoder tensor;
- every scheduled NanoCodec input in engine-call order;
- the complete valid FP32 PCM stream;
- codec-frame and PCM-sample counts.

The expected values are exact SHA-256 digests and counts, not tolerances.
Golden audio is drained internally and never published to the application. A
parse error, execution failure, count difference, or digest difference fails
session creation and destroys the unaccepted resources. No degraded session
or alternate backend is created.

The golden producer and private audio drainer have one guarded lifetime. If
lease draining throws, the gate closes publication by cancellation, waits for
the producer to drain every armed CUDA boundary, joins that producer, and only
then returns the original error through the C ABI. A thread or session
resource cannot escape this failure path.

## Streaming

The accepted schedule is:

- first emission: 4 codec frames;
- steady emission: 8 codec frames;
- final decoded emission: 1 through 8 codec frames;
- or, when EOS is the first frame and no decoded frame remains, one
  zero-sample non-FIRST FINAL control marker that bypasses NanoCodec.

One codec frame is 1,024 mono FP32 samples at 22,050 Hz. The C++ ring is a
bounded synthesis-emission queue, not the physical playback queue. When the
ring is full, generation stops at a codec-frame boundary. It never overwrites
or drops PCM.

Every audio lease contains:

- utterance-global sequence number;
- utterance-global first sample index;
- sample count and sample rate;
- first/final flags;
- PCM format and pointer valid for the lease lifetime;
- committed token progress plus ordered in-chunk alignment events when
  available.

Alignment events are emitted at the played-sample boundary after a Main
Decoder step, normally two codec frames apart. An early EOS event follows the
single terminal frame when only the first frame of that step is emitted. A
frame-zero EOS emits no alignment event because it has no played-sample
boundary. They
carry prepared-token positions and utterance-global, end-exclusive sample
indices, so playback never reports text before its audio has drained. Only
the application frontend can map those positions to source Unicode spans.

## Cancellation and failure

Cancellation is observed at an engine or codec-frame boundary. After
cancellation is accepted:

- no new audio lease may be published;
- unleased PCM is discarded;
- existing leases remain valid until released;
- the request remains `RUNNING` while queued CUDA work is drained;
- only a successful drain publishes `CANCELLED`; a drain failure publishes
  `FAILED` and poisons the session instead.

TensorRT, CUDA, plugin, invariant, or manifest failure poisons the affected
session. A poisoned session cannot start another request. There is no implicit
retry or alternate execution backend.

Host allocation failure is also poisoning. Allocation can fail after work has
entered a CUDA stream but before the host records a completion flag, so such a
request cannot prove that its workspace is reusable.

Context overflow is an explicit error. Text or KV state is never silently
truncated.

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

Local AR uses one immutable CUDA Graph per session. Its TensorRT context is
bound once to fixed addresses: logical-step `unfinished` is staged into one
canonical input, and canonical invalid-row/EOS outputs are committed to the
logical step by one post-graph kernel. The same kernel advances the canonical
RNG counter, appends codec codes, and latches EOS. Tensor addresses and context
state are never modified after capture; a capture or launch failure closes the
session instead of falling back to per-kernel enqueue.

Main Decoder uses two TensorRT contexts and two CUDA Graphs, one for cache
A-to-B and one for B-to-A. Unlike Local AR, these graphs are recreated for
every request because `condition_mask`, the attention prior, and cross K/V
retain that request's dynamic text length. Reusing a startup or preceding
request graph for an equal-looking text length is not an accepted path. The
request initializes one DEVICE `position` scalar to the value immediately
before the first step; every eager or captured invocation increments it on the
generation stream before TensorRT reads it. The host never dispatches an
absolute position per decoder step.

Two fixed generation slots hold codec codes and pinned generation diagnostics.
For batch `n`, the host blocks on that slot's `codes-ready` event because it
must inspect the copied execution status, EOS, and alignment diagnostics before
the codes may enter NanoCodec. The codec stream therefore starts only after
generation is known complete; adding another stream wait on `codes-ready`
would be redundant. It packs the codes into its private input buffer and
records `codes-consumed`. The generation stream may then produce batch `n+1`
in the other slot while NanoCodec decodes batch `n`. Reusing a slot requires an
explicit generation stream wait on its `codes-consumed` event.

The first four-frame NanoCodec call is the exception: the runtime waits for and
publishes its PCM before enqueueing the first steady generation batch, so GPU
competition cannot lengthen the first-audio path. Cross-stream overlap starts
with the second audio chunk.

The host synchronizes CUDA events only at the two boundaries where it needs
host data: execution/EOS/alignment diagnostics for routing and PCM for
publication. These events are created with `cudaEventBlockingSync`, so the
waiting thread sleeps in the driver instead of repeatedly issuing
`cudaEventQuery` and short host sleeps. Healthy request execution uses one
`cudaEventSynchronize` per required host-data boundary and never calls
`cudaDeviceSynchronize`. The single pinned PCM slot is not reused until the
bounded ring and audio lease make it available. Cancellation and non-poisoning
request failures drain every armed slot event before the session can be reused.
Successful events advance text progress, codec state, and published audio
state. Failed work never advances those counters.

`cudaStreamSynchronize` is used at session startup and teardown, plus once per
normal request after both request-specific Main Decoder graphs have launched.
That post-first-audio wait makes the dynamic-text graph-memory measurement
complete before the request continues; it is not on the first-audio path or
the steady replay path. Startup also uses synchronization after discarded
Local AR and fixed-shape NanoCodec warmup enqueues and after graph
upload/execution. The explicit teardown call and destructor failure-unwind
path wait for both session streams before destroying graph executables,
TensorRT contexts, workspace storage, CUDA events, or streams.

## Session startup gate

Creating a session performs a complete deterministic golden generation with
the prepared token IDs and uint32 seed stored in the authenticated bundle. It
uses the same execution contexts, CUDA streams, alignment controller, Local
AR sampler, codec state, and bounded backpressure path as a normal request, so
the successful run also warms the exact production path.

The first valid Local AR and fixed NanoCodec input is enqueued once before
capture to flush TensorRT deferred setup. Those eager results are discarded
without advancing RNG, codec state ownership, or logical progress. Main
Decoder is different because discarding recurrent cache output would change
the sequence. Its first A-to-B eager result produces the second pair of frames
in the initial four-frame audio. Only after that PCM is published does the
first B-to-A eager execution run. The next A-to-B and B-to-A invocations are
captured and immediately launched, and those launch results are used. Every
later Main step is one graph replay. The dynamic NanoCodec tail-1..8 route
remains a direct TensorRT enqueue and is not an eager fallback for a missing
fixed-route graph.

Main Decoder cache and NanoCodec state both alternate between buffers A and B.
Each pair of graph directions owns two distinct TensorRT execution contexts; a
graph never shares a context with its reverse route. Both extra contexts'
`getDeviceMemorySizeV2()` values are charged to explicit context memory before
the session workspace is allocated. Graph executable memory is separate:
session creation records one immutable aggregate baseline before any graph,
then measures aggregate current/high-water growth after the golden has replayed
all routes. Every normal request destroys its preceding Main graphs, captures
the current text shape, and measures Local AR plus NanoCodec plus the new Main
graphs from that same baseline. Thus startup text length is never assumed to
bound another request, and every observed total must fit the authenticated
session device-memory limit.

After the startup golden has discovered and replayed every mode-8 active-K
bank, and before the session is published as ready, the runtime reads the
class table from the authenticated plugin handle through the C ABI. It fully
validates the 7-QK/14-PV classes and all 249 K records, serializes the table as
the canonical ASCII JSON identity, and requires its SHA-256 to equal
`artifacts.plugin.main_device_position_class_table.sha256` in the manifest.
`NOT_READY`, `CONFLICT`, a malformed table, or a digest mismatch closes the
startup gate; the runtime never substitutes a cached or reconstructed table.

The session is not published unless every required graph is ready, all
explicit context and observed graph memory is accounted, the graph-backed
golden matches exactly, and this live plugin class-table identity is
authenticated.

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

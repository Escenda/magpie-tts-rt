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

- `Runtime` owns the CUDA device binding, TensorRT runtime, plugin registry,
  worker scheduler, and immutable runtime fingerprint.
- `Model` owns a verified bundle manifest and immutable deserialized engines.
- `Session` owns one execution context per engine, KV caches, cross-attention
  state, RNG state, alignment state, codec causal state, CUDA streams/events,
  workspaces, and a bounded PCM emission ring.
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

CUDA events join `codes-ready` and `audio-ready` dependencies. The runtime does
not perform a device-wide synchronization during a healthy generation step.
Successful events advance `valid_tokens`, codec state, and published audio
state. Failed work never advances those counters.

## Streaming

The accepted schedule is:

- first emission: 4 codec frames;
- steady emission: 8 codec frames;
- final emission: 1 through 8 codec frames.

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
- committed token/source alignment progress when available.

## Cancellation and failure

Cancellation is observed at an engine or codec-frame boundary. After
cancellation is accepted:

- no new audio lease may be published;
- unleased PCM is discarded;
- existing leases remain valid until released;
- queued CUDA work is drained before the request becomes `CANCELLED`.

TensorRT, CUDA, plugin, invariant, or manifest failure poisons the affected
session. A poisoned session cannot start another request. There is no implicit
retry or alternate execution backend.

Context overflow is an explicit error. Text or KV state is never silently
truncated.

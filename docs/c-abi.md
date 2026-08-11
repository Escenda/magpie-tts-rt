# C ABI version 1

The only exported symbol is:

```c
mtt_status_t mtt_get_api(uint32_t requested_abi_version, mtt_api_v1_t* api);
```

Callers initialize `api.struct_size` and `api.abi_version` before negotiation.
Version 1 requires an exact structure size and ABI version. It does not expose
compatibility aliases or partially populated tables.

Every public input/output structure begins with `struct_size` and
`abi_version`. Reserved and flag fields must be zero unless the active ABI
version defines them.

## Handles

`Runtime`, `Model`, `Session`, and `Request` are opaque native handles.

- A parent must outlive every child.
- A parent destroy operation returns `BUSY` while a child exists.
- A request can be destroyed only after reaching a terminal state and after
  every audio lease has been released.
- Destroy does not implicitly cancel or wait.
- A destroy call consumes its handle only when it returns `OK`. Every non-OK
  return leaves the handle and all ownership relations intact so the caller
  can correct the reported condition and retry.
- A destroyed native handle must never be passed to another C call. The safe
  Rust wrapper enforces this ownership rule.
- Successful `session_destroy`, `model_destroy`, and `runtime_destroy` leave
  the owning runtime's CUDA device selected on the calling thread. This makes
  device-bound teardown fail before consuming a handle; callers that share a
  thread with other CUDA devices must explicitly restore their device after a
  successful destroy. The Rust consumer uses a dedicated inference thread, so
  this state never leaks into application or ROS executor threads.

Dropping the Rust `Request` calls only `request_destroy`. If the request is
still running or owns a lease, native destruction returns `BUSY` and the
wrapper aborts instead of hiding a live GPU resource. Normal callers must
cancel and wait for a terminal snapshot, or wait for completion, release every
lease, and then call `Request::close`.

`InferenceWorker::shutdown(self)` is the only Rust worker operation that sends
the shutdown command and joins the inference thread. Start admission uses a
capacity-one bounded channel and `synthesize` returns typed `Busy` immediately
when that channel is full. Cancel and shutdown use a separate bounded control
channel. After enqueueing either payload, the sender issues a nonblocking wake
through a separate capacity-one channel; a full wake channel means an existing
wake already covers the payload, while a disconnected wake channel is an
explicit command-channel failure. The idle inference thread blocks on that wake
instead of periodically polling. After waking, and before every active-request
wait, it checks the control channel before start admission, so a start flood
cannot delay cancellation or shutdown behind queued synthesis work. Active
requests use `request_wait` with a two-millisecond control-service slice and
call `audio_acquire` only when the returned revision advertises an available
lease. A full capacity-one synthesis event channel remains nonblocking
backpressure: the worker retains at most one pending event and continues its
bounded control checks without overlapping native request operations. Dropping
`InferenceWorker` itself only detaches its `JoinHandle`; it does not
synchronously cancel a request, join, or call native code. After the last
control sender, including senders owned by live `SynthesisStream` values, and
the start sender are dropped, channel disconnection makes the inference thread
explicitly cancel any active request, observe its drained terminal state,
destroy it, and then destroy the remaining native hierarchy on that thread. A
runtime failure moves the worker into a fatal state where no further native
request operation is issued. The worker retries a nonblocking
`SynthesisEvent::RuntimeError` send until the bounded event channel has room,
so a connected consumer receives it exactly once. Receiver disconnection exits
with the original error. Explicit `shutdown` detaches a still-backpressured
event sender and joins with that same original error, so unread audio cannot
block shutdown indefinitely.

Before enqueueing a start command, `InferenceWorker::synthesize` validates a
nonempty prepared-token sequence, the loaded model's exact maximum token
count, the authenticated normal-token/EOS row contract, and the uint32 seed
range. Every token before the terminal position must be in
`[0, tokenizer_vocabulary_size)`, and the final token must equal
`eos_token_id`. BOS, an early EOS, a missing EOS, or any other embedding row
is rejected. These caller errors return `WorkerError::InvalidRequest` without
invoking native code and do not poison the worker. Consequently, any error
returned by the native `request_start` call is an invariant, runtime, or
device failure: the start reply receives that error and the worker exits fail
closed while retaining the same error for explicit `shutdown`.

Version 1 allows one active request per session. A session and its request are
owned by one inference thread. The caller must serialize every operation on
the same `Session` or `Request`; `request_poll`, `request_wait`,
`request_cancel`, `request_destroy`, `audio_acquire`, and `audio_release` must
not overlap. The safe Rust wrapper enforces this with non-`Send`, non-`Sync`
owners and mutable request access. Different sessions may execute
concurrently because they own different TensorRT execution contexts.

Version 1 does not select a speaker at session creation. Each verified model
bundle and its prefill plan fix one baked voice. Loading another voice means
loading another model bundle.

`model_load` requires `mtt_model_desc_v1_t`. Its
`expected_manifest_sha256` is the 32-byte digest obtained from authenticated
release metadata; an all-zero value is invalid. The runtime hashes the exact
manifest snapshot it parses and compares it with this trust anchor before
using any manifest-declared artifact digest.

`model_load` copies the supplied path bytes before it returns and never retains
the caller's pointer. On successful `request_start`, the runtime has copied the
complete token sequence into request-owned storage; it never retains
`text_token_ids`. Callers may release or reuse both input buffers immediately
after the respective function returns.

`model_get_info` returns only properties authenticated by that loaded bundle:
the normal tokenizer row count, text-embedding row count, BOS/EOS/Japanese-pad
IDs, request limits, NanoCodec output format and frame schedule, and the
32-byte tokenizer identity. A Japanese frontend must compare that identity
with the exact frontend assets it verified before it submits any token ID.
Matching the row counts alone is insufficient because a different token order
can remain in range while producing the wrong speech.

`session_create` is also the mandatory device warmup and acceptance boundary.
It runs the bundle's authenticated prepared-token fixture through the complete
Text Encoder, Main Decoder, Local AR, and stateful NanoCodec path. The runtime
reassembles and hashes the complete decoder tensor, hashes each scheduled
NanoCodec input in execution order, and hashes every valid FP32 PCM sample.
All three hashes and both frame/sample counts must equal the authenticated
fixture. Until that succeeds, no session handle is returned and no golden PCM
is exposed through the audio lease API. A mismatch returns
`MTT_STATUS_HASH_MISMATCH`; there is no skip flag, alternate fixture, or
CPU/backend fallback.

Public token IDs are INT64 for a language-neutral ABI. The runtime rejects
negative or wider-than-INT32 values before converting to the locked Text
Encoder input, then enforces the same normal-prefix plus terminal-EOS contract
used by startup golden validation and the Rust worker. Random seeds are
accepted only in `[0, 2^32)`; there is no truncation or modulo conversion.

Bundle paths are UTF-8 byte spans of 1 through
`MTT_MAX_BUNDLE_PATH_BYTES` bytes. They need not be NUL-terminated, but an
embedded NUL is invalid. A request contains 1 through `MTT_MAX_TEXT_TOKENS`
prepared tokens.

## Errors

Every operation returns an explicit `mtt_status_t`. Functions that accept
`mtt_error_v1_t` use that caller-owned structure for call-local details.
There is no global or thread-local `last_error`.

If an error structure is supplied, the caller must initialize its
`struct_size` and `abi_version`. A malformed error structure produces
`MTT_STATUS_ABI_MISMATCH` and cannot receive details.

For every call with a valid error structure:

- `MTT_STATUS_OK` sets `code=OK`, `stage=NONE`, and an empty NUL-terminated
  message;
- `WOULD_BLOCK` and `TIMEOUT` set the matching code, `stage=REQUEST`, and may
  use an empty message;
- every other non-OK return sets the matching code, a non-`NONE` stage, and a
  non-empty NUL-terminated diagnostic.

The Rust wrapper validates this relation for success, control, and failure
statuses. It does not accept a malformed error buffer as a successful poll,
timeout, or empty audio queue.

`request_poll` and `request_wait` can return `OK` while reporting a request
whose state is `FAILED`. In that case the call-local error remains
`OK`/`NONE`/empty because the poll itself succeeded; the asynchronous failure
status, stage, and message are carried by the snapshot fields below.

Each successful `request_poll` or `request_wait` snapshot also has one exact
state/status/diagnostic relation:

| `state` | `terminal_status` | `terminal_error_stage` | `terminal_error_message` |
| --- | --- | --- | --- |
| `RUNNING` | `OK` | `NONE` | empty |
| `COMPLETED` | `OK` | `NONE` | empty |
| `CANCELLED` | `CANCELLED` | `REQUEST` | empty |
| `FAILED` | `INVALID_ARGUMENT`, `ABI_MISMATCH`, `IO_ERROR`, `MANIFEST_ERROR`, `RUNTIME_MISMATCH`, `HASH_MISMATCH`, `ENGINE_ERROR`, `CUDA_ERROR`, `POISONED`, `UNAVAILABLE`, or `INTERNAL_ERROR` | a declared non-`NONE` stage | non-empty |

`BUSY`, `WOULD_BLOCK`, `TIMEOUT`, and `CANCELLED` are control outcomes and
must not appear as the `terminal_status` of a `FAILED` snapshot. An unknown
state, status, or stage is an ABI violation. The terminal message is always
NUL-terminated. Once a request enters `FAILED`, its terminal status, stage,
and message are retained unchanged in every later snapshot, even while audio
lease counts continue to change.

Functions validate the error-buffer header before mutating outputs. When that
header is malformed, output handles and output structures are unchanged. With
a valid error buffer, create/load/start functions set their output handle to
null before later validation and leave it null on failure. Poll/acquire output
structures are valid only after `OK`; they remain caller-owned initialized
storage on every non-OK return.

## Polling and waiting

`request_poll` returns the current valid snapshot without waiting.
`request_wait(after_revision, timeout_nanoseconds, ...)` waits until the
request revision is strictly greater than `after_revision`. If a newer
revision already exists, it returns immediately. A successful return must
contain `snapshot.revision > after_revision`.

If no newer revision becomes available before the timeout, `request_wait`
returns `TIMEOUT`, writes the matching request-stage error status, and leaves
the caller's snapshot unchanged. A timeout of zero is a nonblocking revision
check. Implementations handle spurious internal wakeups and never convert one
into a successful stale snapshot.

## Audio leases

`audio_acquire` returns:

- mono FP32 samples at 22,050 Hz, or the zero-sample FINAL control marker
  described below;
- sequence and utterance-global sample offsets;
- first/final flags;
- committed text-token progress and zero or more in-chunk alignment events
  when alignment is valid;
- a nonzero lease identifier.

Each `mtt_alignment_event_v1_t` records an utterance-global, end-exclusive
played-sample boundary and the end-exclusive prepared text-token position
selected by the alignment controller. The progress becomes valid only after
the audio device has drained every sample before that boundary. It normally
follows a generated two-frame Main Decoder step; if EOS terminates the first
frame, the final event follows that one emitted frame. Events are ordered by
sample index and include only strict token-position advances. They are model
alignment observations, not invented source-character positions. The
application maps them to source spans only through its exact frontend span
map.

An event lies inside the sample range of its owning lease and on a 1,024-sample
codec-frame boundary. The event array has the same lifetime as the PCM
pointer. A zero event count requires a null pointer. When
`ALIGNMENT_VALID` is absent, the event pointer, event count, and
`committed_text_tokens` are all zero. When it is present, the lease-level
`committed_text_tokens` is the progress at the end of the lease and equals the
last event's value when that lease contains an event.

Local AR reports `end_frame_index` as the first EOS frame. EOS is not audio and
is never sent to NanoCodec. If decoded frames remain, FINAL carries the normal
1–8-frame tail. If EOS is at frame zero and no frames remain in that emission,
the runtime publishes one non-FIRST FINAL control marker with
`sample_count == 0`, `samples == NULL`, the next contiguous sequence number,
and an unchanged `first_sample_index`. It contains no alignment events and
cannot advance committed text progress. Consumers release this lease normally,
write no PCM, and complete playback only after already queued PCM has drained.
Zero-sample non-FINAL or FIRST leases are ABI violations.

The sample and alignment-event pointers remain valid until the matching
`audio_release` returns `OK`. That successful return consumes the lease
identifier and ends both buffer lifetimes. If the identifier named a live
lease owned by that request at call entry, every non-OK return leaves that
lease identifier, sample pointer, event pointer, and their bytes valid and
unchanged so the caller can inspect them and retry the same release. Lease
identifiers come from one process-wide,
strictly increasing 64-bit sequence shared by every runtime created through
the loaded library. They are never reused, including after release or request
destruction, so a stale or cross-request identifier cannot alias a different
live lease. Callers must not assume adjacent values because different sessions
can acquire concurrently. If the sequence cannot advance without wrapping,
acquisition fails closed with `POISONED` and returns no lease. The runtime
never reuses or overwrites a live leased buffer. A consumer that retains all
leases eventually fills the bounded ring and pauses generation.

If `audio_acquire` returns `OK`, its lease identifier is nonzero. Returning an
invalid or zero identifier is an ABI violation: there is no valid identifier
with which the caller can release the native ownership. The Rust wrapper
therefore poisons the request and aborts on destruction instead of pretending
that cleanup succeeded.

The C ABI does not resample, convert to PCM16, or write to an audio device.

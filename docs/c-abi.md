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

Dropping the Rust `Request` calls only `request_destroy`. If the request is
still running or owns a lease, native destruction returns `BUSY` and the
wrapper aborts instead of hiding a live GPU resource. Normal callers must
cancel and wait for a terminal snapshot, or wait for completion, release every
lease, and then call `Request::close`.

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

- mono FP32 samples at 22,050 Hz;
- sequence and utterance-global sample offsets;
- first/final flags;
- committed text-token progress when alignment is valid;
- a nonzero lease identifier.

The sample pointer remains valid until the matching `audio_release` returns
`OK`. That successful return consumes the lease identifier and ends the
buffer lifetime. If the identifier named a live lease owned by that request
at call entry, every non-OK return leaves that lease identifier, sample
pointer, and sample bytes valid and unchanged so the caller can inspect them
and retry the same release. Lease identifiers come from one process-wide,
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

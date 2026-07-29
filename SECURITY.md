# Security

TensorRT plans and plugins contain executable code and must already be trusted
by the caller. The current manifest hashes establish internal bundle integrity;
they are not signatures and do not establish who produced a bundle. An
attacker able to replace both the manifest and its artifacts can create a
self-consistent malicious bundle.

The application or deployment system must establish a trust anchor before
calling `model_load`, for example by verifying a signed release and extracting
its expected manifest digest. ABI v1 requires that 32-byte digest in
`mtt_model_desc_v1_t`; an unanchored load is not representable. The runtime
must compare it with the exact manifest snapshot being parsed, then require
every declared artifact digest and runtime contract to match before
deserialization.

Model loading and TensorRT deserialization are not implemented in the initial
contract-only runtime. Until that gate is complete, `model_load` returns
`MTT_STATUS_UNAVAILABLE`.

Do not report a security issue through a public issue. Use GitHub's private
security advisory flow for this repository.

The project does not support loading:

- untrusted or user-uploaded TensorRT plans;
- bundles with unknown manifest fields;
- bundles whose runtime fingerprint differs from the active system;
- plugins outside the bundle's pinned name, ABI version, and SHA-256.

The bundle verifier must produce immutable snapshots for every artifact while
hashing those same bytes. Future plan deserialization and plugin loading must
consume only those sealed snapshots. Reopening a diagnostic canonical path, or
using a mutable source-file descriptor after hashing, is not permitted because
either would reintroduce replacement or in-place mutation races.

Every artifact also declares its exact byte length. The sum must exactly equal
the authenticated bundle snapshot budget and may not exceed the runtime's
16 GiB hard limit. A size mismatch is rejected before deserialization.

A verified plugin can still execute code from its ELF dependencies. Before
plugin loading is implemented, the release contract must either make the
plugin self-contained except for an explicit CUDA/TensorRT/system allowlist
and reject `RPATH`/`RUNPATH`, or authenticate and snapshot its complete
dependency closure. Hashing only the top-level `.so` is not sufficient.

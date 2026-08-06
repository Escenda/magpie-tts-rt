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
compares it with the exact manifest snapshot being parsed, then requires
every declared artifact digest and runtime contract to match before
deserialization.

Model loading copies the manifest, plugin, seven plans, receipts, and startup
fixture into sealed anonymous file snapshots while hashing those same bytes.
The plugin is loaded from its sealed descriptor, and TensorRT deserializes
plans from the verified snapshots. Canonical diagnostic paths are never
reopened for execution.

Do not report a security issue through a public issue. Use GitHub's private
security advisory flow for this repository.

The project does not support loading:

- untrusted or user-uploaded TensorRT plans;
- bundles with unknown manifest fields;
- bundles whose runtime fingerprint differs from the active system;
- plugins outside the bundle's pinned name, ABI version, and SHA-256.

The bundle verifier produces immutable snapshots for every artifact while
hashing those same bytes. Reopening a diagnostic canonical path, or using a
mutable source-file descriptor after hashing, is not permitted because either
would reintroduce replacement or in-place mutation races.

Every artifact also declares its exact byte length. The sum must exactly equal
the authenticated bundle snapshot budget and may not exceed the runtime's
16 GiB hard limit. A size mismatch is rejected before deserialization.

A verified plugin can still execute code from its ELF dependencies. The only
supported packager inspects the staged plugin before `dlopen`, requires SONAME
`libmagpie_tts_rt_plugins.so.0`, requires exactly the fixed CUDA Runtime,
TensorRT, C++, compiler-runtime, C library, and aarch64 loader dependencies,
and rejects `RPATH`, `RUNPATH`, `AUDIT`, `DEPAUDIT`, `FILTER`, and
`AUXILIARY`. The accepted plugin and manifest hashes therefore bind that
dependency contract.

The host operating system and its dynamic-loader search configuration remain
part of the trusted computing base. System dependency files are not copied
into the model bundle. Deployment must prevent untrusted modification of the
system CUDA, TensorRT, C/C++ runtime, loader, and library search environment;
the runtime fingerprint is a compatibility check, not a signature over those
files.

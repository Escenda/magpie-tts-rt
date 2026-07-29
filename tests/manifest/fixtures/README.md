# Manifest test fixtures

`minimal-valid.json` is synthetic parser input. Its artifact hashes, engine
files, and the one-element `fixture_only_causal_state_bundle` do not describe a
runnable MagpieTTS-RT bundle.

A production bundle is not acceptable until the NanoCodec exporter enumerates
every causal convolution history, transposed-convolution overlap, residual
history, and related state tensor. That exact list must be written directly to
`codec.state_bindings`. Replacing it with an empty list or an unrecorded
runtime allocation is rejected by the manifest parser. The initial plan
creates deterministic initial state internally and exposes only its state
outputs; the runtime does not supply an implicit zero state.

The fixture also exercises the schema-version-1 Sofia-only contract: its
217-position speaker context and `AUDIO_BOS` are baked into prefill, CFG row 0
is conditional while row 1 is all-zero/all-false unconditional state, and the
first one-step decoder write uses absolute cache position 218. Profile ranges
are fixed to text `T=1/64/512` and NanoCodec tail `F=1/4/8`; the placeholder
hash values remain synthetic.

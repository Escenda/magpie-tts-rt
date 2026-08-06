# Manifest test fixtures

`minimal-valid.json` is synthetic parser input. Its artifact hashes and engine
files do not describe a runnable MagpieTTS-RT bundle. Its NanoCodec tensor
contract is nevertheless exact: 97 named persistent states, one token input
and 99 outputs for initial-4, and 98 inputs and 99 outputs for steady-8 and
tail-1-through-8.

The source-model placeholder is an acceptance receipt plus the upstream source
SHA-256, not a `.nemo` file. The tokenizer placeholder is an identity receipt
plus the canonical frontend/vocabulary/special-token identity hash, not a raw
tokenizer asset. The top-level golden fixture is a separate authenticated file
artifact from the accepted golden receipt; its three-byte placeholder is not
a valid startup fixture. The eight license/notice entries are likewise
three-byte parser fixtures; the real packager copies and authenticates each
required legal artifact.

The registry contains 92 causal-convolution input histories and five
transposed-convolution pending overlaps. It is generated from
`tools/export/nanocodec_contract.py` and written directly to
`codec.state_bindings`. Replacing it with an aggregate, changing its order,
shape, or binding name, or adding an unrecorded runtime allocation is rejected
by the schema and manifest parser. The initial plan creates deterministic
initial state internally and exposes only its state outputs; the runtime does
not supply an implicit zero state.

The fixture also exercises the schema-version-1 Sofia-only contract: its
217-position speaker context and `AUDIO_BOS` are baked into prefill, CFG row 0
is conditional while row 1 has an all-zero condition and a mask with only text
position zero true, and the first one-step decoder write uses absolute cache
position 218. Profile ranges are fixed to text `T=1/64/512` and NanoCodec tail
`F=1/4/8`; the placeholder hash values remain synthetic.

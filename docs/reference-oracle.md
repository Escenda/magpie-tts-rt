# Reference oracle

The first implementation is based on MagpieTTS multilingual version `v2607`
at immutable upstream revision
`5023df68bd3f5b5ce6d666a50979bc501af145cc`, with the Sofia speaker and the
accepted native BF16 streaming path.

Pinned external assets:

| Asset | Identifier |
| --- | --- |
| NVIDIA NeMo Speech base revision | `9ae3e66b7314b0358c96bce47fbac56d78728bcd` |
| `magpie_tts_multilingual_357m.nemo` SHA-256 | `ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953` |
| source-model acceptance receipt v2 SHA-256 | `8d324dd6f08b9ab79c7d01028a5640410575ece4b69fa50cc03c286e52772981` |

The complete byte-level lock is
[`reference/oracle-lock.json`](../reference/oracle-lock.json). The base
revision alone does not identify the accepted optimized oracle because the
initial experiment contains additional local changes and three new source
files. The lock fixes all 11 optimized files individually, and
[`reference/nemo-oracle-overlay`](../reference/nemo-oracle-overlay) contains
those exact, license-preserving bytes.

The model archive contains five members named `model_config.yaml`. They are not
identical. NeMo extraction applies them in archive order, so the fifth and last
member is the active configuration. Its SHA-256 is
`9577bfa4769981849feccd6a54d38425d81fff41ac16c282f60be97faf46ebe4`.
Reading the first member, concatenating all members, or silently selecting one
by name produces the wrong inference configuration.

The model is not part of this repository. Download and use it only under the
[NVIDIA Open Model License](https://huggingface.co/nvidia/magpie_tts_multilingual_357m).

## Verifying an oracle checkout

Every path is explicit. The verifier does not search caches or substitute
another model:

```bash
python3 tools/oracle/verify_oracle_lock.py \
  --lock reference/oracle-lock.json \
  --speech-root /path/to/accepted/NeMo \
  --model /path/to/magpie_tts_multilingual_357m.nemo \
  --codec-model /path/to/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  --acceptance-receipt \
    reference/source-model-acceptance-sofia-v2-provenance.json
```

Fixture capture additionally verifies that the NeMo modules imported by
Python resolve to the exact files under `--speech-root`. Pointing the byte
verifier at one checkout while importing another checkout is a hard failure.
The frontend lock also records pyopenjtalk `v0.4.1` source revision
`0f0fc44e782a8134cd9a51d80b57b48a7c95bb80`, its Open JTalk submodule
revision, and the size and SHA-256 of all nine accepted dictionary files.

## Capturing TensorRT parity fixtures

`capture_boundary_fixture.py` runs the locked model on CUDA and records
portable little-endian tensors. It captures:

- Japanese token IDs, mask, Text Encoder embedding, and condition;
- the complete 175-row Japanese token table, effective duplicate-token
  resolution, locked frontend versions and OpenJTalk dictionary bytes, and the
  `こんにちは。` G2P/ID golden;
- conditional and unconditional CFG rows and the Sofia prefix;
- Main Decoder prefill output and all valid self/cross K/V state;
- the first Local AR sample, RNG counter advancement, alignment state, and the
  next Main Decoder step;
- complete generated codes; and
- stateful NanoCodec PCM and causal state at first, steady, and final
  boundaries.

The manifest records `local_ar_seed`, and the same explicit seed is used for
both the isolated first Local AR boundary and complete generation. Capture
fails if the first generated frame stack differs between those two paths.
Stochastic generation never infers a seed from process-global RNG state.

The output path must not exist. A failed capture removes its staging directory
instead of publishing a partial fixture. Before publishing the completed
directory, capture runs the same byte-level fixture validator described below.
The capture fixes `float32_matmul_precision=highest`,
`cuda_matmul_allow_tf32=false`, and `cudnn_allow_tf32=false` before model
execution and records all three values in `manifest.runtime`. The validator
rejects a missing or different precision policy; a fixture captured with
PyTorch's default cuDNN TF32 policy is not an IEEE FP32 parity oracle.

```bash
python3 tools/oracle/capture_boundary_fixture.py \
  --lock reference/oracle-lock.json \
  --speech-root /path/to/accepted/NeMo \
  --model /path/to/magpie_tts_multilingual_357m.nemo \
  --codec-model /path/to/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  --acceptance-receipt \
    reference/source-model-acceptance-sofia-v2-provenance.json \
  --local-ar-seed 20260729 \
  --output /new/path/magpie-v2607-sofia-ja-boundaries-v1
```

Validate a transferred or stored fixture with:

```bash
python3 tools/oracle/validate_boundary_fixture.py \
  --fixture /path/to/magpie-v2607-sofia-ja-boundaries-v1 \
  --lock reference/oracle-lock.json
```

`--lock` may be omitted when only the fixture's internal integrity is being
checked. When supplied, the manifest must reference those exact lock bytes.
The validator verifies the manifest checksum, normalized relative paths,
unique names and paths, dtype/shape byte counts, every tensor size and
SHA-256, and rejects symbolic links and every unlisted file or directory.

The accepted oracle emits text token tensors as `int32`. The fixture preserves
that dtype and declares it in both the frontend and decoder contracts; capture
does not widen the oracle boundary to match an engine implementation.

TensorRT plans cannot pass component parity using only summary statistics.
They must reproduce the locked boundary tensors within the tolerances declared
by the acceptance gate.

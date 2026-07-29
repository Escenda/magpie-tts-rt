# Reference oracle

The first implementation is based on MagpieTTS multilingual v2607 with the
Sofia speaker and the accepted native BF16 streaming path.

Pinned external assets:

| Asset | Identifier |
| --- | --- |
| NVIDIA NeMo Speech base revision | `9ae3e66b7314b0358c96bce47fbac56d78728bcd` |
| `magpie_tts_multilingual_357m.nemo` SHA-256 | `ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953` |
| initial streaming acceptance receipt SHA-256 | `35afc2eb229651266f6aa5d88afcdd1122d233016030dffa5acec9e5e9dc7061` |

The base revision alone does not identify the accepted optimized oracle because
the initial experiment contains additional local changes. No TensorRT output
may claim parity until those changes are captured as a clean, hash-addressed
source snapshot or replaced by independently implemented code with equivalent
golden fixtures.

The model is not part of this repository. Download and use it only under the
[NVIDIA Open Model License](https://huggingface.co/nvidia/magpie_tts_multilingual_357m).

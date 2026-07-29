# Source porting map

This map identifies the accepted PyTorch operations that must be replaced. It
does not authorize copying code without preserving its license and notices.

| Runtime component | Reference source responsibility | MagpieTTS-RT destination |
| --- | --- | --- |
| text preparation | language normalization, Japanese Katakana/accent G2P, tokenizer and spans | application frontend; token IDs enter C ABI |
| Text Encoder | token embedding, six causal Transformer layers, output normalization | Text Encoder TensorRT plan |
| speaker preparation | fixed speaker context embedding and AUDIO_BOS | immutable constants in the voice-specific prefill plan |
| Main Decoder prefill | prefix/condition projection, cross K/V creation, self K/V initialization, first alignment | prefill TensorRT plan |
| incremental decoder | one-position self/cross attention, explicit KV update, selected-layer alignment | one-step TensorRT plan and state plugins |
| attention prior | monotonic position tracking and next-step dynamic prior | C++ alignment controller |
| local autoregression | two-layer Local Transformer unrolled for 16 positions | fixed Local AR TensorRT plan |
| fused sampling | CFG, constraints, top-k, Gumbel RNG, token output, next embedding | TensorRT `IPluginV3` CUDA plugin |
| EOS | optional 32,384 projection and reduction against sampled codes | fused EOS CUDA plugin |
| codec conversion | codebook conversion and dequantization | NanoCodec TensorRT plan |
| causal waveform decode | stateful causal convolution, transposed-convolution overlap, residual histories | NanoCodec plan plus explicit-state plugins |
| streaming scheduler | first-4/steady-8/tail routing, CUDA events and bounded callbacks | C++ session worker and PCM lease ring |

The accepted experimental implementation currently extends these NeMo paths:

- `nemo/collections/tts/models/magpietts.py`
- `nemo/collections/tts/modules/transformer_2501.py`
- `nemo/collections/tts/modules/magpietts_modules.py`
- `nemo/collections/tts/modules/magpietts_fused_sampling.py`
- `nemo/collections/tts/modules/streaming_codec.py`
- `nemo/collections/tts/modules/streaming_synthesis.py`

Before exporting the first plan, capture a clean source snapshot or generate
golden boundary fixtures for every row above. A Git commit that omits local
extensions is not an adequate provenance identifier.

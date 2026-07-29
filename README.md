# MagpieTTS-RT

MagpieTTS-RT is an independent C++/CUDA/TensorRT runtime for low-latency
streaming inference with NVIDIA MagpieTTS. It is defining a versioned C ABI and
a Rust ownership wrapper so applications can use the runtime without embedding Python,
PyTorch, NeMo, ROS, or TensorRT types in their public interface.

This project is not an NVIDIA product and is not endorsed by or affiliated
with NVIDIA.

## Status

The repository is in its initial implementation phase. The versioned draft ABI, strict engine
bundle manifest, engine boundaries, and validation gates are being established
first. It does not yet provide a usable TensorRT synthesis engine and must not
be treated as production-ready.

The runtime fails closed when an engine, plugin, model asset, runtime
fingerprint, or required contract is missing or inconsistent. There is no CPU,
PyTorch, or eager inference fallback.

## Architecture

```text
reading normalization and tokenization
                 │ token IDs
                 ▼
┌──────────────── MagpieTTS-RT C ABI ─────────────────┐
│ Text Encoder TensorRT engine                        │
│ Main Decoder prefill TensorRT engine                │
│ Main Decoder one-step TensorRT engine               │
│ 16-position Local AR engine + CUDA plugins          │
│ fused sampling, EOS reduction, and next embedding   │
│ stateful NanoCodec TensorRT engine (4/8/tail)       │
│ C++ session: KV/RNG/alignment/CUDA streams/PCM ring │
└───────────────────────┬─────────────────────────────┘
                        │ leased PCM chunks
                        ▼
              application or Rust wrapper
```

Text normalization, Japanese reading conversion, tokenization, source-span
tracking, ROS topics, and physical audio playback are application concerns.
The C ABI accepts prepared token IDs and returns ordered mono PCM chunks with
explicit offsets and terminal state.

See:

- [Architecture](docs/architecture.md)
- [C ABI v1](docs/c-abi.md)
- [TensorRT engine contracts](docs/engine-contracts.md)
- [Runtime fingerprint](docs/runtime-fingerprint.md)
- [Acceptance gates](docs/acceptance.md)
- [Reference oracle](docs/reference-oracle.md)
- [Source porting map](docs/source-porting-map.md)

## Build the current foundation

The current build supports Linux only. Windows and macOS are not supported.
It requires a CUDA 13 toolkit, TensorRT 10 development packages, CMake 3.24
or newer, a C++20 compiler, Ninja, binutils, `nlohmann-json`, and the OpenSSL 3
development package (`libssl-dev` on Ubuntu). Both the build and installed
CMake package verify the CUDA Runtime and TensorRT ELF SONAME majors instead of
accepting an unversioned library with a different ABI.

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMAGPIE_TTS_RT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

On an AGX Thor with a working CUDA device, add
`-DMAGPIE_TTS_RT_BUILD_GPU_TESTS=ON`. The GPU smoke test creates the real CUDA
and TensorRT runtime but does not claim that model loading or synthesis is
implemented.

The Rust crates are pinned to Rust 1.88:

```bash
cd rust
cargo test --workspace --locked
```

To test the linked library, enable `native-link` and provide the absolute
directory containing `libmagpie_tts_rt.so` through
`MAGPIE_TTS_RT_LIB_DIR`.

Installed CMake consumers can set `MagpieTTSRT_TENSORRT_ROOT` and
`MagpieTTSRT_CUDA_ROOT` when TensorRT or the CUDA Runtime is outside the
system library search path. The pkg-config file intentionally exposes only
MagpieTTS-RT itself; pkg-config consumers must make CUDA Runtime 13 and
TensorRT 10 resolvable by the compiler, linker, and runtime loader.

## Licensing

Source code in this repository is licensed under Apache-2.0. MagpieTTS model
weights are not included and are governed separately by the
[NVIDIA Open Model License](https://huggingface.co/nvidia/magpie_tts_multilingual_357m).
Generated TensorRT plans are also excluded from source control and must be
loaded only after the application establishes a trust anchor and the runtime
verifies the bundle against it. Manifest hashes alone provide integrity, not
publisher authenticity.

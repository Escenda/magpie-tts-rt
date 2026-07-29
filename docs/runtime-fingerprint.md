# Runtime fingerprint

## Current implementation

The manifest parser validates the syntax of every runtime fingerprint field,
and `require_exact_runtime_fingerprint` compares two already populated
fingerprints field by field. The current contract-only runtime does not yet
collect a fingerprint from the active machine. It therefore does not yet
prove that a bundle was built for, or is safe to deserialize on, the running
system.

`model_load` remains `UNAVAILABLE` until the collector and its AGX Thor tests
exist. The canonical fixture is a schema test; it is not evidence that runtime
fingerprint collection is implemented.

## Proposed canonical sources

The collector is expected to use one source for each field:

| Manifest field | Candidate source |
| --- | --- |
| `os_name` | `ID` from `/etc/os-release` |
| `os_version` | `VERSION_ID` from `/etc/os-release` |
| `architecture` | `uname(2)` machine value, restricted to `aarch64` or `x86_64` |
| `endianness` | `std::endian::native` |
| `cuda_version` | the loaded CUDA Runtime through `cudaRuntimeGetVersion` |
| `tensorrt_version` | the loaded TensorRT library through `getInferLibVersion` |
| `driver_version` | the exact NVML system driver string |
| `gpu_name` | the CUDA device name for the selected runtime device |
| `gpu_compute_capability` | CUDA device major and minor capability |
| `plugin_abi_version` | the ABI constant exported by the verified plugin |

These sources are a design proposal, not an implemented compatibility
guarantee. The TensorRT API exposes major, minor, and patch through
`getInferLibVersion`, while distribution package build identifiers may contain
an additional component. Before model loading is enabled, the project must
decide whether that packaging build is a separate authenticated field or is
covered by a hash of the loaded TensorRT library and its dependency closure.

## Release blockers

The first engine-bearing release must additionally freeze and authenticate:

- the exporter source snapshot and patch set;
- TensorRT builder version and all builder flags;
- CUDA architecture targets and tactic sources;
- plugin compiler, linker, and CUDA flags;
- hashes for the runtime libraries used by the accepted AGX Thor build; and
- the exact formatter for every fingerprint value.

Until those values have one tested source and format, a matching string in a
manifest must not be presented as proof of runtime compatibility.

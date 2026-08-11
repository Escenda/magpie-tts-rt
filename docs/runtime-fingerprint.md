# Runtime fingerprint

## Current implementation

The manifest parser validates every runtime fingerprint field, and
`require_exact_runtime_fingerprint` compares two populated fingerprints field
by field. The native collector now reads the active machine through these
fixed sources:

| Manifest field | Canonical source |
| --- | --- |
| `os_name` | fixed `linux` after a successful Linux build and host check |
| `os_version` | `ID-VERSION_ID` from `/etc/os-release`; version 1 accepts Ubuntu only |
| `architecture` | `uname(2)` machine value, restricted to `aarch64` or `x86_64` |
| `endianness` | `std::endian::native` |
| `cuda_version` | the loaded CUDA Runtime through `cudaRuntimeGetVersion` |
| `cublas.api_version_integer` | `cublasGetVersion_v2` and `cublasLtGetVersion`, resolved through the authenticated plugin; both must return the same positive integer |
| `cublas.library` | fixed SONAME `libcublas.so.13`, byte size, and SHA-256 of the exact mapped regular file that provides `cublasGetVersion_v2` |
| `cublas.lt_library` | fixed SONAME `libcublasLt.so.13`, byte size, and SHA-256 of the exact mapped regular file that provides `cublasLtGetVersion` |
| `tensorrt_version` | the loaded TensorRT library's major, minor, patch, and build accessors |
| `driver_version` | the exact NVML system driver string |
| `gpu_name` | the CUDA device name for the selected runtime device |
| `gpu_compute_capability` | CUDA device major and minor capability |
| `plugin_abi_version` | the ABI constant exported by the verified plugin |

The cuBLAS collector does not perform a second loader search. It resolves the
four required API symbols through the already authenticated plugin handle,
uses `dladdr` to identify their providers, and matches each provider mapping
from `/proc/self/maps` to an `O_NOFOLLOW` descriptor by device and inode. It
hashes that descriptor and requires stable `fstat` identity before and after
the read. A missing symbol, unexpected provider name, non-file mapping,
device/inode mismatch, unstable file, API-version mismatch, or receipt/hash
mismatch fails model loading. Paths are deliberately excluded from the
fingerprint because installation paths are not runtime identities.

`model_load` first verifies the externally anchored manifest and creates sealed
snapshots of every artifact, including the plugin build receipt. It then loads
the authenticated plugin from its sealed descriptor, obtains its exported ABI
value, collects this fingerprint, requires an exact manifest match, and only
then deserializes the seven verified plans. A mismatch prevents model
creation.

The fingerprint proves compatibility with the values recorded by the accepted
bundle. It cryptographically measures the loaded cuBLAS and cuBLASLt files,
but it does not authenticate the publisher or measure every host library; the
external manifest trust anchor and the deployment's trusted host remain
separate requirements.

## Required release evidence

The current packager authenticates the final plugin library hash, ABI creator
set, SONAME, dynamic dependency allowlist, and the seven plan hashes. The
benchmark runner records the exact runtime, CUDA Runtime, and NVML library
hashes when it is run. `tools/plugins/create_plugin_build_receipt.py` now
produces a separate immutable receipt binding the plugin source,
compiler/linker versions, CUDA architecture and flags, ELF dependency
contract, and artifact hash. Final acceptance requires that receipt, binds its
hash and size into the consolidated export receipt, and the bundle packager
validates and includes it. The complete list below remains the release
evidence contract.

An engine-bearing release additionally records and authenticates:

- the exporter source snapshot and patch set;
- TensorRT builder version, builder flags, profiles, and plan hashes;
- CUDA architecture targets and tactic sources;
- plugin source, compiler/linker/CUDA settings, dynamic dependency contract,
  library hash, and loaded cuBLAS/cuBLASLt API and file identities;
- the source-model and tokenizer identities;
- the exact runtime, CUDA Runtime, and NVML library hashes used by the Thor
  acceptance runner; and
- the exact formatter and source for every fingerprint value.

Before a production release, those records must be checked by the bundle
packager, native startup gate, and Thor benchmark receipt as applicable. A
matching fingerprint string by itself is never reported as proof that a bundle
is trusted or accepted.

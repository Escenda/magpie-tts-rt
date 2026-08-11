#include "runtime/fingerprint.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#include <dlfcn.h>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  try {
    void* const cublas_scope =
        ::dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
    require(cublas_scope != nullptr, "unable to load cuBLAS test scope");
    const auto cublas_identity =
        magpie_tts_rt::collect_cublas_runtime_identity(cublas_scope);
    const magpie_tts_rt::RuntimeFingerprint fingerprint =
        magpie_tts_rt::collect_runtime_fingerprint(
            0, 1, cublas_identity);
    require(fingerprint.os_name == "linux", "unexpected os_name");
    require(
        fingerprint.os_version == "ubuntu-24.04",
        "unexpected os_version");
    require(
        fingerprint.architecture == "aarch64",
        "unexpected architecture");
    require(
        fingerprint.endianness == magpie_tts_rt::Endianness::little,
        "unexpected endianness");
    require(fingerprint.cuda_version == "13.2", "unexpected CUDA version");
    require(
        fingerprint.cublas.api_version_integer == 130400,
        "unexpected cuBLAS API version");
    require(
        fingerprint.cublas.library.soname == "libcublas.so.13" &&
            fingerprint.cublas.library.size_bytes == 67751616U &&
            fingerprint.cublas.library.sha256 ==
                "826486b8869144621e3a477cddcd28f56733c7c80c6f998b898384fc09e10f91",
        "unexpected cuBLAS artifact identity");
    require(
        fingerprint.cublas.lt_library.soname == "libcublasLt.so.13" &&
            fingerprint.cublas.lt_library.size_bytes == 606744240U &&
            fingerprint.cublas.lt_library.sha256 ==
                "b7aa42c190c2e7490abd6ea987883e05678e26222b7f9f1c9b96374fcbddbf04",
        "unexpected cuBLASLt artifact identity");
    require(
        fingerprint.tensorrt_version == "10.16.2.10",
        "unexpected TensorRT version");
    require(
        fingerprint.driver_version == "595.78",
        "unexpected NVIDIA driver version");
    require(fingerprint.gpu_name == "NVIDIA Thor", "unexpected GPU name");
    require(
        fingerprint.gpu_compute_capability == "11.0",
        "unexpected compute capability");
    require(
        fingerprint.plugin_abi_version == 1,
        "unexpected plugin ABI version");
    require(::dlclose(cublas_scope) == 0, "unable to close cuBLAS test scope");
  } catch (const std::exception& error) {
    std::cerr << "fingerprint GPU test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

#include "runtime/fingerprint.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  try {
    const magpie_tts_rt::RuntimeFingerprint fingerprint =
        magpie_tts_rt::collect_runtime_fingerprint(0, 1);
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
  } catch (const std::exception& error) {
    std::cerr << "fingerprint GPU test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

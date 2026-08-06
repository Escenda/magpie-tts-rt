#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

struct NvmlMemoryV2 {
  unsigned int version;
  unsigned long long total;
  unsigned long long reserved;
  unsigned long long free;
  unsigned long long used;
};

struct NvmlDeviceOpaque {
  std::uint32_t index = 0U;
};

constexpr int kSuccess = 0;
constexpr int kInvalidArgument = 2;
constexpr int kNotSupported = 3;
constexpr unsigned int kMemoryV2Version =
    static_cast<unsigned int>(
        sizeof(NvmlMemoryV2) | (2U << 24U));

NvmlDeviceOpaque g_device;
bool g_initialized = false;

}  // namespace

extern "C" int nvmlInit_v2() {
  g_initialized = true;
  return kSuccess;
}

extern "C" int nvmlShutdown() {
  if (!g_initialized) {
    return kInvalidArgument;
  }
  g_initialized = false;
  return kSuccess;
}

extern "C" int nvmlDeviceGetHandleByIndex_v2(
    const unsigned int index, NvmlDeviceOpaque** const device) {
  if (!g_initialized || device == nullptr || index != 0U) {
    return kInvalidArgument;
  }
  g_device.index = index;
  *device = &g_device;
  return kSuccess;
}

extern "C" int nvmlDeviceGetMemoryInfo_v2(
    NvmlDeviceOpaque* const device, NvmlMemoryV2* const memory) {
  if (!g_initialized || device != &g_device || memory == nullptr ||
      memory->version != kMemoryV2Version) {
    return kInvalidArgument;
  }
  return kNotSupported;
}

extern "C" int nvmlSystemGetDriverVersion(
    char* const version, const unsigned int length) {
  constexpr std::array<char, 14U> kVersion{
      'm', 'o', 'c', 'k', '-', 'd', 'r', 'i', 'v', 'e', 'r', '-',
      '1', '\0'};
  if (!g_initialized || version == nullptr ||
      length < kVersion.size()) {
    return kInvalidArgument;
  }
  std::memcpy(version, kVersion.data(), kVersion.size());
  return kSuccess;
}

extern "C" int nvmlDeviceGetName(
    NvmlDeviceOpaque* const device, char* const name,
    const unsigned int length) {
  constexpr std::array<char, 12U> kName{
      'N', 'V', 'I', 'D', 'I', 'A', ' ', 'T', 'h', 'o', 'r', '\0'};
  if (!g_initialized || device != &g_device || name == nullptr ||
      length < kName.size()) {
    return kInvalidArgument;
  }
  std::memcpy(name, kName.data(), kName.size());
  return kSuccess;
}

extern "C" int nvmlDeviceGetUUID(
    NvmlDeviceOpaque* const device, char* const uuid,
    const unsigned int length) {
  constexpr std::array<char, 14U> kUuid{
      'G', 'P', 'U', '-', 'm', 'o', 'c', 'k', '-', 't', 'h', 'o', 'r',
      '\0'};
  if (!g_initialized || device != &g_device || uuid == nullptr ||
      length < kUuid.size()) {
    return kInvalidArgument;
  }
  std::memcpy(uuid, kUuid.data(), kUuid.size());
  return kSuccess;
}

extern "C" int nvmlDeviceGetCudaComputeCapability(
    NvmlDeviceOpaque* const device, int* const major,
    int* const minor) {
  if (!g_initialized || device != &g_device || major == nullptr ||
      minor == nullptr) {
    return kInvalidArgument;
  }
  *major = 11;
  *minor = 0;
  return kSuccess;
}

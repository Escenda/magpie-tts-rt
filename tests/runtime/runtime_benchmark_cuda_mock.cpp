#include <cstddef>

namespace {

constexpr int kCudaSuccess = 0;
constexpr int kCudaInvalidValue = 1;
constexpr std::size_t kTotalBytes =
    64ULL * 1024ULL * 1024ULL * 1024ULL;

bool g_device_selected = false;
std::size_t g_query_count = 0U;

}  // namespace

extern "C" int cudaSetDevice(const int device) {
  if (device != 0) {
    return kCudaInvalidValue;
  }
  g_device_selected = true;
  return kCudaSuccess;
}

extern "C" int cudaMemGetInfo(
    std::size_t* const free_bytes,
    std::size_t* const total_bytes) {
  if (!g_device_selected || free_bytes == nullptr ||
      total_bytes == nullptr) {
    return kCudaInvalidValue;
  }
  const auto used =
      2ULL * 1024ULL * 1024ULL * 1024ULL +
      g_query_count * 4'096ULL;
  ++g_query_count;
  *total_bytes = kTotalBytes;
  *free_bytes = kTotalBytes - used;
  return kCudaSuccess;
}

extern "C" int cudaDriverGetVersion(int* const version) {
  if (!g_device_selected || version == nullptr) {
    return kCudaInvalidValue;
  }
  *version = 13'020;
  return kCudaSuccess;
}

extern "C" int cudaRuntimeGetVersion(int* const version) {
  if (!g_device_selected || version == nullptr) {
    return kCudaInvalidValue;
  }
  *version = 13'020;
  return kCudaSuccess;
}

extern "C" const char* cudaGetErrorString(const int status) {
  return status == kCudaSuccess ? "success" : "invalid value";
}

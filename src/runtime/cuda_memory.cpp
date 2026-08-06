#include "runtime/cuda_memory.hpp"

#include <limits>
#include <utility>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const CudaMemoryErrorCode code,
    const std::string_view allocation_name,
    const std::string_view detail) {
  return "CUDA memory contract failed [code=" +
         std::string(to_string(code)) + ", allocation=" +
         std::string(allocation_name) + "]: " + std::string(detail);
}

[[noreturn]] void fail(
    const CudaMemoryErrorCode code,
    const std::string_view allocation_name,
    const std::string& detail) {
  throw CudaMemoryError(
      code, std::string(allocation_name), detail);
}

[[nodiscard]] std::size_t checked_size(
    const std::uint64_t size_bytes,
    const std::string_view allocation_name) {
  if (size_bytes == 0) {
    fail(
        CudaMemoryErrorCode::zero_size,
        allocation_name,
        "allocation size must be positive");
  }
  if (size_bytes >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max())) {
    fail(
        CudaMemoryErrorCode::allocation_failed,
        allocation_name,
        "allocation size does not fit size_t");
  }
  return static_cast<std::size_t>(size_bytes);
}

}  // namespace

std::string_view to_string(const CudaMemoryErrorCode code) noexcept {
  switch (code) {
    case CudaMemoryErrorCode::zero_size:
      return "zero_size";
    case CudaMemoryErrorCode::budget_exceeded:
      return "budget_exceeded";
    case CudaMemoryErrorCode::duplicate_name:
      return "duplicate_name";
    case CudaMemoryErrorCode::unknown_name:
      return "unknown_name";
    case CudaMemoryErrorCode::allocation_failed:
      return "allocation_failed";
    case CudaMemoryErrorCode::host_allocation_failed:
      return "host_allocation_failed";
  }
  return "unknown";
}

CudaMemoryError::CudaMemoryError(
    const CudaMemoryErrorCode code,
    std::string allocation_name,
    std::string detail)
    : std::runtime_error(
          error_message(code, allocation_name, detail)),
      code_(code),
      allocation_name_(std::move(allocation_name)),
      detail_(std::move(detail)) {}

CudaMemoryErrorCode CudaMemoryError::code() const noexcept {
  return code_;
}

const std::string& CudaMemoryError::allocation_name() const noexcept {
  return allocation_name_;
}

const std::string& CudaMemoryError::detail() const noexcept {
  return detail_;
}

DeviceAllocation::DeviceAllocation(const std::uint64_t size_bytes)
    : size_bytes_(size_bytes) {
  const std::size_t size = checked_size(size_bytes, "device");
  const cudaError_t status = cudaMalloc(&pointer_, size);
  if (status != cudaSuccess) {
    pointer_ = nullptr;
    size_bytes_ = 0;
    fail(
        CudaMemoryErrorCode::allocation_failed,
        "device",
        cudaGetErrorString(status));
  }
}

DeviceAllocation::~DeviceAllocation() {
  if (pointer_ != nullptr) {
    static_cast<void>(cudaFree(pointer_));
  }
}

DeviceAllocation::DeviceAllocation(DeviceAllocation&& other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

DeviceAllocation& DeviceAllocation::operator=(
    DeviceAllocation&& other) noexcept {
  if (this != &other) {
    if (pointer_ != nullptr) {
      static_cast<void>(cudaFree(pointer_));
    }
    pointer_ = std::exchange(other.pointer_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
  }
  return *this;
}

void* DeviceAllocation::data() noexcept { return pointer_; }

const void* DeviceAllocation::data() const noexcept {
  return pointer_;
}

std::uint64_t DeviceAllocation::size_bytes() const noexcept {
  return size_bytes_;
}

PinnedAllocation::PinnedAllocation(const std::uint64_t size_bytes)
    : size_bytes_(size_bytes) {
  const std::size_t size = checked_size(size_bytes, "pinned_host");
  const cudaError_t status = cudaHostAlloc(
      &pointer_, size, cudaHostAllocPortable);
  if (status != cudaSuccess) {
    pointer_ = nullptr;
    size_bytes_ = 0;
    fail(
        CudaMemoryErrorCode::host_allocation_failed,
        "pinned_host",
        cudaGetErrorString(status));
  }
}

PinnedAllocation::~PinnedAllocation() {
  if (pointer_ != nullptr) {
    static_cast<void>(cudaFreeHost(pointer_));
  }
}

PinnedAllocation::PinnedAllocation(PinnedAllocation&& other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

PinnedAllocation& PinnedAllocation::operator=(
    PinnedAllocation&& other) noexcept {
  if (this != &other) {
    if (pointer_ != nullptr) {
      static_cast<void>(cudaFreeHost(pointer_));
    }
    pointer_ = std::exchange(other.pointer_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
  }
  return *this;
}

void* PinnedAllocation::data() noexcept { return pointer_; }

const void* PinnedAllocation::data() const noexcept {
  return pointer_;
}

std::uint64_t PinnedAllocation::size_bytes() const noexcept {
  return size_bytes_;
}

DeviceMemoryRegistry::DeviceMemoryRegistry(
    const std::uint64_t maximum_bytes)
    : maximum_bytes_(maximum_bytes) {
  if (maximum_bytes_ == 0) {
    fail(
        CudaMemoryErrorCode::zero_size,
        "registry",
        "device-memory budget must be positive");
  }
}

void* DeviceMemoryRegistry::allocate(
    std::string name,
    const std::uint64_t size_bytes) {
  if (name.empty()) {
    fail(
        CudaMemoryErrorCode::unknown_name,
        "",
        "allocation name must be non-empty");
  }
  if (allocations_.contains(name)) {
    fail(
        CudaMemoryErrorCode::duplicate_name,
        name,
        "allocation name is already present");
  }
  if (size_bytes == 0 ||
      size_bytes > maximum_bytes_ - allocated_bytes_) {
    fail(
        size_bytes == 0 ? CudaMemoryErrorCode::zero_size
                        : CudaMemoryErrorCode::budget_exceeded,
        name,
        "requested " + std::to_string(size_bytes) +
            " bytes with " +
            std::to_string(maximum_bytes_ - allocated_bytes_) +
            " bytes remaining");
  }
  auto [iterator, inserted] = allocations_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(std::move(name)),
      std::forward_as_tuple(size_bytes));
  if (!inserted) {
    fail(
        CudaMemoryErrorCode::duplicate_name,
        iterator->first,
        "allocation insertion failed");
  }
  allocated_bytes_ += size_bytes;
  return iterator->second.data();
}

void* DeviceMemoryRegistry::require(const std::string_view name) {
  const auto found = allocations_.find(std::string(name));
  if (found == allocations_.end()) {
    fail(
        CudaMemoryErrorCode::unknown_name,
        name,
        "allocation does not exist");
  }
  return found->second.data();
}

const void* DeviceMemoryRegistry::require(
    const std::string_view name) const {
  const auto found = allocations_.find(std::string(name));
  if (found == allocations_.end()) {
    fail(
        CudaMemoryErrorCode::unknown_name,
        name,
        "allocation does not exist");
  }
  return found->second.data();
}

std::uint64_t DeviceMemoryRegistry::allocation_size(
    const std::string_view name) const {
  const auto found = allocations_.find(std::string(name));
  if (found == allocations_.end()) {
    fail(
        CudaMemoryErrorCode::unknown_name,
        name,
        "allocation does not exist");
  }
  return found->second.size_bytes();
}

std::uint64_t DeviceMemoryRegistry::allocated_bytes() const noexcept {
  return allocated_bytes_;
}

std::uint64_t DeviceMemoryRegistry::maximum_bytes() const noexcept {
  return maximum_bytes_;
}

}  // namespace magpie_tts_rt

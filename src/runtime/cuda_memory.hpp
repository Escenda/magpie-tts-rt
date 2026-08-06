#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <cuda_runtime_api.h>

namespace magpie_tts_rt {

enum class CudaMemoryErrorCode {
  zero_size,
  budget_exceeded,
  duplicate_name,
  unknown_name,
  allocation_failed,
  host_allocation_failed,
};

[[nodiscard]] std::string_view to_string(
    CudaMemoryErrorCode code) noexcept;

class CudaMemoryError final : public std::runtime_error {
 public:
  CudaMemoryError(
      CudaMemoryErrorCode code,
      std::string allocation_name,
      std::string detail);

  [[nodiscard]] CudaMemoryErrorCode code() const noexcept;
  [[nodiscard]] const std::string& allocation_name() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  CudaMemoryErrorCode code_;
  std::string allocation_name_;
  std::string detail_;
};

class DeviceAllocation final {
 public:
  explicit DeviceAllocation(std::uint64_t size_bytes);
  ~DeviceAllocation();

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  DeviceAllocation(DeviceAllocation&& other) noexcept;
  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept;

  [[nodiscard]] void* data() noexcept;
  [[nodiscard]] const void* data() const noexcept;
  [[nodiscard]] std::uint64_t size_bytes() const noexcept;

 private:
  void* pointer_{nullptr};
  std::uint64_t size_bytes_{0};
};

class PinnedAllocation final {
 public:
  explicit PinnedAllocation(std::uint64_t size_bytes);
  ~PinnedAllocation();

  PinnedAllocation(const PinnedAllocation&) = delete;
  PinnedAllocation& operator=(const PinnedAllocation&) = delete;
  PinnedAllocation(PinnedAllocation&& other) noexcept;
  PinnedAllocation& operator=(PinnedAllocation&& other) noexcept;

  [[nodiscard]] void* data() noexcept;
  [[nodiscard]] const void* data() const noexcept;
  [[nodiscard]] std::uint64_t size_bytes() const noexcept;

 private:
  void* pointer_{nullptr};
  std::uint64_t size_bytes_{0};
};

class DeviceMemoryRegistry final {
 public:
  explicit DeviceMemoryRegistry(std::uint64_t maximum_bytes);

  [[nodiscard]] void* allocate(
      std::string name,
      std::uint64_t size_bytes);
  [[nodiscard]] void* require(std::string_view name);
  [[nodiscard]] const void* require(std::string_view name) const;
  [[nodiscard]] std::uint64_t allocation_size(
      std::string_view name) const;
  [[nodiscard]] std::uint64_t allocated_bytes() const noexcept;
  [[nodiscard]] std::uint64_t maximum_bytes() const noexcept;

 private:
  std::uint64_t maximum_bytes_;
  std::uint64_t allocated_bytes_{0};
  std::unordered_map<std::string, DeviceAllocation> allocations_;
};

}  // namespace magpie_tts_rt

#include "runtime/synthesis_kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_cuda(
    const cudaError_t status,
    const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        operation + ": " + cudaGetErrorString(status));
  }
}

template <typename Value>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(const std::size_t count) {
    require_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&pointer_),
            count * sizeof(Value)),
        "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      static_cast<void>(cudaFree(pointer_));
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] Value* get() const noexcept { return pointer_; }

 private:
  Value* pointer_{nullptr};
};

void test_cfg() {
  constexpr std::uint32_t tokens = 3;
  constexpr std::uint32_t width = 768;
  std::vector<__nv_bfloat16> input(tokens * width);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = __float2bfloat16_rn(
        static_cast<float>(index % 31U));
  }
  const std::array<bool, tokens> mask{true, true, false};
  DeviceBuffer<__nv_bfloat16> device_input(input.size());
  DeviceBuffer<bool> device_mask(tokens);
  DeviceBuffer<__nv_bfloat16> device_output(input.size() * 2U);
  DeviceBuffer<bool> device_cfg_mask(tokens * 2U);
  require_cuda(
      cudaMemcpy(
          device_input.get(),
          input.data(),
          input.size() * sizeof(__nv_bfloat16),
          cudaMemcpyHostToDevice),
      "copy condition");
  require_cuda(
      cudaMemcpy(
          device_mask.get(),
          mask.data(),
          mask.size() * sizeof(bool),
          cudaMemcpyHostToDevice),
      "copy mask");
  require_cuda(
      magpie_tts_rt::launch_prepare_cfg_inputs(
          device_input.get(),
          device_mask.get(),
          tokens,
          device_output.get(),
          device_cfg_mask.get(),
          nullptr),
      "launch CFG preparation");
  std::vector<__nv_bfloat16> output(input.size() * 2U);
  std::array<bool, tokens * 2U> cfg_mask{};
  require_cuda(
      cudaMemcpy(
          output.data(),
          device_output.get(),
          output.size() * sizeof(__nv_bfloat16),
          cudaMemcpyDeviceToHost),
      "copy CFG condition");
  require_cuda(
      cudaMemcpy(
          cfg_mask.data(),
          device_cfg_mask.get(),
          cfg_mask.size() * sizeof(bool),
          cudaMemcpyDeviceToHost),
      "copy CFG mask");
  for (std::size_t index = 0; index < input.size(); ++index) {
    require(
        output[index] == input[index],
        "conditional CFG row changed");
    require(
        __bfloat162float(output[input.size() + index]) == 0.0F,
        "unconditional CFG row is not zero");
  }
  require(
      cfg_mask ==
          std::array<bool, tokens * 2U>{
              true, true, false, true, false, false},
      "CFG mask contract mismatch");
}

void test_codes_and_eos_latch() {
  std::array<std::int64_t, 16> step{};
  for (std::size_t index = 0; index < step.size(); ++index) {
    step[index] = static_cast<std::int64_t>(100U + index);
  }
  DeviceBuffer<std::int64_t> device_step(step.size());
  DeviceBuffer<std::int64_t> aggregate(64);
  require_cuda(
      cudaMemcpy(
          device_step.get(),
          step.data(),
          step.size() * sizeof(std::int64_t),
          cudaMemcpyHostToDevice),
      "copy step codes");
  require_cuda(
      cudaMemset(
          aggregate.get(), 0, 64 * sizeof(std::int64_t)),
      "clear aggregate codes");
  require_cuda(
      magpie_tts_rt::launch_append_codec_step(
          device_step.get(), aggregate.get(), 4, nullptr),
      "append step codes");
  std::array<std::int64_t, 64> host_aggregate{};
  require_cuda(
      cudaMemcpy(
          host_aggregate.data(),
          aggregate.get(),
          host_aggregate.size() * sizeof(std::int64_t),
          cudaMemcpyDeviceToHost),
      "copy aggregate codes");
  for (std::size_t codebook = 0; codebook < 8; ++codebook) {
    require(
        host_aggregate[codebook * 8U + 4U] ==
                step[codebook * 2U] &&
            host_aggregate[codebook * 8U + 5U] ==
                step[codebook * 2U + 1U],
        "codebook-major append mismatch");
  }
  DeviceBuffer<std::int64_t> packed(40);
  require_cuda(
      magpie_tts_rt::launch_pack_codec_frames(
          aggregate.get(), packed.get(), 5, nullptr),
      "pack codec frames");
  std::array<std::int64_t, 40> host_packed{};
  require_cuda(
      cudaMemcpy(
          host_packed.data(),
          packed.get(),
          host_packed.size() * sizeof(std::int64_t),
          cudaMemcpyDeviceToHost),
      "copy packed codes");
  for (std::size_t codebook = 0; codebook < 8; ++codebook) {
    for (std::size_t frame = 0; frame < 5; ++frame) {
      require(
          host_packed[codebook * 5U + frame] ==
              host_aggregate[codebook * 8U + frame],
          "packed codec frame mismatch");
    }
  }

  DeviceBuffer<std::int32_t> eos(1);
  DeviceBuffer<bool> finished(1);
  const std::int32_t eos_one = 1;
  const bool false_value = false;
  require_cuda(
      cudaMemcpy(
          eos.get(),
          &eos_one,
          sizeof(eos_one),
          cudaMemcpyHostToDevice),
      "copy EOS");
  require_cuda(
      cudaMemcpy(
          finished.get(),
          &false_value,
          sizeof(false_value),
          cudaMemcpyHostToDevice),
      "clear finished");
  require_cuda(
      magpie_tts_rt::launch_latch_generation_finished(
          eos.get(), finished.get(), nullptr),
      "latch EOS");
  bool host_finished = false;
  require_cuda(
      cudaMemcpy(
          &host_finished,
          finished.get(),
          sizeof(host_finished),
          cudaMemcpyDeviceToHost),
      "copy finished");
  require(host_finished, "EOS was not latched");
}

}  // namespace

int main() {
  try {
    test_cfg();
    test_codes_and_eos_latch();
  } catch (const std::exception& error) {
    std::cerr << "synthesis kernel GPU test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

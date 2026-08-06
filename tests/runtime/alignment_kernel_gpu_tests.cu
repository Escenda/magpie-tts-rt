#include "runtime/alignment_controller.hpp"
#include "runtime/alignment_kernel.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using magpie_tts_rt::AlignmentControllerReference;
using magpie_tts_rt::AlignmentKernelContract;
using magpie_tts_rt::AlignmentManifest;
using magpie_tts_rt::TensorDataType;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_cuda(const cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        operation + ": " + cudaGetErrorString(status));
  }
}

template <typename Value>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    require_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&pointer_),
            count_ * sizeof(Value)),
        "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      static_cast<void>(cudaFree(pointer_));
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] Value* get() noexcept { return pointer_; }

 private:
  Value* pointer_{nullptr};
  std::size_t count_;
};

[[nodiscard]] AlignmentManifest manifest() {
  return AlignmentManifest{
      .dtype = TensorDataType::bf16,
      .source_decoder_layers = {4, 5, 8, 9},
      .prefill_output_binding = "alignment",
      .step_prior_input_binding = "alignment_prior",
      .step_alignment_output_binding = "alignment",
      .prior_epsilon = 0.1,
      .initial_attended = 1,
      .ignored_terminal_tokens = 3,
      .short_text_no_prior_max_tokens = 5,
      .lookahead = 6,
      .sink_threshold = 4,
      .source_position_policy = "exact_frontend_span_only",
  };
}

void test_gpu_matches_reference() {
  constexpr std::uint32_t text_length = 39;
  AlignmentControllerReference reference(text_length, manifest());
  const AlignmentKernelContract contract{
      .text_length = text_length,
      .ignored_terminal_tokens = 3,
      .short_text_no_prior_max_tokens = 5,
      .lookahead = 6,
      .sink_threshold = 4,
  };
  DeviceBuffer<__nv_bfloat16> scores(text_length * 2U);
  DeviceBuffer<std::uint32_t> counters(text_length);
  DeviceBuffer<std::uint32_t> last_attended(1);
  DeviceBuffer<__nv_bfloat16> prior(text_length * 2U);
  DeviceBuffer<std::int64_t> attended(1);
  DeviceBuffer<bool> unfinished(1);
  DeviceBuffer<std::int32_t> invalid(1);
  require_cuda(
      cudaMemset(
          counters.get(), 0, text_length * sizeof(std::uint32_t)),
      "clear counters");
  const std::uint32_t initial_attended = 1;
  require_cuda(
      cudaMemcpy(
          last_attended.get(),
          &initial_attended,
          sizeof(initial_attended),
          cudaMemcpyHostToDevice),
      "initialize attended");

  for (std::uint32_t step = 0; step < 12; ++step) {
    std::vector<float> host_scores(text_length, -2.0F);
    const std::uint32_t peak =
        std::min<std::uint32_t>(2U + step / 2U, 30U);
    host_scores.at(peak) = 4.0F;
    host_scores.at(std::min(peak + 1U, text_length - 1U)) = 3.0F;
    std::vector<__nv_bfloat16> host_bf16(text_length * 2U);
    std::vector<float> rounded(text_length);
    for (std::uint32_t index = 0; index < text_length; ++index) {
      host_bf16.at(index) =
          __float2bfloat16_rn(host_scores.at(index));
      host_bf16.at(text_length + index) =
          __float2bfloat16_rn(0.0F);
      rounded.at(index) =
          __bfloat162float(host_bf16.at(index));
    }
    require_cuda(
        cudaMemcpy(
            scores.get(),
            host_bf16.data(),
            host_bf16.size() * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice),
        "copy scores");
    require_cuda(
        magpie_tts_rt::launch_alignment_controller(
            scores.get(),
            contract,
            counters.get(),
            last_attended.get(),
            prior.get(),
            attended.get(),
            unfinished.get(),
            invalid.get(),
            nullptr),
        "launch alignment");
    require_cuda(cudaDeviceSynchronize(), "synchronize alignment");

    std::int64_t host_attended = -1;
    bool host_unfinished = false;
    std::int32_t host_invalid = -1;
    std::vector<__nv_bfloat16> host_prior(text_length * 2U);
    require_cuda(
        cudaMemcpy(
            &host_attended,
            attended.get(),
            sizeof(host_attended),
            cudaMemcpyDeviceToHost),
        "copy attended");
    require_cuda(
        cudaMemcpy(
            &host_unfinished,
            unfinished.get(),
            sizeof(host_unfinished),
            cudaMemcpyDeviceToHost),
        "copy unfinished");
    require_cuda(
        cudaMemcpy(
            &host_invalid,
            invalid.get(),
            sizeof(host_invalid),
            cudaMemcpyDeviceToHost),
        "copy invalid");
    require_cuda(
        cudaMemcpy(
            host_prior.data(),
            prior.get(),
            host_prior.size() * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost),
        "copy prior");

    const auto expected = reference.advance(rounded);
    require(host_invalid == 0, "GPU alignment reported invalid");
    require(
        host_attended ==
            static_cast<std::int64_t>(
                expected.attended_token_index),
        "attended mismatch");
    require(
        host_unfinished == expected.unfinished_text,
        "unfinished mismatch");
    for (std::size_t index = 0; index < host_prior.size(); ++index) {
      const float actual =
          __bfloat162float(host_prior.at(index));
      const float expected_bf16 = __bfloat162float(
          __float2bfloat16_rn(expected.next_prior.at(index)));
      require(actual == expected_bf16, "prior mismatch");
    }
  }
}

void test_non_finite_fails_closed() {
  constexpr std::uint32_t text_length = 8;
  DeviceBuffer<__nv_bfloat16> scores(text_length * 2U);
  DeviceBuffer<std::uint32_t> counters(text_length);
  DeviceBuffer<std::uint32_t> last_attended(1);
  DeviceBuffer<__nv_bfloat16> prior(text_length * 2U);
  DeviceBuffer<std::int64_t> attended(1);
  DeviceBuffer<bool> unfinished(1);
  DeviceBuffer<std::int32_t> invalid(1);
  std::vector<__nv_bfloat16> host(text_length * 2U);
  for (auto& value : host) {
    value = __float2bfloat16_rn(0.0F);
  }
  host.at(7) = __float2bfloat16_rn(NAN);
  require_cuda(
      cudaMemcpy(
          scores.get(),
          host.data(),
          host.size() * sizeof(__nv_bfloat16),
          cudaMemcpyHostToDevice),
      "copy non-finite scores");
  require_cuda(
      cudaMemset(
          counters.get(), 0, text_length * sizeof(std::uint32_t)),
      "clear counters");
  const std::uint32_t initial = 1;
  require_cuda(
      cudaMemcpy(
          last_attended.get(),
          &initial,
          sizeof(initial),
          cudaMemcpyHostToDevice),
      "initialize attended");
  const AlignmentKernelContract contract{
      .text_length = text_length,
      .ignored_terminal_tokens = 3,
      .short_text_no_prior_max_tokens = 5,
      .lookahead = 6,
      .sink_threshold = 4,
  };
  require_cuda(
      magpie_tts_rt::launch_alignment_controller(
          scores.get(),
          contract,
          counters.get(),
          last_attended.get(),
          prior.get(),
          attended.get(),
          unfinished.get(),
          invalid.get(),
          nullptr),
      "launch invalid alignment");
  std::int32_t host_invalid = 0;
  require_cuda(
      cudaMemcpy(
          &host_invalid,
          invalid.get(),
          sizeof(host_invalid),
          cudaMemcpyDeviceToHost),
      "copy invalid status");
  require(host_invalid != 0, "non-finite score was accepted");
}

}  // namespace

int main() {
  try {
    test_gpu_matches_reference();
    test_non_finite_fails_closed();
  } catch (const std::exception& error) {
    std::cerr << "alignment kernel GPU test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

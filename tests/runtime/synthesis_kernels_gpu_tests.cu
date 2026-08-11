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

void test_local_step_finalization() {
  std::array<std::int64_t, 16> step{};
  for (std::size_t index = 0; index < step.size(); ++index) {
    step[index] = static_cast<std::int64_t>(100U + index);
  }
  DeviceBuffer<std::int64_t> device_step(step.size());
  DeviceBuffer<std::int64_t> aggregate(64);
  DeviceBuffer<std::int32_t> canonical_invalid(1);
  DeviceBuffer<std::int32_t> canonical_end(1);
  DeviceBuffer<std::int32_t> step_invalid(1);
  DeviceBuffer<std::int32_t> step_end(1);
  DeviceBuffer<std::int64_t> updated_counter(1);
  DeviceBuffer<std::int64_t> counter(1);
  DeviceBuffer<bool> finished(1);
  const std::int32_t expected_invalid = 3;
  const std::int32_t expected_end = 1;
  const std::int64_t expected_counter = 77;
  const bool false_value = false;
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
      cudaMemcpy(
          canonical_invalid.get(),
          &expected_invalid,
          sizeof(expected_invalid),
          cudaMemcpyHostToDevice),
      "copy canonical invalid rows");
  require_cuda(
      cudaMemcpy(
          canonical_end.get(),
          &expected_end,
          sizeof(expected_end),
          cudaMemcpyHostToDevice),
      "copy canonical EOS");
  require_cuda(
      cudaMemcpy(
          updated_counter.get(),
          &expected_counter,
          sizeof(expected_counter),
          cudaMemcpyHostToDevice),
      "copy updated RNG counter");
  require_cuda(
      cudaMemcpy(
          finished.get(),
          &false_value,
          sizeof(false_value),
          cudaMemcpyHostToDevice),
      "clear finished");
  require_cuda(
      magpie_tts_rt::launch_finalize_local_step(
          device_step.get(),
          aggregate.get(),
          4,
          canonical_invalid.get(),
          canonical_end.get(),
          step_invalid.get(),
          step_end.get(),
          updated_counter.get(),
          counter.get(),
          finished.get(),
          nullptr),
      "finalize Local AR step");
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
  std::int32_t observed_invalid = 0;
  std::int32_t observed_end = -1;
  std::int64_t observed_counter = 0;
  bool observed_finished = false;
  require_cuda(
      cudaMemcpy(
          &observed_invalid,
          step_invalid.get(),
          sizeof(observed_invalid),
          cudaMemcpyDeviceToHost),
      "copy step invalid rows");
  require_cuda(
      cudaMemcpy(
          &observed_end,
          step_end.get(),
          sizeof(observed_end),
          cudaMemcpyDeviceToHost),
      "copy step EOS");
  require_cuda(
      cudaMemcpy(
          &observed_counter,
          counter.get(),
          sizeof(observed_counter),
          cudaMemcpyDeviceToHost),
      "copy canonical RNG input");
  require_cuda(
      cudaMemcpy(
          &observed_finished,
          finished.get(),
          sizeof(observed_finished),
          cudaMemcpyDeviceToHost),
      "copy finished latch");
  require(
      observed_invalid == expected_invalid,
      "Local AR invalid-row diagnostic was not committed");
  require(
      observed_end == expected_end,
      "Local AR EOS diagnostic was not committed");
  require(
      observed_counter == expected_counter,
      "Local AR RNG counter was not advanced");
  require(observed_finished, "EOS was not latched");
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
}

void test_generation_diagnostics_pack() {
  constexpr std::uint32_t step_count = 2;
  const std::int32_t status = 0;
  const std::array<std::int64_t, 4> attended{3, 5, 77, 88};
  const std::array<std::int32_t, 4> alignment{0, 0, 9, 9};
  const std::array<std::int32_t, 4> invalid_rows{0, 0, 9, 9};
  const std::array<std::int32_t, 4> eos{-1, 1, 9, 9};
  DeviceBuffer<std::int32_t> device_status(1);
  DeviceBuffer<std::int64_t> device_attended(attended.size());
  DeviceBuffer<std::int32_t> device_alignment(alignment.size());
  DeviceBuffer<std::int32_t> device_invalid_rows(invalid_rows.size());
  DeviceBuffer<std::int32_t> device_eos(eos.size());
  DeviceBuffer<magpie_tts_rt::GenerationBatchDiagnostics> device_output(1);
  require_cuda(
      cudaMemcpy(
          device_status.get(),
          &status,
          sizeof(status),
          cudaMemcpyHostToDevice),
      "copy Main Decoder status");
  require_cuda(
      cudaMemcpy(
          device_attended.get(),
          attended.data(),
          sizeof(attended),
          cudaMemcpyHostToDevice),
      "copy attended diagnostics");
  require_cuda(
      cudaMemcpy(
          device_alignment.get(),
          alignment.data(),
          sizeof(alignment),
          cudaMemcpyHostToDevice),
      "copy alignment diagnostics");
  require_cuda(
      cudaMemcpy(
          device_invalid_rows.get(),
          invalid_rows.data(),
          sizeof(invalid_rows),
          cudaMemcpyHostToDevice),
      "copy Local AR diagnostics");
  require_cuda(
      cudaMemcpy(
          device_eos.get(),
          eos.data(),
          sizeof(eos),
          cudaMemcpyHostToDevice),
      "copy EOS diagnostics");

  magpie_tts_rt::GenerationDiagnosticSources sources{};
  sources.main_decoder_execution_status = device_status.get();
  for (std::size_t step = 0; step < attended.size(); ++step) {
    sources.attended_token_indices[step] =
        device_attended.get() + step;
    sources.alignment_invalid_steps[step] =
        device_alignment.get() + step;
    sources.local_invalid_rows[step] =
        device_invalid_rows.get() + step;
    sources.end_frame_indices[step] = device_eos.get() + step;
  }
  require_cuda(
      magpie_tts_rt::launch_pack_generation_diagnostics(
          sources, step_count, device_output.get(), nullptr),
      "pack generation diagnostics");
  magpie_tts_rt::GenerationBatchDiagnostics observed{};
  require_cuda(
      cudaMemcpy(
          &observed,
          device_output.get(),
          sizeof(observed),
          cudaMemcpyDeviceToHost),
      "copy packed generation diagnostics");
  const auto validation =
      magpie_tts_rt::validate_generation_diagnostic_payload(
          observed, step_count);
  require(validation.valid(), "packed diagnostic payload is invalid");
  require(observed.step_count == step_count, "step count changed");
  require(
      observed.main_decoder_execution_status == status,
      "Main Decoder status changed");
  for (std::size_t step = 0; step < step_count; ++step) {
    require(
        observed.steps[step].attended_token_index == attended[step] &&
            observed.steps[step].alignment_invalid == alignment[step] &&
            observed.steps[step].local_invalid_rows ==
                invalid_rows[step] &&
            observed.steps[step].end_frame_index == eos[step],
        "active generation diagnostic changed");
  }
  for (std::size_t step = step_count; step < attended.size(); ++step) {
    require(
        observed.steps[step].attended_token_index == -1 &&
            observed.steps[step].alignment_invalid == 0 &&
            observed.steps[step].local_invalid_rows == 0 &&
            observed.steps[step].end_frame_index == -1 &&
            observed.steps[step].reserved == 0,
        "unused generation diagnostic is not canonical");
  }

  magpie_tts_rt::GenerationDiagnosticSources missing{};
  require(
      magpie_tts_rt::launch_pack_generation_diagnostics(
          missing, step_count, device_output.get(), nullptr) ==
          cudaErrorInvalidValue,
      "missing diagnostic sources were accepted");
}

}  // namespace

int main() {
  try {
    test_cfg();
    test_local_step_finalization();
    test_generation_diagnostics_pack();
  } catch (const std::exception& error) {
    std::cerr << "synthesis kernel GPU test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

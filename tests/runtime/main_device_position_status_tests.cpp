#include "runtime/main_device_position_status.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] constexpr std::int32_t encode(
    const std::uint32_t category,
    const std::uint32_t layer,
    const std::uint32_t operation,
    const std::uint32_t detail) noexcept {
  return static_cast<std::int32_t>(
      (category << MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_SHIFT) |
      (layer << MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_SHIFT) |
      (operation << MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SHIFT) |
      detail);
}

void test_invalid_k_status() {
  const std::int32_t raw = encode(
      MTT_MAIN_DEVICE_POSITION_STATUS_INVALID_K,
      3,
      MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SELECTOR,
      0);
  const magpie_tts_rt::MainDevicePositionStatus decoded =
      magpie_tts_rt::decode_main_device_position_status(raw);
  require(decoded.valid, "canonical invalid-K status was rejected");
  require(decoded.layer_index == 3, "invalid-K layer was lost");
  require(
      magpie_tts_rt::describe_main_device_position_status(raw).find(
          "reason=invalid_k") != std::string::npos,
      "invalid-K diagnostic lost its reason");
}

void test_cuda_update_status() {
  const std::int32_t raw = encode(
      MTT_MAIN_DEVICE_POSITION_STATUS_CUDA_GRAPH_UPDATE,
      11,
      MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_PV,
      400);
  const magpie_tts_rt::MainDevicePositionStatus decoded =
      magpie_tts_rt::decode_main_device_position_status(raw);
  require(decoded.valid, "canonical CUDA-update status was rejected");
  require(decoded.layer_index == 11, "CUDA-update layer was lost");
  require(decoded.detail == 400, "CUDA error number was lost");
  const std::string diagnostic =
      magpie_tts_rt::describe_main_device_position_status(raw);
  require(
      diagnostic.find("operation=pv") != std::string::npos &&
          diagnostic.find("cuda_status=400") != std::string::npos,
      "CUDA-update diagnostic lost operation or error number");
}

void test_malformed_statuses_fail_closed() {
  require(
      !magpie_tts_rt::decode_main_device_position_status(0).valid,
      "success was decoded as an error record");
  require(
      !magpie_tts_rt::decode_main_device_position_status(-1).valid,
      "negative status was accepted");
  require(
      !magpie_tts_rt::decode_main_device_position_status(encode(
           MTT_MAIN_DEVICE_POSITION_STATUS_INVALID_K,
           MTT_MAIN_DEVICE_POSITION_LAYER_COUNT_V1,
           MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SELECTOR,
           0))
           .valid,
      "out-of-range layer was accepted");
  require(
      !magpie_tts_rt::decode_main_device_position_status(encode(
           MTT_MAIN_DEVICE_POSITION_STATUS_CUDA_GRAPH_UPDATE,
           0,
           MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_QK,
           0))
           .valid,
      "CUDA-update status without a CUDA error was accepted");
}

}  // namespace

int main() {
  try {
    test_invalid_k_status();
    test_cuda_update_status();
    test_malformed_statuses_fail_closed();
  } catch (const std::exception& error) {
    std::cerr << "Main device-position status test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

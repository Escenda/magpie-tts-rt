#pragma once

#include <cstdint>
#include <string>

#include "magpie_tts_rt/magpie_tts_rt_plugin.h"

namespace magpie_tts_rt {

enum class MainDevicePositionStatusCategory : std::uint32_t {
  invalid_k = MTT_MAIN_DEVICE_POSITION_STATUS_INVALID_K,
  cuda_graph_update =
      MTT_MAIN_DEVICE_POSITION_STATUS_CUDA_GRAPH_UPDATE,
};

enum class MainDevicePositionStatusOperation : std::uint32_t {
  selector = MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SELECTOR,
  qk = MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_QK,
  pv = MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_PV,
};

struct MainDevicePositionStatus final {
  std::int32_t raw;
  MainDevicePositionStatusCategory category;
  std::uint32_t layer_index;
  MainDevicePositionStatusOperation operation;
  std::uint32_t detail;
  bool valid;
};

[[nodiscard]] inline MainDevicePositionStatus
decode_main_device_position_status(const std::int32_t raw) noexcept {
  const std::uint32_t encoded = static_cast<std::uint32_t>(raw);
  const std::uint32_t category =
      (encoded >> MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_SHIFT) &
      MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_MASK;
  const std::uint32_t layer =
      (encoded >> MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_SHIFT) &
      MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_MASK;
  const std::uint32_t operation =
      (encoded >> MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SHIFT) &
      MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_MASK;
  const std::uint32_t detail =
      encoded & MTT_MAIN_DEVICE_POSITION_STATUS_DETAIL_MASK;
  const bool common_valid =
      raw > 0 && layer < MTT_MAIN_DEVICE_POSITION_LAYER_COUNT_V1;
  const bool invalid_k =
      category == MTT_MAIN_DEVICE_POSITION_STATUS_INVALID_K &&
      operation == MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SELECTOR &&
      detail == 0;
  const bool cuda_update =
      category == MTT_MAIN_DEVICE_POSITION_STATUS_CUDA_GRAPH_UPDATE &&
      (operation == MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_QK ||
       operation == MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_PV) &&
      detail != 0;
  return MainDevicePositionStatus{
      .raw = raw,
      .category =
          static_cast<MainDevicePositionStatusCategory>(category),
      .layer_index = layer,
      .operation =
          static_cast<MainDevicePositionStatusOperation>(operation),
      .detail = detail,
      .valid = common_valid && (invalid_k || cuda_update),
  };
}

[[nodiscard]] inline std::string describe_main_device_position_status(
    const std::int32_t raw) {
  const MainDevicePositionStatus status =
      decode_main_device_position_status(raw);
  if (!status.valid) {
    return "malformed Main Decoder device-position status raw=" +
           std::to_string(raw);
  }
  std::string detail =
      "Main Decoder device-position failure layer=" +
      std::to_string(status.layer_index) + " operation=";
  switch (status.operation) {
    case MainDevicePositionStatusOperation::selector:
      detail += "selector";
      break;
    case MainDevicePositionStatusOperation::qk:
      detail += "qk";
      break;
    case MainDevicePositionStatusOperation::pv:
      detail += "pv";
      break;
  }
  if (status.category == MainDevicePositionStatusCategory::invalid_k) {
    return detail + " reason=invalid_k raw=" + std::to_string(raw);
  }
  return detail + " reason=cuda_graph_update cuda_status=" +
         std::to_string(status.detail) +
         " raw=" + std::to_string(raw);
}

}  // namespace magpie_tts_rt

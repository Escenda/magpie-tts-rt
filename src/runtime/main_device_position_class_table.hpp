#pragma once

#include <stdexcept>
#include <string>

#include "magpie_tts_rt/magpie_tts_rt_plugin.h"

namespace magpie_tts_rt {

enum class MainDevicePositionClassTableErrorCode {
  invalid_header,
  invalid_class_record,
  invalid_k_record,
  unused_class,
  digest_failure,
};

class MainDevicePositionClassTableError final
    : public std::runtime_error {
 public:
  MainDevicePositionClassTableError(
      MainDevicePositionClassTableErrorCode code,
      std::string detail);

  [[nodiscard]] MainDevicePositionClassTableErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  MainDevicePositionClassTableErrorCode code_;
  std::string detail_;
};

struct MainDevicePositionClassTableIdentity {
  std::string canonical_json;
  std::string sha256;
};

// Validates every ABI field, constructs the exact canonical ASCII JSON used
// by tools/export/mode8_class_table_identity.py, and hashes those bytes.
[[nodiscard]] MainDevicePositionClassTableIdentity
collect_main_device_position_class_table_identity(
    const mtt_main_device_position_class_table_v1_t& table);

}  // namespace magpie_tts_rt

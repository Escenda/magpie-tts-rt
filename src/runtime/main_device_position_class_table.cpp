#include "runtime/main_device_position_class_table.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

namespace magpie_tts_rt {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kQkClassCount = 7U;
constexpr std::size_t kPvClassCount = 14U;
constexpr std::int32_t kMinimumActiveK = 219;
constexpr std::int32_t kMaximumActiveK = 467;

[[nodiscard]] std::string error_message(
    const MainDevicePositionClassTableErrorCode code,
    const std::string_view detail) {
  return "mode-8 class-table identity failed [code=" +
         std::to_string(static_cast<int>(code)) + "]: " +
         std::string(detail);
}

[[noreturn]] void fail(
    const MainDevicePositionClassTableErrorCode code,
    const std::string& detail) {
  throw MainDevicePositionClassTableError(code, detail);
}

template <std::size_t Count>
[[nodiscard]] bool all_zero(
    const std::uint64_t (&values)[Count]) noexcept {
  for (const std::uint64_t value : values) {
    if (value != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string decode_function_name(
    const mtt_main_device_position_class_v1_t& record,
    const std::size_t index) {
  const void* const terminator = std::memchr(
      record.function_name,
      '\0',
      sizeof(record.function_name));
  if (terminator == nullptr) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_class_record,
        "classes[" + std::to_string(index) +
            "].function_name is not NUL-terminated");
  }
  const auto* const end = static_cast<const char*>(terminator);
  const std::size_t length = static_cast<std::size_t>(
      end - record.function_name);
  if (length == 0U) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_class_record,
        "classes[" + std::to_string(index) +
            "].function_name is empty");
  }
  for (std::size_t offset = 0U; offset < length; ++offset) {
    const auto character = static_cast<unsigned char>(
        record.function_name[offset]);
    if (character < 0x21U || character > 0x7eU) {
      fail(
          MainDevicePositionClassTableErrorCode::invalid_class_record,
          "classes[" + std::to_string(index) +
              "].function_name contains a non-canonical byte");
    }
  }
  for (std::size_t offset = length + 1U;
       offset < sizeof(record.function_name);
       ++offset) {
    if (record.function_name[offset] != '\0') {
      fail(
          MainDevicePositionClassTableErrorCode::invalid_class_record,
          "classes[" + std::to_string(index) +
              "].function_name has nonzero trailing bytes");
    }
  }
  return std::string(record.function_name, length);
}

[[nodiscard]] std::string sha256_ascii(
    const std::string_view payload) {
  using DigestContext =
      std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  DigestContext context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), payload.data(), payload.size()) != 1) {
    fail(
        MainDevicePositionClassTableErrorCode::digest_failure,
        "unable to initialize or update SHA-256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0U;
  if (EVP_DigestFinal_ex(
          context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != 32U) {
    fail(
        MainDevicePositionClassTableErrorCode::digest_failure,
        "unable to finalize SHA-256");
  }
  constexpr std::string_view hexadecimal = "0123456789abcdef";
  std::string result(digest_size * 2U, '0');
  for (std::size_t index = 0U; index < digest_size; ++index) {
    result.at(index * 2U) = hexadecimal.at(digest.at(index) >> 4U);
    result.at(index * 2U + 1U) =
        hexadecimal.at(digest.at(index) & 0x0fU);
  }
  return result;
}

[[nodiscard]] Json class_document(
    const mtt_main_device_position_class_v1_t& record,
    const std::size_t index) {
  if (record.struct_size !=
          sizeof(mtt_main_device_position_class_v1_t) ||
      record.abi_version != MTT_PLUGIN_ABI_VERSION_1 ||
      record.reserved_0 != 0U || !all_zero(record.reserved)) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_class_record,
        "classes[" + std::to_string(index) +
            "] has an invalid ABI header or reserved field");
  }
  const bool is_qk = index < kQkClassCount;
  const std::uint32_t expected_operation =
      is_qk ? MTT_MAIN_DEVICE_POSITION_OPERATION_QK
            : MTT_MAIN_DEVICE_POSITION_OPERATION_PV;
  const std::uint32_t expected_class_index = static_cast<std::uint32_t>(
      is_qk ? index : index - kQkClassCount);
  if (record.operation != expected_operation ||
      record.class_index != expected_class_index) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_class_record,
        "classes[" + std::to_string(index) +
            "] operation/class ordering mismatch");
  }
  std::string transport;
  if (record.parameter_transport ==
      MTT_MAIN_DEVICE_POSITION_PARAMETER_KERNEL_PARAMS) {
    transport = "kernel_params";
  } else if (record.parameter_transport ==
             MTT_MAIN_DEVICE_POSITION_PARAMETER_EXTRA) {
    transport = "extra";
  } else {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_class_record,
        "classes[" + std::to_string(index) +
            "].parameter_transport is invalid");
  }
  if (record.block_x == 0U || record.block_y == 0U ||
      record.block_z == 0U || record.parameter_size == 0U) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_class_record,
        "classes[" + std::to_string(index) +
            "] has an empty launch contract");
  }
  return Json{
      {"operation", is_qk ? "qk" : "pv"},
      {"class_index", record.class_index},
      {"function_name", decode_function_name(record, index)},
      {"parameter_transport", std::move(transport)},
      {"block", Json::array({record.block_x, record.block_y, record.block_z})},
      {"shared_memory_bytes", record.shared_memory_bytes},
      {"parameter_offset", record.parameter_offset},
      {"parameter_size", record.parameter_size},
  };
}

[[nodiscard]] Json k_document(
    const mtt_main_device_position_k_v1_t& record,
    const std::size_t index,
    std::array<bool, kQkClassCount>& used_qk,
    std::array<bool, kPvClassCount>& used_pv) {
  const std::int32_t expected_active_k =
      kMinimumActiveK + static_cast<std::int32_t>(index);
  if (record.struct_size != sizeof(mtt_main_device_position_k_v1_t) ||
      record.abi_version != MTT_PLUGIN_ABI_VERSION_1 ||
      record.active_k != expected_active_k || record.reserved_0 != 0U ||
      !all_zero(record.reserved)) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_k_record,
        "k_records[" + std::to_string(index) +
            "] has an invalid ABI header, active K, or reserved field");
  }
  if (record.qk_class_index < 0 ||
      record.qk_class_index >= static_cast<std::int32_t>(kQkClassCount) ||
      record.pv_class_index < 0 ||
      record.pv_class_index >= static_cast<std::int32_t>(kPvClassCount)) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_k_record,
        "k_records[" + std::to_string(index) +
            "] contains an invalid class index");
  }
  if (record.qk_grid_x == 0U || record.qk_grid_y == 0U ||
      record.qk_grid_z == 0U || record.pv_grid_x == 0U ||
      record.pv_grid_y == 0U || record.pv_grid_z == 0U) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_k_record,
        "k_records[" + std::to_string(index) +
            "] contains an empty grid dimension");
  }
  used_qk.at(static_cast<std::size_t>(record.qk_class_index)) = true;
  used_pv.at(static_cast<std::size_t>(record.pv_class_index)) = true;
  return Json{
      {"active_k", record.active_k},
      {"qk",
       Json{
           {"class_index", record.qk_class_index},
           {"grid",
            Json::array(
                {record.qk_grid_x,
                 record.qk_grid_y,
                 record.qk_grid_z})},
       }},
      {"pv",
       Json{
           {"class_index", record.pv_class_index},
           {"grid",
            Json::array(
                {record.pv_grid_x,
                 record.pv_grid_y,
                 record.pv_grid_z})},
       }},
  };
}

}  // namespace

MainDevicePositionClassTableError::MainDevicePositionClassTableError(
    const MainDevicePositionClassTableErrorCode code,
    std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

MainDevicePositionClassTableErrorCode
MainDevicePositionClassTableError::code() const noexcept {
  return code_;
}

const std::string& MainDevicePositionClassTableError::detail() const noexcept {
  return detail_;
}

MainDevicePositionClassTableIdentity
collect_main_device_position_class_table_identity(
    const mtt_main_device_position_class_table_v1_t& table) {
  if (table.struct_size !=
          sizeof(mtt_main_device_position_class_table_v1_t) ||
      table.abi_version != MTT_PLUGIN_ABI_VERSION_1 ||
      table.class_count != MTT_MAIN_DEVICE_POSITION_CLASS_COUNT_V1 ||
      table.k_count != MTT_MAIN_DEVICE_POSITION_K_COUNT_V1 ||
      !all_zero(table.reserved)) {
    fail(
        MainDevicePositionClassTableErrorCode::invalid_header,
        "class-table ABI header, dimensions, or reserved fields are invalid");
  }

  Json classes = Json::array();
  for (std::size_t index = 0U;
       index < MTT_MAIN_DEVICE_POSITION_CLASS_COUNT_V1;
       ++index) {
    classes.push_back(class_document(table.classes[index], index));
  }
  Json k_records = Json::array();
  std::array<bool, kQkClassCount> used_qk{};
  std::array<bool, kPvClassCount> used_pv{};
  for (std::size_t index = 0U;
       index < MTT_MAIN_DEVICE_POSITION_K_COUNT_V1;
       ++index) {
    k_records.push_back(k_document(
        table.k_records[index], index, used_qk, used_pv));
  }
  if (!std::ranges::all_of(used_qk, [](const bool used) { return used; }) ||
      !std::ranges::all_of(used_pv, [](const bool used) { return used; })) {
    fail(
        MainDevicePositionClassTableErrorCode::unused_class,
        "class table contains an unused QK or PV class");
  }

  const Json document{
      {"schema_version", 1},
      {"active_k_range", Json::array({kMinimumActiveK, kMaximumActiveK})},
      {"qk_class_count", kQkClassCount},
      {"pv_class_count", kPvClassCount},
      {"classes", std::move(classes)},
      {"k_records", std::move(k_records)},
  };
  std::string canonical_json = document.dump(
      -1, ' ', true, Json::error_handler_t::strict);
  const std::string digest = sha256_ascii(canonical_json);
  return MainDevicePositionClassTableIdentity{
      .canonical_json = std::move(canonical_json),
      .sha256 = digest,
  };
}

}  // namespace magpie_tts_rt

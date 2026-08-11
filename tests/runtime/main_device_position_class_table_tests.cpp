#include "runtime/main_device_position_class_table.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] mtt_main_device_position_class_table_v1_t synthetic_table() {
  mtt_main_device_position_class_table_v1_t table{};
  table.struct_size = sizeof(table);
  table.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  table.class_count = MTT_MAIN_DEVICE_POSITION_CLASS_COUNT_V1;
  table.k_count = MTT_MAIN_DEVICE_POSITION_K_COUNT_V1;
  for (std::size_t index = 0U;
       index < MTT_MAIN_DEVICE_POSITION_CLASS_COUNT_V1;
       ++index) {
    auto& record = table.classes[index];
    const bool is_qk = index < 7U;
    const std::size_t class_index = is_qk ? index : index - 7U;
    record.struct_size = sizeof(record);
    record.abi_version = MTT_PLUGIN_ABI_VERSION_1;
    record.operation =
        is_qk ? MTT_MAIN_DEVICE_POSITION_OPERATION_QK
              : MTT_MAIN_DEVICE_POSITION_OPERATION_PV;
    record.class_index = static_cast<std::uint32_t>(class_index);
    record.parameter_transport =
        index % 2U == 0U
            ? MTT_MAIN_DEVICE_POSITION_PARAMETER_KERNEL_PARAMS
            : MTT_MAIN_DEVICE_POSITION_PARAMETER_EXTRA;
    record.block_x = static_cast<std::uint32_t>(32U * (1U + index % 4U));
    record.block_y = static_cast<std::uint32_t>(1U + index % 2U);
    record.block_z = 1U;
    record.shared_memory_bytes = static_cast<std::uint32_t>(index * 16U);
    record.parameter_offset = static_cast<std::uint64_t>(index * 8U);
    record.parameter_size = static_cast<std::uint64_t>(
        8U + (index % 3U) * 8U);
    const int written = std::snprintf(
        record.function_name,
        sizeof(record.function_name),
        "kernel_%s_%zu",
        is_qk ? "qk" : "pv",
        class_index);
    require(
        written > 0 &&
            static_cast<std::size_t>(written) <
                sizeof(record.function_name),
        "synthetic function name overflow");
  }
  for (std::size_t index = 0U;
       index < MTT_MAIN_DEVICE_POSITION_K_COUNT_V1;
       ++index) {
    auto& record = table.k_records[index];
    record.struct_size = sizeof(record);
    record.abi_version = MTT_PLUGIN_ABI_VERSION_1;
    record.active_k = 219 + static_cast<std::int32_t>(index);
    record.qk_class_index = static_cast<std::int32_t>(index % 7U);
    record.qk_grid_x = static_cast<std::uint32_t>(1U + index % 5U);
    record.qk_grid_y = 1U;
    record.qk_grid_z = 1U;
    record.pv_class_index = static_cast<std::int32_t>(index % 14U);
    record.pv_grid_x = static_cast<std::uint32_t>(1U + index % 3U);
    record.pv_grid_y = static_cast<std::uint32_t>(1U + index % 2U);
    record.pv_grid_z = 1U;
  }
  return table;
}

void test_cross_language_fixture() {
  const auto identity =
      magpie_tts_rt::collect_main_device_position_class_table_identity(
          synthetic_table());
  require(
      identity.canonical_json.size() == 27'210U,
      "canonical fixture byte length mismatch");
  require(
      !identity.canonical_json.ends_with('\n'),
      "canonical fixture has a trailing newline");
  require(
      identity.sha256 ==
          "9dc9516d753da031147f50ff34296b2bf"
          "5f4498d4c789d18dc2fc19120190dbe",
      "cross-language fixture digest mismatch");
}

void test_mutation_changes_digest() {
  auto table = synthetic_table();
  const std::string baseline =
      magpie_tts_rt::collect_main_device_position_class_table_identity(table)
          .sha256;
  table.classes[0].block_x = 64U;
  const std::string mutated =
      magpie_tts_rt::collect_main_device_position_class_table_identity(table)
          .sha256;
  require(mutated != baseline, "class-table mutation retained its digest");
}

void test_noncanonical_record_fails_closed() {
  auto table = synthetic_table();
  table.k_records[0].reserved[0] = 1U;
  bool rejected = false;
  try {
    static_cast<void>(
        magpie_tts_rt::collect_main_device_position_class_table_identity(
            table));
  } catch (const magpie_tts_rt::MainDevicePositionClassTableError&) {
    rejected = true;
  }
  require(rejected, "nonzero reserved field was accepted");
}

}  // namespace

int main() {
  try {
    test_cross_language_fixture();
    test_mutation_changes_digest();
    test_noncanonical_record_fails_closed();
  } catch (const std::exception& error) {
    std::cerr << "Main device-position class-table test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

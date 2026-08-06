#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

#include "magpie_tts_rt/magpie_tts_rt.h"

namespace {

void require(const bool condition) {
  if (!condition) {
    std::abort();
  }
}

[[nodiscard]] mtt_error_v1_t new_error() {
  mtt_error_v1_t error{};
  error.struct_size = sizeof(error);
  error.abi_version = MTT_ABI_VERSION_1;
  return error;
}

[[nodiscard]] mtt_api_v1_t load_api() {
  mtt_api_v1_t api{};
  api.struct_size = sizeof(api);
  api.abi_version = MTT_ABI_VERSION_1;
  require(mtt_get_api(MTT_ABI_VERSION_1, &api) == MTT_STATUS_OK);
  return api;
}

[[nodiscard]] mtt_model_desc_v1_t new_model_desc(
    const std::string_view path) {
  mtt_model_desc_v1_t desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = MTT_ABI_VERSION_1;
  desc.bundle_path = path.data();
  desc.bundle_path_length = static_cast<std::uint64_t>(path.size());
  desc.expected_manifest_sha256[0] = 1;
  return desc;
}

void test_abi_negotiation() {
  require(mtt_get_api(MTT_ABI_VERSION_1, nullptr) == MTT_STATUS_INVALID_ARGUMENT);

  mtt_api_v1_t api{};
  api.struct_size = sizeof(api);
  api.abi_version = MTT_ABI_VERSION_1;
  require(mtt_get_api(2, &api) == MTT_STATUS_ABI_MISMATCH);

  api.struct_size -= 1;
  require(mtt_get_api(MTT_ABI_VERSION_1, &api) == MTT_STATUS_ABI_MISMATCH);
}

void test_runtime_validation_and_lifecycle() {
  const auto api = load_api();
  auto error = new_error();

  mtt_runtime_desc_v1_t desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = MTT_ABI_VERSION_1;
  desc.cuda_device_index = 0;

  mtt_runtime_t* runtime = nullptr;
  require(api.runtime_create(nullptr, &runtime, &error) == MTT_STATUS_INVALID_ARGUMENT);
  require(runtime == nullptr);
  require(error.code == MTT_STATUS_INVALID_ARGUMENT);

  desc.flags = 1;
  error = new_error();
  require(api.runtime_create(&desc, &runtime, &error) == MTT_STATUS_INVALID_ARGUMENT);
  require(runtime == nullptr);

  desc.flags = 0;
  error = new_error();
  require(api.runtime_create(&desc, &runtime, &error) == MTT_STATUS_OK);
  require(runtime != nullptr);
  require(error.code == MTT_STATUS_OK);

  mtt_model_t* model = nullptr;
  constexpr std::string_view path{"bundle"};
  auto model_desc = new_model_desc(path);
  error = new_error();
  require(
      api.model_load(runtime, &model_desc, &model, &error) ==
      MTT_STATUS_IO_ERROR);
  require(model == nullptr);
  require(error.code == MTT_STATUS_IO_ERROR);
  require(std::strlen(error.message) > 0);

  const std::array<char, 3> embedded_nul{'a', '\0', 'b'};
  model_desc.bundle_path = embedded_nul.data();
  model_desc.bundle_path_length =
      static_cast<std::uint64_t>(embedded_nul.size());
  error = new_error();
  require(
      api.model_load(runtime, &model_desc, &model, &error) ==
      MTT_STATUS_INVALID_ARGUMENT);
  require(model == nullptr);

  const std::array<char, 2> invalid_utf8{
      static_cast<char>(0xC0), static_cast<char>(0x80)};
  model_desc.bundle_path = invalid_utf8.data();
  model_desc.bundle_path_length =
      static_cast<std::uint64_t>(invalid_utf8.size());
  error = new_error();
  require(
      api.model_load(runtime, &model_desc, &model, &error) ==
      MTT_STATUS_INVALID_ARGUMENT);
  require(model == nullptr);

  model_desc = new_model_desc(path);
  model_desc.expected_manifest_sha256[0] = 0;
  error = new_error();
  require(
      api.model_load(runtime, &model_desc, &model, &error) ==
      MTT_STATUS_INVALID_ARGUMENT);
  require(model == nullptr);

  model_desc = new_model_desc(path);
  model_desc.struct_size = sizeof(std::uint32_t) * 2;
  model_desc.bundle_path = nullptr;
  error = new_error();
  require(
      api.model_load(runtime, &model_desc, &model, &error) ==
      MTT_STATUS_ABI_MISMATCH);
  require(model == nullptr);

  error = new_error();
  require(api.runtime_destroy(runtime, &error) == MTT_STATUS_OK);
  require(error.code == MTT_STATUS_OK);
}

void test_error_struct_is_versioned() {
  const auto api = load_api();
  mtt_runtime_desc_v1_t desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = MTT_ABI_VERSION_1;
  desc.cuda_device_index = 0;

  mtt_error_v1_t invalid_error{};
  invalid_error.struct_size = sizeof(invalid_error) - 1;
  invalid_error.abi_version = MTT_ABI_VERSION_1;
  mtt_runtime_t* runtime = nullptr;
  require(api.runtime_create(&desc, &runtime, &invalid_error) == MTT_STATUS_ABI_MISMATCH);
  require(runtime == nullptr);
}

void test_request_descriptor_header_precedes_payload_access() {
  const auto api = load_api();
  auto error = new_error();

  mtt_request_desc_v1_t desc{};
  desc.struct_size = sizeof(std::uint32_t) * 2;
  desc.abi_version = MTT_ABI_VERSION_1;
  desc.text_token_ids = nullptr;
  desc.text_token_count = 0;

  std::uint8_t opaque_session_storage = 0;
  auto* session =
      reinterpret_cast<mtt_session_t*>(&opaque_session_storage);
  mtt_request_t* request = nullptr;
  require(
      api.request_start(session, &desc, &request, &error) ==
      MTT_STATUS_ABI_MISMATCH);
  require(request == nullptr);
  require(error.code == MTT_STATUS_ABI_MISMATCH);
  require(error.stage == MTT_ERROR_STAGE_ABI);

  const std::int64_t token = 1;
  desc.struct_size = sizeof(desc);
  desc.text_token_ids = &token;
  desc.text_token_count = MTT_MAX_TEXT_TOKENS + 1;
  error = new_error();
  require(
      api.request_start(session, &desc, &request, &error) ==
      MTT_STATUS_INVALID_ARGUMENT);
  require(request == nullptr);
  require(error.code == MTT_STATUS_INVALID_ARGUMENT);

  desc.text_token_count = 1;
  const std::int64_t negative_token = -1;
  desc.text_token_ids = &negative_token;
  error = new_error();
  require(
      api.request_start(session, &desc, &request, &error) ==
      MTT_STATUS_INVALID_ARGUMENT);
  require(request == nullptr);
  require(error.code == MTT_STATUS_INVALID_ARGUMENT);

  const std::int64_t wider_than_int32 =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  desc.text_token_ids = &wider_than_int32;
  error = new_error();
  require(
      api.request_start(session, &desc, &request, &error) ==
      MTT_STATUS_INVALID_ARGUMENT);
  require(request == nullptr);
  require(error.code == MTT_STATUS_INVALID_ARGUMENT);

  desc.text_token_ids = &token;
  desc.random_seed =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  error = new_error();
  require(
      api.request_start(session, &desc, &request, &error) ==
      MTT_STATUS_INVALID_ARGUMENT);
  require(request == nullptr);
  require(error.code == MTT_STATUS_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  test_abi_negotiation();
  test_runtime_validation_and_lifecycle();
  test_error_struct_is_versioned();
  test_request_descriptor_header_precedes_payload_access();
  return 0;
}

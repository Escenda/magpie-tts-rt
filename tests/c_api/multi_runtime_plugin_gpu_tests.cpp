#include "magpie_tts_rt/magpie_tts_rt.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] std::uint8_t hex_nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  throw std::invalid_argument(
      "manifest digest must be lowercase hexadecimal");
}

[[nodiscard]] std::array<std::uint8_t, MTT_SHA256_BYTES>
parse_sha256(const std::string_view value) {
  if (value.size() != MTT_SHA256_BYTES * 2U) {
    throw std::invalid_argument(
        "manifest digest must contain 64 hexadecimal digits");
  }
  std::array<std::uint8_t, MTT_SHA256_BYTES> digest{};
  for (std::size_t index = 0; index < digest.size(); ++index) {
    digest[index] = static_cast<std::uint8_t>(
        (hex_nibble(value[index * 2U]) << 4U) |
        hex_nibble(value[index * 2U + 1U]));
  }
  return digest;
}

[[nodiscard]] mtt_error_v1_t error_buffer() {
  mtt_error_v1_t error{};
  error.struct_size = sizeof(error);
  error.abi_version = MTT_ABI_VERSION_1;
  return error;
}

[[nodiscard]] mtt_runtime_t* create_runtime(
    const mtt_api_v1_t& api,
    const std::int32_t cuda_device) {
  mtt_runtime_desc_v1_t desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = MTT_ABI_VERSION_1;
  desc.cuda_device_index = cuda_device;
  mtt_runtime_t* runtime = nullptr;
  auto error = error_buffer();
  const mtt_status_t status =
      api.runtime_create(&desc, &runtime, &error);
  require(
      status == MTT_STATUS_OK && runtime != nullptr,
      "runtime_create failed: " + std::string(error.message));
  return runtime;
}

[[nodiscard]] mtt_model_t* load_model(
    const mtt_api_v1_t& api,
    mtt_runtime_t* runtime,
    const std::string& bundle_path,
    const std::array<std::uint8_t, MTT_SHA256_BYTES>& digest) {
  mtt_model_desc_v1_t desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = MTT_ABI_VERSION_1;
  desc.bundle_path = bundle_path.data();
  desc.bundle_path_length =
      static_cast<std::uint64_t>(bundle_path.size());
  std::memcpy(
      desc.expected_manifest_sha256,
      digest.data(),
      digest.size());
  mtt_model_t* model = nullptr;
  auto error = error_buffer();
  const mtt_status_t status =
      api.model_load(runtime, &desc, &model, &error);
  require(
      status == MTT_STATUS_OK && model != nullptr,
      "model_load failed: " + std::string(error.message));
  return model;
}

void destroy_model(
    const mtt_api_v1_t& api,
    mtt_model_t* model) {
  auto error = error_buffer();
  require(
      api.model_destroy(model, &error) == MTT_STATUS_OK,
      "model_destroy failed: " + std::string(error.message));
}

void destroy_runtime(
    const mtt_api_v1_t& api,
    mtt_runtime_t* runtime) {
  auto error = error_buffer();
  require(
      api.runtime_destroy(runtime, &error) == MTT_STATUS_OK,
      "runtime_destroy failed: " + std::string(error.message));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: mtt_multi_runtime_plugin_gpu_tests "
          "/absolute/runtime-bundle <manifest-sha256> <cuda-device>");
    }
    const std::string bundle_path(argv[1]);
    const auto digest = parse_sha256(argv[2]);
    const std::int32_t cuda_device = std::stoi(argv[3]);

    mtt_api_v1_t api{};
    api.struct_size = sizeof(api);
    api.abi_version = MTT_ABI_VERSION_1;
    require(
        mtt_get_api(MTT_ABI_VERSION_1, &api) ==
            MTT_STATUS_OK,
        "mtt_get_api failed");

    mtt_runtime_t* first_runtime =
        create_runtime(api, cuda_device);
    mtt_runtime_t* second_runtime =
        create_runtime(api, cuda_device);
    mtt_model_t* first_model = nullptr;
    mtt_model_t* second_model = nullptr;
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::thread first_loader([&]() {
      try {
        first_model = load_model(
            api, first_runtime, bundle_path, digest);
      } catch (...) {
        first_error = std::current_exception();
      }
    });
    std::thread second_loader([&]() {
      try {
        second_model = load_model(
            api, second_runtime, bundle_path, digest);
      } catch (...) {
        second_error = std::current_exception();
      }
    });
    first_loader.join();
    second_loader.join();
    if (first_error != nullptr) {
      std::rethrow_exception(first_error);
    }
    if (second_error != nullptr) {
      std::rethrow_exception(second_error);
    }
    destroy_model(api, first_model);
    destroy_model(api, second_model);
    destroy_runtime(api, first_runtime);
    destroy_runtime(api, second_runtime);

    mtt_runtime_t* sequential_runtime =
        create_runtime(api, cuda_device);
    mtt_model_t* sequential_model = load_model(
        api, sequential_runtime, bundle_path, digest);
    destroy_model(api, sequential_model);
    destroy_runtime(api, sequential_runtime);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "multi-runtime plugin test failed: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

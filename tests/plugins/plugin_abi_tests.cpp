#include "magpie_tts_rt/magpie_tts_rt_plugin.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using GetApiFunction = mtt_plugin_status_t (*)(mtt_plugin_api_v1_t*);
using GetClassTableFunction = mtt_plugin_status_t (*)(
    mtt_main_device_position_class_table_v1_t*);

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] GetApiFunction load_get_api(void* handle) {
  static_cast<void>(dlerror());
  void* symbol = dlsym(handle, "mtt_plugin_get_api_v1");
  const char* error = dlerror();
  require(error == nullptr, error == nullptr ? "" : error);
  require(symbol != nullptr, "plugin API symbol is null");
  GetApiFunction function = nullptr;
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

[[nodiscard]] GetClassTableFunction load_get_class_table(void* handle) {
  static_cast<void>(dlerror());
  void* symbol = dlsym(
      handle, "mtt_plugin_get_main_device_position_class_table_v1");
  const char* error = dlerror();
  require(error == nullptr, error == nullptr ? "" : error);
  require(symbol != nullptr, "plugin class-table symbol is null");
  GetClassTableFunction function = nullptr;
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

void test_dlopen_and_api() {
  void* handle = dlopen(MTT_PLUGIN_TEST_LIBRARY, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    throw std::runtime_error(
        error == nullptr ? "dlopen failed without a diagnostic" : error);
  }

  static_cast<void>(dlerror());
  require(
      dlsym(handle, "launch_local_ar_sampling") == nullptr,
      "internal C++ launch helper must not be exported");
  require(dlerror() != nullptr, "hidden symbol lookup must report an error");

  GetApiFunction get_api = load_get_api(handle);
  require(
      get_api(nullptr) == MTT_PLUGIN_STATUS_INVALID_ARGUMENT,
      "null API pointer must be rejected");

  mtt_plugin_api_v1_t wrong_version{};
  wrong_version.struct_size = sizeof(wrong_version);
  wrong_version.abi_version = MTT_PLUGIN_ABI_VERSION_1 + 1U;
  require(
      get_api(&wrong_version) == MTT_PLUGIN_STATUS_ABI_MISMATCH,
      "wrong ABI version must be rejected");

  mtt_plugin_api_v1_t wrong_size{};
  wrong_size.struct_size = sizeof(wrong_size) - 1U;
  wrong_size.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  require(
      get_api(&wrong_size) == MTT_PLUGIN_STATUS_ABI_MISMATCH,
      "wrong API size must be rejected");

  mtt_plugin_api_v1_t api{};
  api.struct_size = sizeof(api);
  api.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  require(
      get_api(&api) == MTT_PLUGIN_STATUS_OK,
      "valid API descriptor must load");
  require(
      api.creator_count == MTT_PLUGIN_CREATOR_COUNT_V1,
      "unexpected creator count");
  require(api.register_plugins != nullptr, "register function is null");

  const mtt_plugin_creator_v1_t& sampling = api.creators[0];
  require(
      sampling.struct_size == sizeof(mtt_plugin_creator_v1_t),
      "sampling creator size");
  require(
      sampling.abi_version == MTT_PLUGIN_ABI_VERSION_1,
      "sampling creator ABI");
  require(
      std::string(sampling.name) == "MagpieLocalARSampling",
      "sampling creator name");
  require(std::string(sampling.version) == "1", "sampling creator version");
  require(
      std::string(sampling.plugin_namespace) == "magpie_tts_rt",
      "sampling creator namespace");

  const mtt_plugin_creator_v1_t& eos = api.creators[1];
  require(
      std::string(eos.name) == "MagpieLocalAREos",
      "EOS creator name");
  require(std::string(eos.version) == "1", "EOS creator version");
  require(
      std::string(eos.plugin_namespace) == "magpie_tts_rt",
      "EOS creator namespace");

  const mtt_plugin_creator_v1_t& layer_norm = api.creators[2];
  require(
      std::string(layer_norm.name) == "MagpieLayerNorm",
      "LayerNorm creator name");
  require(
      std::string(layer_norm.version) == "1",
      "LayerNorm creator version");
  require(
      std::string(layer_norm.plugin_namespace) == "magpie_tts_rt",
      "LayerNorm creator namespace");

  const mtt_plugin_creator_v1_t& gelu_tanh = api.creators[3];
  require(
      std::string(gelu_tanh.name) == "MagpieGeluTanh",
      "GELU creator name");
  require(
      std::string(gelu_tanh.version) == "1",
      "GELU creator version");
  require(
      std::string(gelu_tanh.plugin_namespace) == "magpie_tts_rt",
      "GELU creator namespace");

  const mtt_plugin_creator_v1_t& softmax = api.creators[4];
  require(
      std::string(softmax.name) == "MagpieSoftmax",
      "Softmax creator name");
  require(
      std::string(softmax.version) == "1",
      "Softmax creator version");
  require(
      std::string(softmax.plugin_namespace) == "magpie_tts_rt",
      "Softmax creator namespace");

  require(
      api.register_plugins() == MTT_PLUGIN_STATUS_OK,
      "first explicit registration must succeed");
  require(
      api.register_plugins() == MTT_PLUGIN_STATUS_ALREADY_REGISTERED,
      "second explicit registration must report already registered");

  GetClassTableFunction get_class_table = load_get_class_table(handle);
  require(
      get_class_table(nullptr) == MTT_PLUGIN_STATUS_INVALID_ARGUMENT,
      "null class-table pointer must be rejected");
  mtt_main_device_position_class_table_v1_t wrong_table{};
  wrong_table.struct_size = sizeof(wrong_table) - 1U;
  wrong_table.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  require(
      get_class_table(&wrong_table) == MTT_PLUGIN_STATUS_ABI_MISMATCH,
      "wrong class-table size must be rejected");
  mtt_main_device_position_class_table_v1_t table{};
  table.struct_size = sizeof(table);
  table.abi_version = MTT_PLUGIN_ABI_VERSION_1;
  require(
      get_class_table(&table) == MTT_PLUGIN_STATUS_NOT_READY,
      "class table must be unavailable before mode-8 discovery");
  require(dlclose(handle) == 0, "dlclose failed");
}

}  // namespace

int main() {
  try {
    test_dlopen_and_api();
  } catch (const std::exception& error) {
    std::cerr << "plugin ABI test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

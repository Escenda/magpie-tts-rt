#include "magpie_tts_rt/magpie_tts_rt.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "bundle/bundle.hpp"
#include "manifest/manifest.hpp"
#include "runtime/fingerprint.hpp"
#include "runtime/model_loader.hpp"
#include "runtime/request_failure_policy.hpp"
#include "runtime/request_state.hpp"
#include "runtime/session_resources.hpp"
#include "runtime/startup_gate.hpp"
#include "runtime/startup_golden.hpp"
#include "runtime/synthesis_pipeline.hpp"

namespace {

static_assert(
    NV_TENSORRT_MAJOR == 10,
    "MagpieTTS-RT ABI v1 requires TensorRT major version 10");
static_assert(
    CUDART_VERSION / 1000 == 13,
    "MagpieTTS-RT ABI v1 requires CUDA Runtime major version 13");

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity > Severity::kWARNING || message == nullptr) {
      return;
    }
    std::fprintf(stderr, "MagpieTTS-RT TensorRT: %s\n", message);
  }
};

[[nodiscard]] magpie_tts_rt::LeaseIdSequence&
process_lease_id_sequence() {
  static magpie_tts_rt::LeaseIdSequence sequence;
  return sequence;
}

}  // namespace

struct mtt_runtime {
  explicit mtt_runtime(const int32_t selected_device) : cuda_device_index(selected_device) {}

  TensorRtLogger logger;
  nvinfer1::IRuntime* tensorrt_runtime{nullptr};
  magpie_tts_rt::RuntimePluginState plugin;
  std::atomic<uint32_t> live_models{0};
  int32_t cuda_device_index;
};

struct mtt_model {
  mtt_model(
      mtt_runtime* owning_runtime,
      magpie_tts_rt::RuntimeBundleManifest loaded_manifest,
      magpie_tts_rt::StartupGoldenFixture loaded_golden_fixture,
      std::vector<magpie_tts_rt::LoadedEngine> loaded_engines)
      : runtime(owning_runtime),
        manifest(std::move(loaded_manifest)),
        golden_fixture(std::move(loaded_golden_fixture)),
        engines(std::move(loaded_engines)) {}

  mtt_runtime* runtime;
  magpie_tts_rt::RuntimeBundleManifest manifest;
  magpie_tts_rt::StartupGoldenFixture golden_fixture;
  std::vector<magpie_tts_rt::LoadedEngine> engines;
  std::atomic<uint32_t> live_sessions{0};
};

struct mtt_session {
  mtt_session(
      mtt_model* owning_model,
      std::unique_ptr<magpie_tts_rt::SessionResources> session_resources)
      : model(owning_model), resources(std::move(session_resources)) {}

  mtt_model* model;
  std::unique_ptr<magpie_tts_rt::SessionResources> resources;
  std::atomic<uint32_t> live_requests{0};
  std::atomic<bool> poisoned{false};
};

struct mtt_request {
  mtt_request(
      mtt_session* owning_session,
      std::vector<std::int32_t> prepared_tokens)
      : session(owning_session),
        text_token_ids(std::move(prepared_tokens)),
        state(
            text_token_ids.size(),
            owning_session->model->manifest.limits
                .pcm_ring_capacity_frames,
            process_lease_id_sequence()) {}

  mtt_session* session;
  std::vector<std::int32_t> text_token_ids;
  magpie_tts_rt::StreamingRequestState state;
  std::thread worker;
  std::map<
      std::uint64_t,
      std::vector<mtt_alignment_event_v1_t>>
      lease_alignment_events;
};

namespace {

[[nodiscard]] bool has_valid_header(
    const uint32_t struct_size,
    const uint32_t abi_version,
    const uint32_t expected_size) noexcept {
  return struct_size == expected_size && abi_version == MTT_ABI_VERSION_1;
}

[[nodiscard]] bool error_is_usable(const mtt_error_v1_t* error) noexcept {
  return error == nullptr ||
         has_valid_header(error->struct_size, error->abi_version, sizeof(mtt_error_v1_t));
}

void write_error(
    mtt_error_v1_t* error,
    const mtt_status_t code,
    const mtt_error_stage_t stage,
    const std::string_view message) noexcept {
  if (error == nullptr ||
      !has_valid_header(error->struct_size, error->abi_version, sizeof(mtt_error_v1_t))) {
    return;
  }
  error->code = code;
  error->stage = stage;
  const auto copy_size =
      message.size() < MTT_ERROR_MESSAGE_CAPACITY - 1 ? message.size() : MTT_ERROR_MESSAGE_CAPACITY - 1;
  std::memcpy(error->message, message.data(), copy_size);
  error->message[copy_size] = '\0';
}

void clear_error(mtt_error_v1_t* error) noexcept {
  write_error(error, MTT_STATUS_OK, MTT_ERROR_STAGE_NONE, "");
}

[[nodiscard]] bool reserved_is_zero(const uint64_t (&values)[4]) noexcept {
  for (const uint64_t value : values) {
    if (value != 0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool is_valid_utf8(const std::string_view text) noexcept {
  std::size_t index = 0;
  while (index < text.size()) {
    const auto lead = static_cast<unsigned char>(text[index]);
    std::size_t continuation_count = 0;
    std::uint32_t code_point = 0;
    if (lead <= 0x7FU) {
      ++index;
      continue;
    }
    if ((lead & 0xE0U) == 0xC0U) {
      continuation_count = 1;
      code_point = lead & 0x1FU;
      if (code_point == 0) {
        return false;
      }
    } else if ((lead & 0xF0U) == 0xE0U) {
      continuation_count = 2;
      code_point = lead & 0x0FU;
    } else if ((lead & 0xF8U) == 0xF0U) {
      continuation_count = 3;
      code_point = lead & 0x07U;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index - 1) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const auto continuation =
          static_cast<unsigned char>(text[index + offset]);
      if ((continuation & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (continuation & 0x3FU);
    }
    if ((continuation_count == 1 && code_point < 0x80U) ||
        (continuation_count == 2 && code_point < 0x800U) ||
        (continuation_count == 3 && code_point < 0x10000U) ||
        code_point > 0x10FFFFU ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

[[nodiscard]] mtt_status_t reject_unusable_error(mtt_error_v1_t* error) noexcept {
  if (error_is_usable(error)) {
    return MTT_STATUS_OK;
  }
  return MTT_STATUS_ABI_MISMATCH;
}

[[nodiscard]] std::string sha256_hex(
    const uint8_t (&digest)[MTT_SHA256_BYTES]) {
  constexpr char digits[] = "0123456789abcdef";
  std::string encoded;
  encoded.resize(MTT_SHA256_BYTES * 2U);
  for (std::size_t index = 0; index < MTT_SHA256_BYTES; ++index) {
    const uint8_t value = digest[index];
    encoded[index * 2U] = digits[value >> 4U];
    encoded[index * 2U + 1U] = digits[value & 0x0FU];
  }
  return encoded;
}

[[nodiscard]] int hex_digit(const char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

[[nodiscard]] bool decode_sha256_hex(
    const std::string_view encoded,
    uint8_t (&decoded)[MTT_SHA256_BYTES]) noexcept {
  if (encoded.size() != MTT_SHA256_BYTES * 2U) {
    return false;
  }
  for (std::size_t index = 0; index < MTT_SHA256_BYTES; ++index) {
    const int high = hex_digit(encoded[index * 2U]);
    const int low = hex_digit(encoded[index * 2U + 1U]);
    if (high < 0 || low < 0) {
      return false;
    }
    decoded[index] =
        static_cast<uint8_t>((static_cast<unsigned>(high) << 4U) |
                             static_cast<unsigned>(low));
  }
  return true;
}

[[nodiscard]] mtt_status_t bundle_status(
    const magpie_tts_rt::BundleError& error) noexcept {
  if (error.stage() == magpie_tts_rt::BundleStage::sha256 ||
      error.code() == magpie_tts_rt::BundleErrorCode::digest_mismatch ||
      error.code() == magpie_tts_rt::BundleErrorCode::invalid_digest) {
    return MTT_STATUS_HASH_MISMATCH;
  }
  return MTT_STATUS_IO_ERROR;
}

[[nodiscard]] mtt_status_t manifest_status(
    const magpie_tts_rt::ManifestError& error) noexcept {
  if (error.stage() ==
          magpie_tts_rt::ManifestStage::runtime_compatibility ||
      error.code() ==
          magpie_tts_rt::ManifestErrorCode::fingerprint_mismatch) {
    return MTT_STATUS_RUNTIME_MISMATCH;
  }
  if (error.code() == magpie_tts_rt::ManifestErrorCode::io_error) {
    return MTT_STATUS_IO_ERROR;
  }
  return MTT_STATUS_MANIFEST_ERROR;
}

[[nodiscard]] mtt_status_t plugin_status(
    const magpie_tts_rt::PluginLoadError& error) noexcept {
  if (error.code() ==
      magpie_tts_rt::PluginLoadErrorCode::abi_mismatch) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (error.code() ==
      magpie_tts_rt::PluginLoadErrorCode::conflicting_artifact) {
    return MTT_STATUS_RUNTIME_MISMATCH;
  }
  return MTT_STATUS_ENGINE_ERROR;
}

[[nodiscard]] bool restore_cuda_device(
    const int previous_device,
    mtt_error_v1_t* error) noexcept {
  const cudaError_t status = cudaSetDevice(previous_device);
  if (status == cudaSuccess) {
    return true;
  }
  write_error(
      error,
      MTT_STATUS_CUDA_ERROR,
      MTT_ERROR_STAGE_CUDA,
      cudaGetErrorString(status));
  return false;
}

[[nodiscard]] bool reserve_bounded_counter(
    std::atomic<uint32_t>& counter,
    const uint32_t maximum) noexcept {
  uint32_t current = counter.load(std::memory_order_acquire);
  while (current < maximum) {
    if (counter.compare_exchange_weak(
            current,
            current + 1U,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] mtt_status_t runtime_create(
    const mtt_runtime_desc_v1_t* desc,
    mtt_runtime_t** runtime,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (runtime == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_RUNTIME, "runtime output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *runtime = nullptr;
  if (desc == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_RUNTIME, "runtime descriptor is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_runtime_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "runtime descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->cuda_device_index < 0 || desc->flags != 0 || !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_RUNTIME,
        "runtime descriptor requires a non-negative CUDA device and zero flags/reserved fields");
    return MTT_STATUS_INVALID_ARGUMENT;
  }

  auto* created = new (std::nothrow) mtt_runtime(desc->cuda_device_index);
  if (created == nullptr) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_RUNTIME,
        "failed to allocate runtime state");
    return MTT_STATUS_INTERNAL_ERROR;
  }

  int previous_cuda_device = 0;
  const cudaError_t get_device_status = cudaGetDevice(&previous_cuda_device);
  if (get_device_status != cudaSuccess) {
    delete created;
    write_error(
        error, MTT_STATUS_CUDA_ERROR, MTT_ERROR_STAGE_CUDA, cudaGetErrorString(get_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  const cudaError_t set_device_status = cudaSetDevice(desc->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    delete created;
    write_error(
        error, MTT_STATUS_CUDA_ERROR, MTT_ERROR_STAGE_CUDA, cudaGetErrorString(set_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  created->tensorrt_runtime = nvinfer1::createInferRuntime(created->logger);
  if (created->tensorrt_runtime == nullptr) {
    delete created;
    const cudaError_t restore_status = cudaSetDevice(previous_cuda_device);
    if (restore_status != cudaSuccess) {
      write_error(
          error,
          MTT_STATUS_CUDA_ERROR,
          MTT_ERROR_STAGE_CUDA,
          cudaGetErrorString(restore_status));
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_ENGINE_ERROR,
        MTT_ERROR_STAGE_TENSORRT,
        "TensorRT runtime creation failed");
    return MTT_STATUS_ENGINE_ERROR;
  }

  const cudaError_t restore_device_status = cudaSetDevice(previous_cuda_device);
  if (restore_device_status != cudaSuccess) {
    delete created->tensorrt_runtime;
    created->tensorrt_runtime = nullptr;
    delete created;
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(restore_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  *runtime = created;
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t runtime_destroy(
    mtt_runtime_t* runtime,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (runtime == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_RUNTIME, "runtime is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (runtime->live_models.load(std::memory_order_acquire) != 0) {
    write_error(
        error,
        MTT_STATUS_BUSY,
        MTT_ERROR_STAGE_RUNTIME,
        "runtime still owns live models");
    return MTT_STATUS_BUSY;
  }
  const cudaError_t set_device_status =
      cudaSetDevice(runtime->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(set_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }
  delete runtime->tensorrt_runtime;
  runtime->tensorrt_runtime = nullptr;
  delete runtime;
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t model_load(
    mtt_runtime_t* runtime,
    const mtt_model_desc_v1_t* desc,
    mtt_model_t** model,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (model == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_MODEL, "model output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *model = nullptr;
  if (runtime == nullptr || desc == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "runtime and model descriptor are required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_model_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "model descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->flags != 0 || desc->reserved_0 != 0 ||
      !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "model flags and reserved fields must be zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->bundle_path == nullptr || desc->bundle_path_length == 0) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "a non-empty bundle path is required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->bundle_path_length > MTT_MAX_BUNDLE_PATH_BYTES ||
      desc->bundle_path_length >
          static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "bundle path exceeds the ABI v1 byte limit");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  bool digest_is_all_zero = true;
  for (const uint8_t byte : desc->expected_manifest_sha256) {
    if (byte != 0) {
      digest_is_all_zero = false;
      break;
    }
  }
  if (digest_is_all_zero) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "expected manifest SHA-256 trust anchor must not be all zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  const std::string_view path(
      desc->bundle_path,
      static_cast<std::size_t>(desc->bundle_path_length));
  if (path.find('\0') != std::string_view::npos || !is_valid_utf8(path)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "bundle path must be valid UTF-8 without embedded NUL bytes");
    return MTT_STATUS_INVALID_ARGUMENT;
  }

  int previous_cuda_device = 0;
  const cudaError_t get_device_status =
      cudaGetDevice(&previous_cuda_device);
  if (get_device_status != cudaSuccess) {
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(get_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }
  const cudaError_t set_device_status =
      cudaSetDevice(runtime->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(set_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  try {
    magpie_tts_rt::VerifiedRuntimeBundle bundle =
        magpie_tts_rt::load_and_verify_runtime_bundle(
            std::filesystem::path(std::string(path)),
            sha256_hex(desc->expected_manifest_sha256));
    magpie_tts_rt::StartupGoldenFixture golden_fixture =
        magpie_tts_rt::load_startup_golden_fixture(bundle);
    runtime->plugin.authenticate_and_register(
        bundle, runtime->cuda_device_index);
    std::vector<magpie_tts_rt::LoadedEngine> engines =
        magpie_tts_rt::deserialize_verified_engines(
            *runtime->tensorrt_runtime, bundle);
    auto* created = new (std::nothrow) mtt_model(
        runtime,
        std::move(bundle.manifest),
        std::move(golden_fixture),
        std::move(engines));
    if (created == nullptr) {
      if (!restore_cuda_device(previous_cuda_device, error)) {
        return MTT_STATUS_CUDA_ERROR;
      }
      write_error(
          error,
          MTT_STATUS_INTERNAL_ERROR,
          MTT_ERROR_STAGE_MODEL,
          "failed to allocate model state");
      return MTT_STATUS_INTERNAL_ERROR;
    }
    if (!restore_cuda_device(previous_cuda_device, error)) {
      delete created;
      return MTT_STATUS_CUDA_ERROR;
    }
    runtime->live_models.fetch_add(1, std::memory_order_release);
    *model = created;
    return MTT_STATUS_OK;
  } catch (const magpie_tts_rt::BundleError& caught) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    const mtt_status_t status = bundle_status(caught);
    write_error(error, status, MTT_ERROR_STAGE_MODEL, caught.what());
    return status;
  } catch (const magpie_tts_rt::ManifestError& caught) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    const mtt_status_t status = manifest_status(caught);
    const mtt_error_stage_t stage =
        status == MTT_STATUS_RUNTIME_MISMATCH
            ? MTT_ERROR_STAGE_RUNTIME
            : MTT_ERROR_STAGE_MANIFEST;
    write_error(error, status, stage, caught.what());
    return status;
  } catch (const magpie_tts_rt::StartupGoldenError& caught) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_MANIFEST_ERROR,
        MTT_ERROR_STAGE_MANIFEST,
        caught.what());
    return MTT_STATUS_MANIFEST_ERROR;
  } catch (const magpie_tts_rt::RuntimeFingerprintError& caught) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_RUNTIME_MISMATCH,
        MTT_ERROR_STAGE_RUNTIME,
        caught.what());
    return MTT_STATUS_RUNTIME_MISMATCH;
  } catch (const magpie_tts_rt::PluginLoadError& caught) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    const mtt_status_t status = plugin_status(caught);
    write_error(error, status, MTT_ERROR_STAGE_PLUGIN, caught.what());
    return status;
  } catch (const magpie_tts_rt::EngineLoadError& caught) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_ENGINE_ERROR,
        MTT_ERROR_STAGE_TENSORRT,
        caught.what());
    return MTT_STATUS_ENGINE_ERROR;
  } catch (const std::bad_alloc&) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_MODEL,
        "model loading exhausted host memory");
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (const std::exception& caught) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_MODEL,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (...) {
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_MODEL,
        "model loading failed with a non-standard exception");
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

[[nodiscard]] mtt_status_t model_destroy(
    mtt_model_t* model,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (model == nullptr) {
    write_error(error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_MODEL, "model is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (model->live_sessions.load(std::memory_order_acquire) != 0) {
    write_error(
        error,
        MTT_STATUS_BUSY,
        MTT_ERROR_STAGE_MODEL,
        "model still owns live sessions");
    return MTT_STATUS_BUSY;
  }
  mtt_runtime* runtime = model->runtime;
  const cudaError_t set_device_status =
      cudaSetDevice(runtime->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(set_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }
  delete model;
  runtime->live_models.fetch_sub(1, std::memory_order_release);
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t model_get_info(
    mtt_model_t* model,
    mtt_model_info_v1_t* info,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (model == nullptr || info == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_MODEL,
        "model and model info output are required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          info->struct_size,
          info->abi_version,
          sizeof(mtt_model_info_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "model info does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }

  mtt_model_info_v1_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = MTT_ABI_VERSION_1;
  result.tokenizer_vocabulary_size =
      model->manifest.artifacts.tokenizer.tokenizer_vocabulary_size;
  result.text_embedding_rows =
      model->manifest.artifacts.tokenizer.text_embedding_rows;
  result.bos_token_id =
      model->manifest.artifacts.tokenizer.bos_token_id;
  result.eos_token_id =
      model->manifest.artifacts.tokenizer.eos_token_id;
  result.japanese_global_pad_token_id =
      model->manifest.artifacts.tokenizer.japanese_global_pad_token_id;
  result.maximum_text_tokens =
      model->manifest.limits.maximum_text_tokens;
  result.maximum_audio_frames =
      model->manifest.limits.maximum_audio_frames;
  result.sample_rate_hz = model->manifest.codec.sample_rate_hz;
  result.channels = model->manifest.codec.channels;
  result.pcm_format = MTT_PCM_FORMAT_F32_MONO;
  result.codec_frame_samples =
      model->manifest.codec.hop_length_samples;
  result.initial_frames = model->manifest.codec.initial_frames;
  result.steady_frames = model->manifest.codec.steady_frames;
  result.tail_min_frames = model->manifest.codec.tail_min_frames;
  result.tail_max_frames = model->manifest.codec.tail_max_frames;
  if (!decode_sha256_hex(
          model->manifest.artifacts.tokenizer.identity_sha256,
          result.tokenizer_identity_sha256)) {
    write_error(
        error,
        MTT_STATUS_MANIFEST_ERROR,
        MTT_ERROR_STAGE_MANIFEST,
        "authenticated tokenizer identity is not a lowercase SHA-256 digest");
    return MTT_STATUS_MANIFEST_ERROR;
  }
  *info = result;
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t session_create(
    mtt_model_t* model,
    const mtt_session_desc_v1_t* desc,
    mtt_session_t** session,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (session == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_SESSION, "session output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *session = nullptr;
  if (model == nullptr || desc == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_SESSION,
        "model and session descriptor are required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_session_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "session descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->flags != 0 || desc->reserved_0 != 0 ||
      !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_SESSION,
        "session flags and reserved fields must be zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!reserve_bounded_counter(
          model->live_sessions,
          model->manifest.limits.maximum_sessions)) {
    write_error(
        error,
        MTT_STATUS_BUSY,
        MTT_ERROR_STAGE_SESSION,
        "model has reached its authenticated session limit");
    return MTT_STATUS_BUSY;
  }

  int previous_cuda_device = 0;
  const cudaError_t get_device_status =
      cudaGetDevice(&previous_cuda_device);
  if (get_device_status != cudaSuccess) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(get_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }
  const cudaError_t set_device_status =
      cudaSetDevice(model->runtime->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(set_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }

  try {
    auto resources =
        std::make_unique<magpie_tts_rt::SessionResources>(
            model->engines, model->manifest);
    magpie_tts_rt::run_startup_golden_gate(
        model->manifest, model->golden_fixture, *resources);
    auto* created = new (std::nothrow)
        mtt_session(model, std::move(resources));
    if (created == nullptr) {
      model->live_sessions.fetch_sub(1, std::memory_order_release);
      if (!restore_cuda_device(previous_cuda_device, error)) {
        return MTT_STATUS_CUDA_ERROR;
      }
      write_error(
          error,
          MTT_STATUS_INTERNAL_ERROR,
          MTT_ERROR_STAGE_SESSION,
          "failed to allocate session state");
      return MTT_STATUS_INTERNAL_ERROR;
    }
    if (!restore_cuda_device(previous_cuda_device, error)) {
      delete created;
      model->live_sessions.fetch_sub(1, std::memory_order_release);
      return MTT_STATUS_CUDA_ERROR;
    }
    *session = created;
    return MTT_STATUS_OK;
  } catch (const magpie_tts_rt::SessionResourceError& caught) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    const mtt_status_t status =
        caught.code() ==
                magpie_tts_rt::SessionResourceErrorCode::
                    device_memory_limit_exceeded
            ? MTT_STATUS_MANIFEST_ERROR
            : MTT_STATUS_ENGINE_ERROR;
    const mtt_error_stage_t stage =
        status == MTT_STATUS_MANIFEST_ERROR
            ? MTT_ERROR_STAGE_MANIFEST
            : MTT_ERROR_STAGE_TENSORRT;
    write_error(error, status, stage, caught.what());
    return status;
  } catch (const magpie_tts_rt::StartupGoldenError& caught) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    const bool fixture_invalid =
        caught.code() ==
        magpie_tts_rt::StartupGoldenErrorCode::invalid_fixture;
    const bool mismatch =
        caught.code() ==
            magpie_tts_rt::StartupGoldenErrorCode::count_mismatch ||
        caught.code() ==
            magpie_tts_rt::StartupGoldenErrorCode::hash_mismatch;
    const mtt_status_t status =
        fixture_invalid
            ? MTT_STATUS_MANIFEST_ERROR
            : (mismatch ? MTT_STATUS_HASH_MISMATCH
                        : MTT_STATUS_INTERNAL_ERROR);
    write_error(
        error,
        status,
        fixture_invalid ? MTT_ERROR_STAGE_MANIFEST
                        : MTT_ERROR_STAGE_SESSION,
        caught.what());
    return status;
  } catch (const magpie_tts_rt::SynthesisPipelineError& caught) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    mtt_status_t status = MTT_STATUS_ENGINE_ERROR;
    mtt_error_stage_t stage = MTT_ERROR_STAGE_TENSORRT;
    switch (caught.code()) {
      case magpie_tts_rt::SynthesisPipelineErrorCode::cuda_failure:
        status = MTT_STATUS_CUDA_ERROR;
        stage = MTT_ERROR_STAGE_CUDA;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::engine_failure:
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::alignment_failure:
        stage = MTT_ERROR_STAGE_ALIGNMENT;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::local_ar_failure:
        stage = MTT_ERROR_STAGE_PLUGIN;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::codec_failure:
        stage = MTT_ERROR_STAGE_CODEC;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::context_exhausted:
        stage = MTT_ERROR_STAGE_REQUEST;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::invariant_failure:
        status = MTT_STATUS_INTERNAL_ERROR;
        stage = MTT_ERROR_STAGE_SESSION;
        break;
    }
    write_error(error, status, stage, caught.what());
    return status;
  } catch (const magpie_tts_rt::SessionWorkspaceError& caught) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_MANIFEST_ERROR,
        MTT_ERROR_STAGE_MANIFEST,
        caught.what());
    return MTT_STATUS_MANIFEST_ERROR;
  } catch (const magpie_tts_rt::CudaMemoryError& caught) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    const bool authenticated_limit =
        caught.code() ==
        magpie_tts_rt::CudaMemoryErrorCode::budget_exceeded;
    const mtt_status_t status =
        authenticated_limit ? MTT_STATUS_MANIFEST_ERROR
                            : MTT_STATUS_CUDA_ERROR;
    write_error(
        error,
        status,
        authenticated_limit ? MTT_ERROR_STAGE_MANIFEST
                            : MTT_ERROR_STAGE_CUDA,
        caught.what());
    return status;
  } catch (const std::bad_alloc&) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_SESSION,
        "session creation exhausted host memory");
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (const std::exception& caught) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_SESSION,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (...) {
    model->live_sessions.fetch_sub(1, std::memory_order_release);
    if (!restore_cuda_device(previous_cuda_device, error)) {
      return MTT_STATUS_CUDA_ERROR;
    }
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_SESSION,
        "session creation failed with a non-standard exception");
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

[[nodiscard]] mtt_status_t session_destroy(
    mtt_session_t* session,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (session == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_SESSION, "session is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (session->live_requests.load(std::memory_order_acquire) != 0) {
    write_error(
        error,
        MTT_STATUS_BUSY,
        MTT_ERROR_STAGE_SESSION,
        "session still owns a live request");
    return MTT_STATUS_BUSY;
  }
  const cudaError_t set_device_status =
      cudaSetDevice(session->model->runtime->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(set_device_status));
    return MTT_STATUS_CUDA_ERROR;
  }
  try {
    session->resources->synchronize_for_teardown();
  } catch (const magpie_tts_rt::SessionResourceError& caught) {
    write_error(
        error,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        caught.what());
    return MTT_STATUS_CUDA_ERROR;
  }
  mtt_model* model = session->model;
  delete session;
  model->live_sessions.fetch_sub(1, std::memory_order_release);
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_request_state_t public_request_state(
    const magpie_tts_rt::RequestLifecycleState state) {
  switch (state) {
    case magpie_tts_rt::RequestLifecycleState::running:
      return MTT_REQUEST_STATE_RUNNING;
    case magpie_tts_rt::RequestLifecycleState::completed:
      return MTT_REQUEST_STATE_COMPLETED;
    case magpie_tts_rt::RequestLifecycleState::cancelled:
      return MTT_REQUEST_STATE_CANCELLED;
    case magpie_tts_rt::RequestLifecycleState::failed:
      return MTT_REQUEST_STATE_FAILED;
  }
  return MTT_REQUEST_STATE_FAILED;
}

void copy_terminal_message(
    char (&destination)[MTT_ERROR_MESSAGE_CAPACITY],
    const std::string_view source) {
  const std::size_t copy_size =
      std::min<std::size_t>(
          source.size(), MTT_ERROR_MESSAGE_CAPACITY - 1U);
  std::memcpy(destination, source.data(), copy_size);
  destination[copy_size] = '\0';
}

[[nodiscard]] mtt_request_snapshot_v1_t public_snapshot(
    const magpie_tts_rt::RequestStateSnapshot& source) {
  mtt_request_snapshot_v1_t snapshot{};
  snapshot.struct_size = sizeof(snapshot);
  snapshot.abi_version = MTT_ABI_VERSION_1;
  snapshot.revision = source.revision;
  snapshot.state = public_request_state(source.state);
  snapshot.available_audio_leases =
      source.available_audio_leases;
  snapshot.generated_codec_frames =
      source.generated_codec_frames;
  snapshot.published_samples = source.published_samples;
  snapshot.committed_text_tokens =
      source.committed_text_tokens;
  snapshot.terminal_status = source.terminal_status;
  snapshot.terminal_error_stage =
      source.terminal_error_stage;
  copy_terminal_message(
      snapshot.terminal_error_message,
      source.terminal_error_message);
  return snapshot;
}

void fail_request_worker(
    mtt_request* request,
    const mtt_status_t status,
    const mtt_error_stage_t stage,
    const std::string& message,
    const bool poison_session) noexcept {
  if (poison_session) {
    request->session->poisoned.store(
        true, std::memory_order_release);
  }
  try {
    if (!request->state.is_terminal()) {
      request->state.fail(status, stage, message);
    }
  } catch (...) {
    request->session->poisoned.store(
        true, std::memory_order_release);
  }
}

void run_request_worker(
    mtt_request* request,
    const std::uint32_t random_seed) noexcept {
  const cudaError_t set_device_status =
      cudaSetDevice(
          request->session->model->runtime->cuda_device_index);
  if (set_device_status != cudaSuccess) {
    fail_request_worker(
        request,
        MTT_STATUS_CUDA_ERROR,
        MTT_ERROR_STAGE_CUDA,
        cudaGetErrorString(set_device_status),
        true);
    return;
  }
  try {
    magpie_tts_rt::run_synthesis_pipeline(
        request->session->model->manifest,
        *request->session->resources,
        request->text_token_ids,
        random_seed,
        request->state,
        nullptr);
  } catch (const magpie_tts_rt::SynthesisPipelineError& caught) {
    mtt_status_t status = MTT_STATUS_ENGINE_ERROR;
    mtt_error_stage_t stage = MTT_ERROR_STAGE_TENSORRT;
    auto failure_class =
        magpie_tts_rt::RequestWorkerFailureClass::
            execution_state_unknown;
    switch (caught.code()) {
      case magpie_tts_rt::SynthesisPipelineErrorCode::
          cuda_failure:
        status = MTT_STATUS_CUDA_ERROR;
        stage = MTT_ERROR_STAGE_CUDA;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::
          engine_failure:
        stage = MTT_ERROR_STAGE_TENSORRT;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::
          alignment_failure:
        stage = MTT_ERROR_STAGE_ALIGNMENT;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::
          local_ar_failure:
        stage = MTT_ERROR_STAGE_PLUGIN;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::
          codec_failure:
        stage = MTT_ERROR_STAGE_CODEC;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::
          context_exhausted:
        stage = MTT_ERROR_STAGE_REQUEST;
        failure_class =
            magpie_tts_rt::RequestWorkerFailureClass::
                context_exhausted_at_proven_quiescent_boundary;
        break;
      case magpie_tts_rt::SynthesisPipelineErrorCode::
          invariant_failure:
        status = MTT_STATUS_INTERNAL_ERROR;
        stage = MTT_ERROR_STAGE_REQUEST;
        break;
    }
    fail_request_worker(
        request,
        status,
        stage,
        caught.what(),
        magpie_tts_rt::request_failure_requires_session_poison(
            failure_class));
  } catch (const std::bad_alloc&) {
    // Allocation can fail after TensorRT/CUDA has accepted work but before
    // the host records the corresponding completion flag. That path cannot
    // prove workspace quiescence, so the session must never be reused.
    fail_request_worker(
        request,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        "synthesis worker exhausted host memory",
        magpie_tts_rt::request_failure_requires_session_poison(
            magpie_tts_rt::RequestWorkerFailureClass::
                host_allocation_after_possible_enqueue));
  } catch (const std::exception& caught) {
    fail_request_worker(
        request,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what(),
        true);
  } catch (...) {
    fail_request_worker(
        request,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        "synthesis worker failed with a non-standard exception",
        true);
  }
}

[[nodiscard]] mtt_status_t request_start(
    mtt_session_t* session,
    const mtt_request_desc_v1_t* desc,
    mtt_request_t** request,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (request == nullptr) {
    write_error(
        error, MTT_STATUS_INVALID_ARGUMENT, MTT_ERROR_STAGE_REQUEST, "request output is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  *request = nullptr;
  if (session == nullptr || desc == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "session and request descriptor are required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          desc->struct_size, desc->abi_version, sizeof(mtt_request_desc_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "request descriptor does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (desc->text_token_ids == nullptr || desc->text_token_count == 0) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "a non-empty token sequence is required");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->text_token_count > MTT_MAX_TEXT_TOKENS) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "token sequence exceeds the ABI v1 token limit");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  for (uint64_t index = 0; index < desc->text_token_count; ++index) {
    const int64_t token = desc->text_token_ids[index];
    if (token < 0 ||
        token > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      write_error(
          error,
          MTT_STATUS_INVALID_ARGUMENT,
          MTT_ERROR_STAGE_REQUEST,
          "every token identifier must fit a non-negative INT32 value");
      return MTT_STATUS_INVALID_ARGUMENT;
    }
  }
  if (desc->random_seed >
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "random seed must be in [0, 2^32)");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->flags != 0 || desc->reserved_0 != 0 || !reserved_is_zero(desc->reserved)) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "request flags and reserved fields must be zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (desc->text_token_count >
      session->model->manifest.limits.maximum_text_tokens) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "token sequence exceeds the authenticated model limit");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (session->poisoned.load(std::memory_order_acquire)) {
    write_error(
        error,
        MTT_STATUS_POISONED,
        MTT_ERROR_STAGE_SESSION,
        "session is poisoned by an earlier execution failure");
    return MTT_STATUS_POISONED;
  }

  std::vector<std::int32_t> prepared_tokens;
  try {
    const std::span<const std::int64_t> input_tokens(
        desc->text_token_ids,
        static_cast<std::size_t>(desc->text_token_count));
    const magpie_tts_rt::PreparedTokenValidation token_validation =
        magpie_tts_rt::validate_prepared_token_ids(
            input_tokens,
            session->model->manifest.artifacts.tokenizer
                .tokenizer_vocabulary_size,
            session->model->manifest.artifacts.tokenizer.eos_token_id);
    if (!token_validation.valid()) {
      write_error(
          error,
          MTT_STATUS_INVALID_ARGUMENT,
          MTT_ERROR_STAGE_REQUEST,
          "token sequence violates the authenticated prepared frontend "
          "contract");
      return MTT_STATUS_INVALID_ARGUMENT;
    }
    prepared_tokens.reserve(
        static_cast<std::size_t>(desc->text_token_count));
    for (std::uint64_t index = 0;
         index < desc->text_token_count;
         ++index) {
      const std::int64_t token = desc->text_token_ids[index];
      prepared_tokens.push_back(
          static_cast<std::int32_t>(token));
    }
  } catch (const std::bad_alloc&) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        "copying request tokens exhausted host memory");
    return MTT_STATUS_INTERNAL_ERROR;
  }

  if (!reserve_bounded_counter(
          session->live_requests,
          magpie_tts_rt::kMaximumActiveRequestsPerSession)) {
    write_error(
        error,
        MTT_STATUS_BUSY,
        MTT_ERROR_STAGE_REQUEST,
        "session already owns its active request");
    return MTT_STATUS_BUSY;
  }

  mtt_request* created = nullptr;
  try {
    created = new mtt_request(
        session, std::move(prepared_tokens));
    created->worker = std::thread(
        run_request_worker,
        created,
        static_cast<std::uint32_t>(desc->random_seed));
  } catch (const std::bad_alloc&) {
    delete created;
    session->live_requests.fetch_sub(
        1, std::memory_order_release);
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        "allocating request state exhausted host memory");
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (const std::exception& caught) {
    delete created;
    session->live_requests.fetch_sub(
        1, std::memory_order_release);
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (...) {
    delete created;
    session->live_requests.fetch_sub(
        1, std::memory_order_release);
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        "request creation failed with a non-standard exception");
    return MTT_STATUS_INTERNAL_ERROR;
  }
  *request = created;
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t validate_request_call(
    mtt_request_t* request,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (request == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "request is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t validate_snapshot_output(
    mtt_request_t* request,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error) noexcept {
  const mtt_status_t request_status =
      validate_request_call(request, error);
  if (request_status != MTT_STATUS_OK) {
    return request_status;
  }
  if (snapshot == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "request snapshot is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(
          snapshot->struct_size,
          snapshot->abi_version,
          sizeof(mtt_request_snapshot_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "request snapshot does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  return MTT_STATUS_OK;
}

[[nodiscard]] mtt_status_t request_poll(
    mtt_request_t* request,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error) noexcept {
  const mtt_status_t validation =
      validate_snapshot_output(request, snapshot, error);
  if (validation != MTT_STATUS_OK) {
    return validation;
  }
  try {
    const mtt_request_snapshot_v1_t result =
        public_snapshot(request->state.snapshot());
    *snapshot = result;
    return MTT_STATUS_OK;
  } catch (const std::exception& caught) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

[[nodiscard]] mtt_status_t request_wait(
    mtt_request_t* request,
    const uint64_t after_revision,
    const uint64_t timeout_nanoseconds,
    mtt_request_snapshot_v1_t* snapshot,
    mtt_error_v1_t* error) noexcept {
  const mtt_status_t validation =
      validate_snapshot_output(request, snapshot, error);
  if (validation != MTT_STATUS_OK) {
    return validation;
  }
  if (timeout_nanoseconds >
      static_cast<std::uint64_t>(
          std::numeric_limits<
              std::chrono::nanoseconds::rep>::max())) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "wait timeout does not fit the runtime duration type");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  try {
    magpie_tts_rt::RequestStateSnapshot result{};
    const bool changed = request->state.wait_for_revision(
        after_revision,
        std::chrono::nanoseconds(
            static_cast<std::chrono::nanoseconds::rep>(
                timeout_nanoseconds)),
        result);
    if (!changed) {
      write_error(
          error,
          MTT_STATUS_TIMEOUT,
          MTT_ERROR_STAGE_REQUEST,
          "");
      return MTT_STATUS_TIMEOUT;
    }
    *snapshot = public_snapshot(result);
    return MTT_STATUS_OK;
  } catch (const std::exception& caught) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

[[nodiscard]] mtt_status_t request_cancel(
    mtt_request_t* request,
    mtt_error_v1_t* error) noexcept {
  const mtt_status_t validation =
      validate_request_call(request, error);
  if (validation != MTT_STATUS_OK) {
    return validation;
  }
  try {
    if (!request->state.request_cancellation()) {
      write_error(
          error,
          MTT_STATUS_INVALID_ARGUMENT,
          MTT_ERROR_STAGE_REQUEST,
          "only a running request can be cancelled");
      return MTT_STATUS_INVALID_ARGUMENT;
    }
    return MTT_STATUS_OK;
  } catch (const std::exception& caught) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

[[nodiscard]] mtt_status_t request_destroy(
    mtt_request_t* request,
    mtt_error_v1_t* error) noexcept {
  const mtt_status_t validation =
      validate_request_call(request, error);
  if (validation != MTT_STATUS_OK) {
    return validation;
  }
  try {
    if (!request->state.is_terminal()) {
      write_error(
          error,
          MTT_STATUS_BUSY,
          MTT_ERROR_STAGE_REQUEST,
          "request is still running");
      return MTT_STATUS_BUSY;
    }
    if (request->state.has_live_leases() ||
        !request->lease_alignment_events.empty()) {
      write_error(
          error,
          MTT_STATUS_BUSY,
          MTT_ERROR_STAGE_REQUEST,
          "request still owns live audio leases");
      return MTT_STATUS_BUSY;
    }
    if (request->worker.joinable()) {
      request->worker.join();
    }
    mtt_session* session = request->session;
    delete request;
    session->live_requests.fetch_sub(
        1, std::memory_order_release);
    return MTT_STATUS_OK;
  } catch (const std::exception& caught) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

[[nodiscard]] mtt_status_t audio_acquire(
    mtt_request_t* request,
    mtt_audio_lease_v1_t* lease,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (lease == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "audio lease is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (!has_valid_header(lease->struct_size, lease->abi_version, sizeof(mtt_audio_lease_v1_t))) {
    write_error(
        error,
        MTT_STATUS_ABI_MISMATCH,
        MTT_ERROR_STAGE_ABI,
        "audio lease does not match ABI version 1");
    return MTT_STATUS_ABI_MISMATCH;
  }
  if (request == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "request is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  try {
    const magpie_tts_rt::AudioLeaseView view =
        request->state.acquire_audio();
    const std::uint64_t lease_id = view.lease_id;
    try {
      std::vector<mtt_alignment_event_v1_t> events;
      events.reserve(
          static_cast<std::size_t>(
              view.alignment_event_count));
      for (std::uint64_t index = 0;
           index < view.alignment_event_count;
           ++index) {
        const magpie_tts_rt::AlignmentProgress& source =
            view.alignment_events[index];
        mtt_alignment_event_v1_t event{};
        event.struct_size = sizeof(event);
        event.abi_version = MTT_ABI_VERSION_1;
        event.sample_index = source.sample_index;
        event.committed_text_tokens =
            source.committed_text_tokens;
        events.push_back(event);
      }
      const auto [iterator, inserted] =
          request->lease_alignment_events.emplace(
              lease_id, std::move(events));
      if (!inserted) {
        request->state.release_audio(lease_id);
        write_error(
            error,
            MTT_STATUS_POISONED,
            MTT_ERROR_STAGE_REQUEST,
            "audio lease identifier was reused");
        request->session->poisoned.store(
            true, std::memory_order_release);
        return MTT_STATUS_POISONED;
      }
      const std::vector<mtt_alignment_event_v1_t>& stored =
          iterator->second;
      mtt_audio_lease_v1_t result{};
      result.struct_size = sizeof(result);
      result.abi_version = MTT_ABI_VERSION_1;
      result.lease_id = lease_id;
      result.samples = view.samples;
      result.sample_count = view.sample_count;
      result.first_sample_index = view.first_sample_index;
      result.sequence = view.sequence;
      result.sample_rate_hz =
          magpie_tts_rt::kStreamingSampleRateHz;
      result.channels = 1;
      result.format = MTT_PCM_FORMAT_F32_MONO;
      result.flags =
          (view.first ? MTT_AUDIO_FLAG_FIRST : 0U) |
          (view.final ? MTT_AUDIO_FLAG_FINAL : 0U) |
          (view.alignment_valid
               ? MTT_AUDIO_FLAG_ALIGNMENT_VALID
               : 0U);
      result.committed_text_tokens =
          view.committed_text_tokens;
      result.alignment_events =
          stored.empty() ? nullptr : stored.data();
      result.alignment_event_count = stored.size();
      *lease = result;
      return MTT_STATUS_OK;
    } catch (...) {
      request->state.release_audio(lease_id);
      throw;
    }
  } catch (const magpie_tts_rt::RequestStateError& caught) {
    if (caught.code() ==
        magpie_tts_rt::RequestStateErrorCode::no_audio) {
      write_error(
          error,
          MTT_STATUS_WOULD_BLOCK,
          MTT_ERROR_STAGE_REQUEST,
          "");
      return MTT_STATUS_WOULD_BLOCK;
    }
    const mtt_status_t status =
        caught.code() ==
                magpie_tts_rt::RequestStateErrorCode::
                    lease_sequence_exhausted
            ? MTT_STATUS_POISONED
            : MTT_STATUS_INTERNAL_ERROR;
    if (status == MTT_STATUS_POISONED) {
      request->session->poisoned.store(
          true, std::memory_order_release);
    }
    write_error(
        error, status, MTT_ERROR_STAGE_REQUEST, caught.what());
    return status;
  } catch (const std::bad_alloc&) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        "allocating audio lease metadata exhausted host memory");
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (const std::exception& caught) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

[[nodiscard]] mtt_status_t audio_release(
    mtt_request_t* request,
    const uint64_t lease_id,
    mtt_error_v1_t* error) noexcept {
  if (reject_unusable_error(error) != MTT_STATUS_OK) {
    return MTT_STATUS_ABI_MISMATCH;
  }
  clear_error(error);
  if (lease_id == 0) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "audio lease identifier must be non-zero");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (request == nullptr) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "request is null");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  const auto metadata =
      request->lease_alignment_events.find(lease_id);
  if (metadata == request->lease_alignment_events.end()) {
    write_error(
        error,
        MTT_STATUS_INVALID_ARGUMENT,
        MTT_ERROR_STAGE_REQUEST,
        "lease identifier is not live on this request");
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  try {
    request->state.release_audio(lease_id);
    request->lease_alignment_events.erase(metadata);
    return MTT_STATUS_OK;
  } catch (const magpie_tts_rt::RequestStateError& caught) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  } catch (const std::exception& caught) {
    write_error(
        error,
        MTT_STATUS_INTERNAL_ERROR,
        MTT_ERROR_STAGE_REQUEST,
        caught.what());
    return MTT_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace

extern "C" MTT_API mtt_status_t mtt_get_api(
    const uint32_t requested_abi_version,
    mtt_api_v1_t* api) {
  if (api == nullptr) {
    return MTT_STATUS_INVALID_ARGUMENT;
  }
  if (requested_abi_version != MTT_ABI_VERSION_1 ||
      !has_valid_header(api->struct_size, api->abi_version, sizeof(mtt_api_v1_t))) {
    return MTT_STATUS_ABI_MISMATCH;
  }

  api->runtime_create = runtime_create;
  api->runtime_destroy = runtime_destroy;
  api->model_load = model_load;
  api->model_destroy = model_destroy;
  api->model_get_info = model_get_info;
  api->session_create = session_create;
  api->session_destroy = session_destroy;
  api->request_start = request_start;
  api->request_poll = request_poll;
  api->request_wait = request_wait;
  api->request_cancel = request_cancel;
  api->request_destroy = request_destroy;
  api->audio_acquire = audio_acquire;
  api->audio_release = audio_release;
  return MTT_STATUS_OK;
}

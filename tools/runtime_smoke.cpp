#include "magpie_tts_rt/magpie_tts_rt.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uintmax_t kMaximumSmokeJsonBytes = 16U * 1024U * 1024U;
constexpr std::uint32_t kCanonicalSampleRateHz = 22'050U;
constexpr std::uint32_t kCanonicalChannels = 1U;
constexpr std::uint64_t kCodecFrameSamples = 1'024U;
constexpr std::uint64_t kDecoderStepSamples = 2U * kCodecFrameSamples;
constexpr std::uint32_t kKnownAudioFlags =
    MTT_AUDIO_FLAG_FIRST | MTT_AUDIO_FLAG_FINAL |
    MTT_AUDIO_FLAG_ALIGNMENT_VALID;
constexpr std::size_t kMaximumPublicArtifactNameBytes = 255U;

struct Arguments {
  std::filesystem::path library;
  std::filesystem::path bundle;
  std::array<std::uint8_t, MTT_SHA256_BYTES> manifest_sha256{};
  std::string manifest_sha256_hex;
  std::optional<std::filesystem::path> receipt_json;
  std::int32_t cuda_device_index = 0;
  std::chrono::milliseconds request_timeout{120'000};
  bool help = false;
};

struct GoldenInput {
  std::vector<std::int64_t> token_ids;
  std::uint64_t seed = 0;
};

struct StreamMetrics {
  std::uint64_t chunk_count = 0;
  std::uint64_t sample_count = 0;
  std::uint64_t expected_sequence = 0;
  std::uint64_t expected_sample_index = 0;
  std::optional<std::uint64_t> last_lease_id;
  std::uint64_t last_alignment_sample = 0;
  std::uint64_t last_alignment_token = 0;
  bool first_seen = false;
  bool final_seen = false;
  bool terminal_control_marker_seen = false;
  std::optional<Clock::time_point> first_audio_at;
  std::optional<mtt_request_snapshot_v1_t> last_snapshot;
  std::optional<mtt_request_snapshot_v1_t> final_snapshot;
};

struct NativeObjects {
  void* library = nullptr;
  mtt_api_v1_t api{};
  mtt_runtime_t* runtime = nullptr;
  mtt_model_t* model = nullptr;
  mtt_session_t* session = nullptr;
  mtt_request_t* request = nullptr;
};

[[nodiscard]] std::string usage() {
  return
      "usage: mtt-runtime-smoke --library /absolute/libmagpie_tts_rt.so "
      "--bundle /absolute/bundle --manifest-sha256 <64-lowercase-hex> "
      "[--cuda-device INDEX] [--request-timeout-ms MILLISECONDS] "
      "[--receipt-json /absolute/new-receipt.json]\n"
      "\n"
      "The optional receipt is created only after the complete stream and "
      "native teardown pass.\n"
      "An existing receipt is never overwritten.";
}

[[nodiscard]] std::string require_option_value(
    int& index, const int argc, char** argv, const std::string_view option) {
  ++index;
  if (index >= argc) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  return argv[index];
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(
    const std::string_view value, const std::string_view label) {
  Integer parsed{};
  const char* const begin = value.data();
  const char* const end = begin + value.size();
  const auto [position, error] = std::from_chars(begin, end, parsed);
  if (error != std::errc{} || position != end) {
    throw std::invalid_argument(
        std::string(label) + " must be a base-10 integer");
  }
  return parsed;
}

[[nodiscard]] std::uint8_t hex_nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  throw std::invalid_argument(
      "manifest SHA-256 must contain exactly 64 lowercase hex digits");
}

[[nodiscard]] std::array<std::uint8_t, MTT_SHA256_BYTES> parse_sha256(
    const std::string_view value) {
  if (value.size() != MTT_SHA256_BYTES * 2U) {
    throw std::invalid_argument(
        "manifest SHA-256 must contain exactly 64 lowercase hex digits");
  }
  std::array<std::uint8_t, MTT_SHA256_BYTES> digest{};
  for (std::size_t index = 0; index < digest.size(); ++index) {
    const auto high = hex_nibble(value[index * 2U]);
    const auto low = hex_nibble(value[index * 2U + 1U]);
    digest[index] = static_cast<std::uint8_t>((high << 4U) | low);
  }
  if (digest == std::array<std::uint8_t, MTT_SHA256_BYTES>{}) {
    throw std::invalid_argument("manifest SHA-256 must not be all zero");
  }
  return digest;
}

void require_absolute_regular_file(
    const std::filesystem::path& path, const std::string_view label) {
  if (!path.is_absolute()) {
    throw std::invalid_argument(std::string(label) + " must be absolute");
  }
  const auto status = std::filesystem::symlink_status(path);
  if (!std::filesystem::is_regular_file(status)) {
    throw std::invalid_argument(
        std::string(label) + " must be a non-symlink regular file");
  }
  if (std::filesystem::file_size(path) == 0U) {
    throw std::invalid_argument(std::string(label) + " must not be empty");
  }
}

void require_absolute_directory(
    const std::filesystem::path& path, const std::string_view label) {
  if (!path.is_absolute()) {
    throw std::invalid_argument(std::string(label) + " must be absolute");
  }
  const auto status = std::filesystem::symlink_status(path);
  if (!std::filesystem::is_directory(status)) {
    throw std::invalid_argument(
        std::string(label) + " must be a non-symlink directory");
  }
}

void require_new_absolute_receipt_path(
    const std::filesystem::path& path) {
  if (!path.is_absolute() || path.filename().empty()) {
    throw std::invalid_argument(
        "--receipt-json must be an absolute file path");
  }
  require_absolute_directory(path.parent_path(), "--receipt-json parent");
  if (std::filesystem::exists(std::filesystem::symlink_status(path))) {
    throw std::invalid_argument(
        "--receipt-json target already exists; receipts are immutable");
  }
}

[[nodiscard]] bool is_ascii_alphanumeric(
    const unsigned char character) {
  return
      (character >= static_cast<unsigned char>('a') &&
       character <= static_cast<unsigned char>('z')) ||
      (character >= static_cast<unsigned char>('A') &&
       character <= static_cast<unsigned char>('Z')) ||
      (character >= static_cast<unsigned char>('0') &&
       character <= static_cast<unsigned char>('9'));
}

[[nodiscard]] bool is_public_artifact_name_character(
    const unsigned char character) {
  return
      is_ascii_alphanumeric(character) ||
      character == static_cast<unsigned char>('.') ||
      character == static_cast<unsigned char>('_') ||
      character == static_cast<unsigned char>('-') ||
      character == static_cast<unsigned char>('+') ||
      character == static_cast<unsigned char>('~');
}

[[nodiscard]] std::string public_artifact_name(
    const std::filesystem::path& path, const std::string_view label) {
  auto normalized = path.lexically_normal();
  if (normalized.filename().empty() &&
      normalized != normalized.root_path()) {
    normalized = normalized.parent_path();
  }
  const auto name = normalized.filename().string();
  if (name.empty() ||
      name.size() > kMaximumPublicArtifactNameBytes ||
      !is_ascii_alphanumeric(
          static_cast<unsigned char>(name.front())) ||
      !std::all_of(
          name.begin(), name.end(), [](const char character) {
            return is_public_artifact_name_character(
                static_cast<unsigned char>(character));
          })) {
    throw std::runtime_error(
        std::string(label) +
        " basename cannot be represented in a public receipt");
  }
  return name;
}

[[nodiscard]] std::uintmax_t public_regular_file_size(
    const std::filesystem::path& path, const std::string_view label) {
  require_absolute_regular_file(path, label);
  return std::filesystem::file_size(path);
}

[[nodiscard]] Arguments parse_arguments(const int argc, char** argv) {
  Arguments arguments;
  bool have_library = false;
  bool have_bundle = false;
  bool have_manifest = false;
  bool have_cuda_device = false;
  bool have_request_timeout = false;
  bool have_receipt_json = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--library") {
      if (have_library) {
        throw std::invalid_argument("--library may be specified only once");
      }
      arguments.library =
          require_option_value(index, argc, argv, option);
      have_library = true;
    } else if (option == "--bundle") {
      if (have_bundle) {
        throw std::invalid_argument("--bundle may be specified only once");
      }
      arguments.bundle =
          require_option_value(index, argc, argv, option);
      have_bundle = true;
    } else if (option == "--manifest-sha256") {
      if (have_manifest) {
        throw std::invalid_argument(
            "--manifest-sha256 may be specified only once");
      }
      arguments.manifest_sha256_hex =
          require_option_value(index, argc, argv, option);
      arguments.manifest_sha256 =
          parse_sha256(arguments.manifest_sha256_hex);
      have_manifest = true;
    } else if (option == "--cuda-device") {
      if (have_cuda_device) {
        throw std::invalid_argument(
            "--cuda-device may be specified only once");
      }
      arguments.cuda_device_index = parse_integer<std::int32_t>(
          require_option_value(index, argc, argv, option), option);
      if (arguments.cuda_device_index < 0) {
        throw std::invalid_argument("--cuda-device must be non-negative");
      }
      have_cuda_device = true;
    } else if (option == "--request-timeout-ms") {
      if (have_request_timeout) {
        throw std::invalid_argument(
            "--request-timeout-ms may be specified only once");
      }
      const auto timeout = parse_integer<std::uint64_t>(
          require_option_value(index, argc, argv, option), option);
      if (timeout == 0U ||
          timeout >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(
            "--request-timeout-ms must be a positive int64 value");
      }
      arguments.request_timeout = std::chrono::milliseconds(
          static_cast<std::chrono::milliseconds::rep>(timeout));
      have_request_timeout = true;
    } else if (option == "--receipt-json") {
      if (have_receipt_json) {
        throw std::invalid_argument(
            "--receipt-json may be specified only once");
      }
      arguments.receipt_json =
          require_option_value(index, argc, argv, option);
      have_receipt_json = true;
    } else if (option == "--help" || option == "-h") {
      arguments.help = true;
    } else {
      throw std::invalid_argument(
          "unknown option " + std::string(option) + "\n" + usage());
    }
  }
  if (arguments.help) {
    return arguments;
  }
  if (!have_library || !have_bundle || !have_manifest) {
    throw std::invalid_argument("missing required option\n" + usage());
  }
  require_absolute_regular_file(arguments.library, "--library");
  require_absolute_directory(arguments.bundle, "--bundle");
  if (arguments.receipt_json.has_value()) {
    require_new_absolute_receipt_path(*arguments.receipt_json);
  }
  return arguments;
}

[[nodiscard]] nlohmann::json read_json_file(
    const std::filesystem::path& path, const std::string_view label) {
  require_absolute_regular_file(path, label);
  if (std::filesystem::file_size(path) > kMaximumSmokeJsonBytes) {
    throw std::runtime_error(
        std::string(label) + " exceeds the 16 MiB smoke-input limit");
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error(
        "failed to open " + std::string(label) + ": " + path.string());
  }
  nlohmann::json document;
  input >> document;
  if (!input.eof() && input.fail()) {
    throw std::runtime_error(
        "failed to read " + std::string(label) + ": " + path.string());
  }
  return document;
}

[[nodiscard]] std::string require_json_string(
    const nlohmann::json& parent, const std::string_view member,
    const std::string_view pointer) {
  const auto iterator = parent.find(member);
  if (iterator == parent.end() || !iterator->is_string()) {
    throw std::runtime_error(
        std::string(pointer) + " must be a string");
  }
  return iterator->get<std::string>();
}

[[nodiscard]] const nlohmann::json& require_json_object(
    const nlohmann::json& parent, const std::string_view member,
    const std::string_view pointer) {
  const auto iterator = parent.find(member);
  if (iterator == parent.end() || !iterator->is_object()) {
    throw std::runtime_error(
        std::string(pointer) + " must be an object");
  }
  return *iterator;
}

[[nodiscard]] GoldenInput load_golden_input(
    const std::filesystem::path& bundle) {
  const auto manifest_path = bundle / "runtime-bundle-manifest.json";
  const auto manifest = read_json_file(manifest_path, "bundle manifest");
  const auto& fixture_record =
      require_json_object(manifest, "golden_fixture", "/golden_fixture");
  const auto fixture_relative = require_json_string(
      fixture_record, "path", "/golden_fixture/path");
  const std::filesystem::path relative_path(fixture_relative);
  if (relative_path.empty() || relative_path.is_absolute()) {
    throw std::runtime_error(
        "/golden_fixture/path must be a non-empty relative path");
  }
  for (const auto& component : relative_path) {
    if (component == "." || component == "..") {
      throw std::runtime_error(
          "/golden_fixture/path must not contain dot components");
    }
  }
  const auto fixture = read_json_file(
      bundle / relative_path, "bundle golden fixture");
  const auto token_iterator = fixture.find("prepared_token_ids");
  if (token_iterator == fixture.end() || !token_iterator->is_array() ||
      token_iterator->empty() ||
      token_iterator->size() > MTT_MAX_TEXT_TOKENS) {
    throw std::runtime_error(
        "/prepared_token_ids must be a non-empty bounded array");
  }
  GoldenInput input;
  input.token_ids.reserve(token_iterator->size());
  for (const auto& token : *token_iterator) {
    if (!token.is_number_unsigned() && !token.is_number_integer()) {
      throw std::runtime_error(
          "/prepared_token_ids entries must be non-negative integers");
    }
    const auto value = token.get<std::int64_t>();
    if (value < 0 ||
        value > static_cast<std::int64_t>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::runtime_error(
          "/prepared_token_ids entry is outside INT32");
    }
    input.token_ids.push_back(value);
  }
  const auto seed_iterator = fixture.find("seed");
  if (seed_iterator == fixture.end() ||
      (!seed_iterator->is_number_unsigned() &&
       !seed_iterator->is_number_integer())) {
    throw std::runtime_error("/seed must be a uint32");
  }
  input.seed = seed_iterator->get<std::uint64_t>();
  if (input.seed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("/seed must be a uint32");
  }
  return input;
}

[[nodiscard]] mtt_error_v1_t fresh_error() {
  mtt_error_v1_t error{};
  error.struct_size = sizeof(error);
  error.abi_version = MTT_ABI_VERSION_1;
  return error;
}

[[nodiscard]] bool is_declared_status(const mtt_status_t status) {
  return status >= MTT_STATUS_OK &&
         status <= MTT_STATUS_INTERNAL_ERROR;
}

[[nodiscard]] bool is_declared_error_stage(
    const mtt_error_stage_t stage) {
  return stage >= MTT_ERROR_STAGE_NONE &&
         stage <= MTT_ERROR_STAGE_CODEC;
}

[[nodiscard]] bool is_terminal_failure_status(
    const mtt_status_t status) {
  switch (status) {
    case MTT_STATUS_INVALID_ARGUMENT:
    case MTT_STATUS_ABI_MISMATCH:
    case MTT_STATUS_IO_ERROR:
    case MTT_STATUS_MANIFEST_ERROR:
    case MTT_STATUS_RUNTIME_MISMATCH:
    case MTT_STATUS_HASH_MISMATCH:
    case MTT_STATUS_ENGINE_ERROR:
    case MTT_STATUS_CUDA_ERROR:
    case MTT_STATUS_POISONED:
    case MTT_STATUS_UNAVAILABLE:
    case MTT_STATUS_INTERNAL_ERROR:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] std::optional<std::string> fixed_message(
    const char* const message, const std::size_t capacity) {
  const void* const terminator = std::memchr(message, '\0', capacity);
  if (terminator == nullptr) {
    return std::nullopt;
  }
  const auto* const end = static_cast<const char*>(terminator);
  return std::string(message, end);
}

[[nodiscard]] mtt_status_t validate_call_status(
    const std::string_view operation, const mtt_status_t status,
    const mtt_error_v1_t& error) {
  const auto message =
      fixed_message(error.message, MTT_ERROR_MESSAGE_CAPACITY);
  std::string reason;
  if (error.struct_size != sizeof(error)) {
    reason = "error struct_size does not match ABI v1";
  } else if (error.abi_version != MTT_ABI_VERSION_1) {
    reason = "error abi_version does not match ABI v1";
  } else if (!is_declared_status(status)) {
    reason = "returned status is not declared by ABI v1";
  } else if (error.code != status) {
    reason = "error code does not match the returned status";
  } else if (!is_declared_error_stage(error.stage)) {
    reason = "error stage is not declared by ABI v1";
  } else if (!message.has_value()) {
    reason = "error message is not NUL-terminated";
  } else if (
      status == MTT_STATUS_OK &&
      (error.stage != MTT_ERROR_STAGE_NONE || !message->empty())) {
    reason = "successful call returned a diagnostic payload";
  } else if (
      status != MTT_STATUS_OK && error.stage == MTT_ERROR_STAGE_NONE) {
    reason = "non-success status has no error stage";
  } else if (
      (status == MTT_STATUS_TIMEOUT ||
       status == MTT_STATUS_WOULD_BLOCK) &&
      error.stage != MTT_ERROR_STAGE_REQUEST) {
    reason = "request control status has a non-request error stage";
  } else if (
      status != MTT_STATUS_OK &&
      status != MTT_STATUS_TIMEOUT &&
      status != MTT_STATUS_WOULD_BLOCK && message->empty()) {
    reason = "non-control failure has no diagnostic message";
  }
  if (!reason.empty()) {
    throw std::runtime_error(
        std::string(operation) +
        " returned an invalid C ABI error buffer: " + reason);
  }
  return status;
}

[[nodiscard]] std::string native_error_message(
    const std::string_view operation, const mtt_status_t status,
    const mtt_error_v1_t& error) {
  const auto message =
      fixed_message(error.message, MTT_ERROR_MESSAGE_CAPACITY);
  return std::string(operation) + " failed: status=" +
         std::to_string(status) + " stage=" + std::to_string(error.stage) +
         " message=" + message.value_or("<not-NUL-terminated>");
}

void require_ok(
    const std::string_view operation, const mtt_status_t status,
    const mtt_error_v1_t& error) {
  const auto validated = validate_call_status(operation, status, error);
  if (validated != MTT_STATUS_OK) {
    throw std::runtime_error(
        native_error_message(operation, validated, error));
  }
}

template <typename Function>
[[nodiscard]] bool cleanup_call(
    const std::string_view operation, Function&& function, bool& clean) {
  auto error = fresh_error();
  try {
    const auto status =
        validate_call_status(operation, function(&error), error);
    if (status == MTT_STATUS_OK) {
      return true;
    }
    clean = false;
    std::cerr << "cleanup: "
              << native_error_message(operation, status, error) << '\n';
    return false;
  } catch (const std::exception& exception) {
    clean = false;
    std::cerr << "cleanup: " << exception.what() << '\n';
    return false;
  }
}

[[nodiscard]] bool drain_for_cleanup(
    NativeObjects& native, const bool reject_audio) {
  bool clean = true;
  for (;;) {
    mtt_audio_lease_v1_t lease{};
    lease.struct_size = sizeof(lease);
    lease.abi_version = MTT_ABI_VERSION_1;
    auto error = fresh_error();
    mtt_status_t status = MTT_STATUS_INTERNAL_ERROR;
    try {
      status = validate_call_status(
          "audio_acquire",
          native.api.audio_acquire(native.request, &lease, &error),
          error);
    } catch (const std::exception& exception) {
      std::cerr << "cleanup: " << exception.what() << '\n';
      return false;
    }
    if (status == MTT_STATUS_WOULD_BLOCK) {
      return clean;
    }
    if (status != MTT_STATUS_OK) {
      std::cerr << "cleanup: "
                << native_error_message("audio_acquire", status, error)
                << '\n';
      return false;
    }
    const bool released = cleanup_call(
        "audio_release",
        [&](mtt_error_v1_t* release_error) {
          return native.api.audio_release(
              native.request, lease.lease_id, release_error);
        },
        clean);
    if (!released) {
      return false;
    }
    if (reject_audio) {
      clean = false;
      std::cerr
          << "cleanup: audio arrived after the validated FINAL lease\n";
    }
  }
}

[[nodiscard]] bool cleanup_native(
    NativeObjects& native,
    const bool expect_completed_and_empty = false) {
  bool clean = true;
  if (native.request != nullptr) {
    mtt_request_snapshot_v1_t snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    snapshot.abi_version = MTT_ABI_VERSION_1;
    auto error = fresh_error();
    mtt_status_t poll_status = MTT_STATUS_INTERNAL_ERROR;
    try {
      poll_status = validate_call_status(
          "request_poll",
          native.api.request_poll(native.request, &snapshot, &error),
          error);
    } catch (const std::exception& exception) {
      clean = false;
      std::cerr << "cleanup: " << exception.what() << '\n';
    }
    if (expect_completed_and_empty &&
        (poll_status != MTT_STATUS_OK ||
         snapshot.state != MTT_REQUEST_STATE_COMPLETED)) {
      clean = false;
      std::cerr
          << "cleanup: request no longer reports COMPLETED after "
             "the validated final snapshot\n";
    }
    if (poll_status == MTT_STATUS_OK &&
        snapshot.state == MTT_REQUEST_STATE_RUNNING) {
      static_cast<void>(cleanup_call(
          "request_cancel",
          [&](mtt_error_v1_t* cancel_error) {
            return native.api.request_cancel(native.request, cancel_error);
          },
          clean));
      const auto deadline = Clock::now() + std::chrono::seconds(5);
      std::uint64_t revision = snapshot.revision;
      while (Clock::now() < deadline) {
        clean =
            drain_for_cleanup(
                native, expect_completed_and_empty) &&
            clean;
        snapshot = {};
        snapshot.struct_size = sizeof(snapshot);
        snapshot.abi_version = MTT_ABI_VERSION_1;
        error = fresh_error();
        mtt_status_t wait_status = MTT_STATUS_INTERNAL_ERROR;
        try {
          wait_status = validate_call_status(
              "request_wait",
              native.api.request_wait(
                  native.request, revision, 100'000'000U, &snapshot,
                  &error),
              error);
        } catch (const std::exception& exception) {
          clean = false;
          std::cerr << "cleanup: " << exception.what() << '\n';
          break;
        }
        if (wait_status == MTT_STATUS_TIMEOUT) {
          continue;
        }
        if (wait_status != MTT_STATUS_OK) {
          clean = false;
          std::cerr << "cleanup: "
                    << native_error_message(
                           "request_wait", wait_status, error)
                    << '\n';
          break;
        }
        revision = snapshot.revision;
        if (snapshot.state != MTT_REQUEST_STATE_RUNNING) {
          break;
        }
      }
    } else if (poll_status != MTT_STATUS_OK) {
      clean = false;
      std::cerr << "cleanup: "
                << native_error_message(
                       "request_poll", poll_status, error)
                << '\n';
    }
    clean =
        drain_for_cleanup(native, expect_completed_and_empty) &&
        clean;
    const bool request_destroyed = cleanup_call(
        "request_destroy",
        [&](mtt_error_v1_t* destroy_error) {
          return native.api.request_destroy(native.request, destroy_error);
        },
        clean);
    if (!request_destroyed) {
      return false;
    }
    native.request = nullptr;
  }
  if (native.session != nullptr) {
    const bool session_destroyed = cleanup_call(
        "session_destroy",
        [&](mtt_error_v1_t* error) {
          return native.api.session_destroy(native.session, error);
        },
        clean);
    if (!session_destroyed) {
      return false;
    }
    native.session = nullptr;
  }
  if (native.model != nullptr) {
    const bool model_destroyed = cleanup_call(
        "model_destroy",
        [&](mtt_error_v1_t* error) {
          return native.api.model_destroy(native.model, error);
        },
        clean);
    if (!model_destroyed) {
      return false;
    }
    native.model = nullptr;
  }
  if (native.runtime != nullptr) {
    const bool runtime_destroyed = cleanup_call(
        "runtime_destroy",
        [&](mtt_error_v1_t* error) {
          return native.api.runtime_destroy(native.runtime, error);
        },
        clean);
    if (!runtime_destroyed) {
      return false;
    }
    native.runtime = nullptr;
  }
  if (native.library != nullptr) {
    if (dlclose(native.library) != 0) {
      clean = false;
      const char* const message = dlerror();
      std::cerr << "cleanup: dlclose failed: "
                << (message == nullptr ? "unknown dynamic-loader error"
                                       : message)
                << '\n';
    }
    native.library = nullptr;
  }
  return clean;
}

void validate_api_table(const mtt_api_v1_t& api) {
  if (api.struct_size != sizeof(api) ||
      api.abi_version != MTT_ABI_VERSION_1) {
    throw std::runtime_error(
        "mtt_get_api returned an invalid ABI v1 table header");
  }
  if (api.runtime_create == nullptr || api.runtime_destroy == nullptr ||
      api.model_load == nullptr || api.model_destroy == nullptr ||
      api.model_get_info == nullptr || api.session_create == nullptr ||
      api.session_destroy == nullptr || api.request_start == nullptr ||
      api.request_poll == nullptr || api.request_wait == nullptr ||
      api.request_cancel == nullptr || api.request_destroy == nullptr ||
      api.audio_acquire == nullptr || api.audio_release == nullptr) {
    throw std::runtime_error(
        "mtt_get_api returned an incomplete ABI v1 function table");
  }
}

void validate_model_info(const mtt_model_info_v1_t& info) {
  const bool tokenizer_digest_is_zero = std::all_of(
      std::begin(info.tokenizer_identity_sha256),
      std::end(info.tokenizer_identity_sha256),
      [](const std::uint8_t byte) { return byte == 0U; });
  const bool reserved_is_zero = std::all_of(
      std::begin(info.reserved), std::end(info.reserved),
      [](const std::uint64_t value) { return value == 0U; });
  if (info.struct_size != sizeof(info) ||
      info.abi_version != MTT_ABI_VERSION_1 ||
      info.tokenizer_vocabulary_size == 0U ||
      info.tokenizer_vocabulary_size >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      info.text_embedding_rows <= info.tokenizer_vocabulary_size ||
      info.text_embedding_rows >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      info.bos_token_id < info.tokenizer_vocabulary_size ||
      info.bos_token_id >= info.text_embedding_rows ||
      info.eos_token_id < info.tokenizer_vocabulary_size ||
      info.eos_token_id >= info.text_embedding_rows ||
      info.bos_token_id == info.eos_token_id ||
      info.japanese_global_pad_token_id >=
          info.tokenizer_vocabulary_size ||
      info.maximum_text_tokens == 0U ||
      info.maximum_text_tokens > MTT_MAX_TEXT_TOKENS ||
      info.maximum_audio_frames == 0U ||
      info.sample_rate_hz != kCanonicalSampleRateHz ||
      info.channels != kCanonicalChannels ||
      info.pcm_format != MTT_PCM_FORMAT_F32_MONO ||
      info.codec_frame_samples != kCodecFrameSamples ||
      info.initial_frames != 4U || info.steady_frames != 8U ||
      info.tail_min_frames != 1U || info.tail_max_frames != 8U ||
      info.reserved_0 != 0U || !reserved_is_zero ||
      tokenizer_digest_is_zero) {
    throw std::runtime_error(
        "model_get_info returned properties outside the canonical ABI v1 "
        "MagpieTTS contract");
  }
}

[[nodiscard]] bool snapshot_fields_equal(
    const mtt_request_snapshot_v1_t& left,
    const mtt_request_snapshot_v1_t& right) {
  const auto left_message = fixed_message(
      left.terminal_error_message, MTT_ERROR_MESSAGE_CAPACITY);
  const auto right_message = fixed_message(
      right.terminal_error_message, MTT_ERROR_MESSAGE_CAPACITY);
  return left.struct_size == right.struct_size &&
         left.abi_version == right.abi_version &&
         left.revision == right.revision &&
         left.state == right.state &&
         left.available_audio_leases == right.available_audio_leases &&
         left.generated_codec_frames == right.generated_codec_frames &&
         left.published_samples == right.published_samples &&
         left.committed_text_tokens == right.committed_text_tokens &&
         left.terminal_status == right.terminal_status &&
         left.terminal_error_stage == right.terminal_error_stage &&
         left_message == right_message &&
         std::equal(
             std::begin(left.reserved), std::end(left.reserved),
             std::begin(right.reserved));
}

void validate_snapshot(
    const std::string_view operation,
    const mtt_request_snapshot_v1_t& snapshot,
    const std::uint64_t request_token_count,
    const mtt_model_info_v1_t& model_info,
    const std::optional<std::uint64_t> minimum_revision_exclusive,
    StreamMetrics& metrics) {
  const bool reserved_is_zero = std::all_of(
      std::begin(snapshot.reserved), std::end(snapshot.reserved),
      [](const std::uint64_t value) { return value == 0U; });
  if (snapshot.struct_size != sizeof(snapshot) ||
      snapshot.abi_version != MTT_ABI_VERSION_1) {
    throw std::runtime_error(
        std::string(operation) +
        " returned an invalid request snapshot header");
  }
  if (!reserved_is_zero) {
    throw std::runtime_error(
        std::string(operation) +
        " returned nonzero request snapshot reserved fields");
  }
  if (snapshot.committed_text_tokens > request_token_count) {
    throw std::runtime_error(
        std::string(operation) +
        " returned text progress beyond the request token count");
  }
  if (minimum_revision_exclusive.has_value() &&
      snapshot.revision <= *minimum_revision_exclusive) {
    throw std::runtime_error(
        std::string(operation) +
        " returned a revision that did not advance");
  }
  if (snapshot.generated_codec_frames >
      model_info.maximum_audio_frames) {
    throw std::runtime_error(
        std::string(operation) +
        " returned generated frames beyond the authenticated limit");
  }
  if (snapshot.generated_codec_frames >
          std::numeric_limits<std::uint64_t>::max() /
              kCodecFrameSamples ||
      snapshot.published_samples !=
          snapshot.generated_codec_frames * kCodecFrameSamples) {
    throw std::runtime_error(
        std::string(operation) +
        " returned inconsistent codec-frame and PCM-sample counters");
  }

  const auto terminal_message = fixed_message(
      snapshot.terminal_error_message, MTT_ERROR_MESSAGE_CAPACITY);
  if (!terminal_message.has_value()) {
    throw std::runtime_error(
        std::string(operation) +
        " returned a non-NUL-terminated terminal diagnostic");
  }
  if (!is_declared_error_stage(snapshot.terminal_error_stage)) {
    throw std::runtime_error(
        std::string(operation) +
        " returned an unknown terminal error stage");
  }
  switch (snapshot.state) {
    case MTT_REQUEST_STATE_RUNNING:
    case MTT_REQUEST_STATE_COMPLETED:
      if (snapshot.terminal_status != MTT_STATUS_OK ||
          snapshot.terminal_error_stage != MTT_ERROR_STAGE_NONE ||
          !terminal_message->empty()) {
        throw std::runtime_error(
            std::string(operation) +
            " returned a success state with a terminal diagnostic");
      }
      break;
    case MTT_REQUEST_STATE_CANCELLED:
      if (snapshot.terminal_status != MTT_STATUS_CANCELLED ||
          snapshot.terminal_error_stage != MTT_ERROR_STAGE_REQUEST ||
          !terminal_message->empty()) {
        throw std::runtime_error(
            std::string(operation) +
            " returned an invalid CANCELLED terminal diagnostic");
      }
      break;
    case MTT_REQUEST_STATE_FAILED:
      if (!is_terminal_failure_status(snapshot.terminal_status) ||
          snapshot.terminal_error_stage == MTT_ERROR_STAGE_NONE ||
          terminal_message->empty()) {
        throw std::runtime_error(
            std::string(operation) +
            " returned an invalid FAILED terminal diagnostic");
      }
      break;
    default:
      throw std::runtime_error(
          std::string(operation) +
          " returned an unknown ABI v1 request state");
  }

  if (metrics.last_snapshot.has_value()) {
    const auto& previous = *metrics.last_snapshot;
    if (snapshot.revision < previous.revision ||
        snapshot.generated_codec_frames <
            previous.generated_codec_frames ||
        snapshot.published_samples < previous.published_samples ||
        snapshot.committed_text_tokens <
            previous.committed_text_tokens) {
      throw std::runtime_error(
          std::string(operation) +
          " returned non-monotonic request counters");
    }
    if (snapshot.revision == previous.revision &&
        !snapshot_fields_equal(snapshot, previous)) {
      throw std::runtime_error(
          std::string(operation) +
          " changed a request snapshot without advancing its revision");
    }
    if (previous.state != MTT_REQUEST_STATE_RUNNING &&
        snapshot.state != previous.state) {
      throw std::runtime_error(
          std::string(operation) +
          " changed a terminal request state");
    }
    if (previous.state == MTT_REQUEST_STATE_FAILED &&
        (snapshot.terminal_status != previous.terminal_status ||
         snapshot.terminal_error_stage !=
             previous.terminal_error_stage ||
         fixed_message(
             snapshot.terminal_error_message,
             MTT_ERROR_MESSAGE_CAPACITY) !=
             fixed_message(
                 previous.terminal_error_message,
                 MTT_ERROR_MESSAGE_CAPACITY))) {
      throw std::runtime_error(
          std::string(operation) +
          " changed a retained FAILED diagnostic");
    }
  }
  metrics.last_snapshot = snapshot;
}

void validate_and_release_lease(
    NativeObjects& native, const mtt_audio_lease_v1_t& lease,
    const mtt_model_info_v1_t& model_info,
    const std::uint64_t request_token_count, StreamMetrics& metrics) {
  std::string validation_error;
  std::uint64_t lease_end = 0U;
  std::uint64_t codec_frames = 0U;
  std::uint64_t next_alignment_sample = metrics.last_alignment_sample;
  std::uint64_t next_alignment_token = metrics.last_alignment_token;
  const bool first = (lease.flags & MTT_AUDIO_FLAG_FIRST) != 0U;
  const bool final = (lease.flags & MTT_AUDIO_FLAG_FINAL) != 0U;
  const bool alignment_valid =
      (lease.flags & MTT_AUDIO_FLAG_ALIGNMENT_VALID) != 0U;
  const bool terminal_control_marker =
      lease.sample_count == 0U;
  const bool reserved_is_zero = std::all_of(
      std::begin(lease.reserved), std::end(lease.reserved),
      [](const std::uint64_t value) { return value == 0U; });

  if (lease.struct_size != sizeof(lease) ||
      lease.abi_version != MTT_ABI_VERSION_1) {
    validation_error = "audio lease header does not match ABI v1";
  } else if (!reserved_is_zero) {
    validation_error = "audio lease reserved fields are not zero";
  } else if ((lease.flags & ~kKnownAudioFlags) != 0U) {
    validation_error = "audio lease contains flags unknown to ABI v1";
  } else if (lease.lease_id == 0U) {
    validation_error = "audio lease identifier is zero";
  } else if (
      metrics.last_lease_id.has_value() &&
      lease.lease_id <= *metrics.last_lease_id) {
    validation_error =
        "audio lease identifier did not strictly increase";
  } else if (
      terminal_control_marker &&
      lease.samples != nullptr) {
    validation_error =
        "zero-sample FINAL control marker has a PCM pointer";
  } else if (
      !terminal_control_marker &&
      lease.samples == nullptr) {
    validation_error = "decoded audio lease PCM pointer is null";
  } else if (
      !terminal_control_marker &&
      reinterpret_cast<std::uintptr_t>(lease.samples) %
              alignof(float) !=
          0U) {
    validation_error = "audio lease PCM pointer is not aligned for float";
  } else if (
      lease.sample_count % kCodecFrameSamples != 0U) {
    validation_error =
        "audio lease does not contain complete 1024-sample codec frames";
  } else if (
      (codec_frames = lease.sample_count / kCodecFrameSamples) >
      model_info.tail_max_frames) {
    validation_error = "audio lease codec-frame count exceeds eight";
  } else if (
      lease.sequence != metrics.expected_sequence ||
      lease.first_sample_index != metrics.expected_sample_index) {
    validation_error =
        "audio lease sequence or sample offset is not contiguous";
  } else if (
      lease.first_sample_index >
      std::numeric_limits<std::uint64_t>::max() - lease.sample_count) {
    validation_error = "audio lease sample range overflowed";
  } else if (
      lease.sample_rate_hz != kCanonicalSampleRateHz ||
      lease.channels != kCanonicalChannels ||
      lease.format != MTT_PCM_FORMAT_F32_MONO ||
      lease.sample_rate_hz != model_info.sample_rate_hz ||
      lease.channels != model_info.channels ||
      lease.format != model_info.pcm_format) {
    validation_error =
        "audio lease is not canonical 22050 Hz mono F32 PCM";
  } else if (first != (metrics.expected_sequence == 0U)) {
    validation_error = "FIRST flag does not identify sequence zero";
  } else if (metrics.final_seen) {
    validation_error = "audio arrived after FINAL";
  } else if (
      terminal_control_marker &&
      (!final || first)) {
    validation_error =
        "zero samples require a non-FIRST FINAL control marker";
  } else if (
      first && codec_frames != model_info.initial_frames) {
    validation_error =
        "the first audio lease is not the canonical four-frame chunk";
  } else if (
      !first && !final &&
      codec_frames != model_info.steady_frames) {
    validation_error =
        "a non-terminal audio lease is not the canonical eight-frame chunk";
  } else if (
      final && !terminal_control_marker &&
      (codec_frames < model_info.tail_min_frames ||
       codec_frames > model_info.tail_max_frames)) {
    validation_error =
        "the terminal audio lease is outside the canonical 1-8 frame tail";
  } else if (
      metrics.sample_count / kCodecFrameSamples >
          model_info.maximum_audio_frames ||
      codec_frames >
          model_info.maximum_audio_frames -
              metrics.sample_count / kCodecFrameSamples) {
    validation_error =
        "audio stream exceeds the authenticated maximum frame count";
  } else {
    lease_end = lease.first_sample_index + lease.sample_count;
    for (std::uint64_t index = 0; index < lease.sample_count; ++index) {
      if (!std::isfinite(lease.samples[index])) {
        validation_error = "audio lease contains non-finite PCM";
        break;
      }
    }
  }

  if (validation_error.empty()) {
    if ((lease.alignment_event_count == 0U &&
         lease.alignment_events != nullptr) ||
        (lease.alignment_event_count != 0U &&
         lease.alignment_events == nullptr)) {
      validation_error =
          "audio alignment pointer/count relation is invalid";
    } else if (
        !alignment_valid &&
        (lease.alignment_event_count != 0U ||
         lease.committed_text_tokens != 0U)) {
      validation_error =
          "audio without alignment carries committed text progress";
    } else if (
        alignment_valid &&
        (lease.committed_text_tokens < metrics.last_alignment_token ||
         lease.committed_text_tokens > request_token_count)) {
      validation_error =
          "lease text progress is not monotonic within the request";
    } else if (
        alignment_valid && lease.alignment_event_count == 0U &&
        lease.committed_text_tokens != metrics.last_alignment_token) {
      validation_error =
          "lease text progress advanced without an alignment event";
    } else if (
        lease.alignment_event_count >
        (lease.sample_count + kDecoderStepSamples - 1U) /
            kDecoderStepSamples) {
      validation_error =
          "audio lease has more than one alignment event per decoder step";
    } else if (
        lease.alignment_event_count != 0U &&
        reinterpret_cast<std::uintptr_t>(lease.alignment_events) %
                alignof(mtt_alignment_event_v1_t) !=
            0U) {
      validation_error =
          "audio alignment event pointer is not correctly aligned";
    } else {
      for (std::uint64_t index = 0;
           index < lease.alignment_event_count; ++index) {
        const auto& event = lease.alignment_events[index];
        const bool event_reserved_is_zero = std::all_of(
            std::begin(event.reserved), std::end(event.reserved),
            [](const std::uint64_t value) { return value == 0U; });
        const bool full_decoder_step =
            event.sample_index % kDecoderStepSamples == 0U;
        const bool terminal_single_frame_boundary =
            final && event.sample_index == lease_end &&
            event.sample_index % kCodecFrameSamples == 0U;
        if (event.struct_size != sizeof(event) ||
            event.abi_version != MTT_ABI_VERSION_1 ||
            !event_reserved_is_zero ||
            event.sample_index <= next_alignment_sample ||
            event.sample_index <= lease.first_sample_index ||
            event.sample_index > lease_end ||
            (!full_decoder_step &&
             !terminal_single_frame_boundary) ||
            event.committed_text_tokens <= next_alignment_token ||
            event.committed_text_tokens > request_token_count) {
          validation_error =
              "audio alignment event violates its ABI v1 sample/token "
              "boundary";
          break;
        }
        next_alignment_sample = event.sample_index;
        next_alignment_token = event.committed_text_tokens;
      }
      if (validation_error.empty() && alignment_valid &&
          lease.committed_text_tokens != next_alignment_token) {
        validation_error =
            "lease text progress does not equal its last alignment event";
      }
    }
  }

  auto release_error = fresh_error();
  const auto release_status = native.api.audio_release(
      native.request, lease.lease_id, &release_error);
  require_ok("audio_release", release_status, release_error);
  if (!validation_error.empty()) {
    throw std::runtime_error(validation_error);
  }

  if (lease.sample_count != 0U &&
      !metrics.first_audio_at.has_value()) {
    metrics.first_audio_at = Clock::now();
  }
  metrics.first_seen = metrics.first_seen || first;
  metrics.final_seen = final;
  metrics.terminal_control_marker_seen =
      metrics.terminal_control_marker_seen ||
      terminal_control_marker;
  metrics.chunk_count += 1U;
  metrics.sample_count += lease.sample_count;
  metrics.last_lease_id = lease.lease_id;
  metrics.last_alignment_sample = next_alignment_sample;
  metrics.last_alignment_token = next_alignment_token;
  metrics.expected_sequence += 1U;
  metrics.expected_sample_index = lease_end;
}

void drain_audio(
    NativeObjects& native, const mtt_model_info_v1_t& model_info,
    const std::uint64_t request_token_count, StreamMetrics& metrics) {
  for (;;) {
    mtt_audio_lease_v1_t lease{};
    lease.struct_size = sizeof(lease);
    lease.abi_version = MTT_ABI_VERSION_1;
    auto error = fresh_error();
    const auto status = validate_call_status(
        "audio_acquire",
        native.api.audio_acquire(native.request, &lease, &error), error);
    if (status == MTT_STATUS_WOULD_BLOCK) {
      return;
    }
    require_ok("audio_acquire", status, error);
    validate_and_release_lease(
        native, lease, model_info, request_token_count, metrics);
  }
}

[[nodiscard]] double milliseconds(const Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

[[nodiscard]] std::string sha256_hex(
    const std::uint8_t (&digest)[MTT_SHA256_BYTES]) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result(MTT_SHA256_BYTES * 2U, '0');
  for (std::size_t index = 0; index < MTT_SHA256_BYTES; ++index) {
    result[index * 2U] = kHexDigits[digest[index] >> 4U];
    result[index * 2U + 1U] = kHexDigits[digest[index] & 0x0fU];
  }
  return result;
}

void write_all(const int descriptor, const std::string_view bytes) {
  std::size_t written = 0U;
  while (written < bytes.size()) {
    const auto result = ::write(
        descriptor, bytes.data() + written, bytes.size() - written);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::system_error(
          errno, std::generic_category(), "write receipt");
    }
    if (result == 0) {
      throw std::runtime_error(
          "receipt write made no forward progress");
    }
    written += static_cast<std::size_t>(result);
  }
}

void write_new_receipt(
    const std::filesystem::path& target,
    const nlohmann::json& receipt) {
  require_new_absolute_receipt_path(target);
  const auto parent = target.parent_path();
  const auto stem = "." + target.filename().string() + ".tmp.";
  std::filesystem::path temporary;
  int descriptor = -1;
  for (std::uint32_t attempt = 0U; attempt < 32U; ++attempt) {
    temporary =
        parent /
        (stem + std::to_string(static_cast<std::uint64_t>(::getpid())) +
         "." + std::to_string(attempt));
    descriptor = ::open(
        temporary.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (descriptor >= 0) {
      break;
    }
    if (errno != EEXIST) {
      throw std::system_error(
          errno, std::generic_category(), "create receipt temporary");
    }
  }
  if (descriptor < 0) {
    throw std::runtime_error(
        "unable to allocate an exclusive receipt temporary");
  }

  bool target_linked = false;
  try {
    const auto bytes = receipt.dump(2) + "\n";
    write_all(descriptor, bytes);
    if (::fsync(descriptor) != 0) {
      throw std::system_error(
          errno, std::generic_category(), "fsync receipt temporary");
    }
    if (::close(descriptor) != 0) {
      descriptor = -1;
      throw std::system_error(
          errno, std::generic_category(), "close receipt temporary");
    }
    descriptor = -1;
    if (::link(temporary.c_str(), target.c_str()) != 0) {
      throw std::system_error(
          errno, std::generic_category(),
          "publish immutable receipt");
    }
    target_linked = true;
    if (::unlink(temporary.c_str()) != 0) {
      throw std::system_error(
          errno, std::generic_category(),
          "unlink receipt temporary");
    }
    temporary.clear();
    const int directory = ::open(
        parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory < 0) {
      throw std::system_error(
          errno, std::generic_category(), "open receipt parent");
    }
    const int sync_result = ::fsync(directory);
    const int sync_errno = errno;
    const int close_result = ::close(directory);
    const int close_errno = errno;
    if (sync_result != 0) {
      throw std::system_error(
          sync_errno, std::generic_category(),
          "fsync receipt parent");
    }
    if (close_result != 0) {
      throw std::system_error(
          close_errno, std::generic_category(),
          "close receipt parent");
    }
  } catch (...) {
    const int saved_errno = errno;
    if (descriptor >= 0) {
      static_cast<void>(::close(descriptor));
    }
    if (!temporary.empty()) {
      static_cast<void>(::unlink(temporary.c_str()));
    }
    if (target_linked) {
      static_cast<void>(::unlink(target.c_str()));
    }
    errno = saved_errno;
    throw;
  }
}

[[nodiscard]] nlohmann::json build_receipt(
    const Arguments& arguments, const GoldenInput& input,
    const mtt_model_info_v1_t& model_info,
    const StreamMetrics& metrics,
    const Clock::duration startup_gate,
    const Clock::duration request_ttfa,
    const Clock::duration request_total) {
  if (!metrics.final_snapshot.has_value()) {
    throw std::logic_error(
        "cannot build a receipt without a validated final snapshot");
  }
  const auto& final_snapshot = *metrics.final_snapshot;
  const auto audio_seconds =
      static_cast<double>(metrics.sample_count) /
      static_cast<double>(model_info.sample_rate_hz);
  const auto request_seconds =
      std::chrono::duration<double>(request_total).count();
  const auto completed_at_unix_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  const auto manifest_path =
      arguments.bundle / "runtime-bundle-manifest.json";
  return {
      {"schema_version", "magpie-tts-rt.runtime-smoke.v1"},
      {"status", "accepted"},
      {"completed_at_unix_ms", completed_at_unix_ms},
      {"abi_version", MTT_ABI_VERSION_1},
      {"inputs",
       {
           {"library_name",
            public_artifact_name(arguments.library, "runtime library")},
           {"library_size_bytes",
            public_regular_file_size(
                arguments.library, "runtime library")},
           {"bundle_name",
            public_artifact_name(arguments.bundle, "runtime bundle")},
           {"bundle_manifest_name", "runtime-bundle-manifest.json"},
           {"bundle_manifest_sha256",
            arguments.manifest_sha256_hex},
           {"bundle_manifest_size_bytes",
            public_regular_file_size(
                manifest_path, "runtime bundle manifest")},
           {"cuda_device_index", arguments.cuda_device_index},
           {"request_timeout_ms", arguments.request_timeout.count()},
           {"prepared_token_count", input.token_ids.size()},
           {"random_seed", input.seed},
       }},
      {"model",
       {
           {"tokenizer_vocabulary_size",
            model_info.tokenizer_vocabulary_size},
           {"text_embedding_rows", model_info.text_embedding_rows},
           {"bos_token_id", model_info.bos_token_id},
           {"eos_token_id", model_info.eos_token_id},
           {"japanese_global_pad_token_id",
            model_info.japanese_global_pad_token_id},
           {"maximum_text_tokens", model_info.maximum_text_tokens},
           {"maximum_audio_frames", model_info.maximum_audio_frames},
           {"tokenizer_identity_sha256",
            sha256_hex(model_info.tokenizer_identity_sha256)},
       }},
      {"stream_contract",
       {
           {"sample_rate_hz", model_info.sample_rate_hz},
           {"channels", model_info.channels},
           {"pcm_format", "f32_mono"},
           {"codec_frame_samples", model_info.codec_frame_samples},
           {"initial_frames", model_info.initial_frames},
           {"steady_frames", model_info.steady_frames},
           {"tail_min_frames", model_info.tail_min_frames},
           {"tail_max_frames", model_info.tail_max_frames},
       }},
      {"result",
       {
           {"chunk_count", metrics.chunk_count},
           {"sample_count", metrics.sample_count},
           {"terminal_control_marker_seen",
            metrics.terminal_control_marker_seen},
           {"audio_seconds", audio_seconds},
           {"last_alignment_sample", metrics.last_alignment_sample},
           {"committed_text_tokens",
            final_snapshot.committed_text_tokens},
           {"final_revision", final_snapshot.revision},
           {"generated_codec_frames",
            final_snapshot.generated_codec_frames},
       }},
      {"latency",
       {
           {"startup_gate_ms", milliseconds(startup_gate)},
           {"request_ttfa_ms", milliseconds(request_ttfa)},
           {"request_total_ms", milliseconds(request_total)},
           {"rtf", request_seconds / audio_seconds},
       }},
      {"verification",
       {
           {"startup_golden", "passed"},
           {"stream_first_to_final", "passed"},
           {"snapshot_contract", "passed"},
           {"native_cleanup", "passed"},
       }},
  };
}

void load_native(
    const Arguments& arguments, NativeObjects& native,
    mtt_model_info_v1_t& model_info) {
  native.library =
      dlopen(arguments.library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (native.library == nullptr) {
    const char* const message = dlerror();
    throw std::runtime_error(
        "dlopen failed: " +
        std::string(
            message == nullptr ? "unknown dynamic-loader error" : message));
  }
  dlerror();
  void* const symbol = dlsym(native.library, "mtt_get_api");
  if (const char* const message = dlerror(); message != nullptr) {
    throw std::runtime_error("dlsym(mtt_get_api) failed: " +
                             std::string(message));
  }
  if (symbol == nullptr) {
    throw std::runtime_error("dlsym(mtt_get_api) returned null");
  }
  mtt_status_t (*get_api)(std::uint32_t, mtt_api_v1_t*) = nullptr;
  static_assert(sizeof(get_api) == sizeof(symbol));
  std::memcpy(&get_api, &symbol, sizeof(get_api));
  native.api = {};
  native.api.struct_size = sizeof(native.api);
  native.api.abi_version = MTT_ABI_VERSION_1;
  const auto negotiation_status =
      get_api(MTT_ABI_VERSION_1, &native.api);
  if (negotiation_status != MTT_STATUS_OK) {
    throw std::runtime_error(
        "mtt_get_api failed: status=" +
        std::to_string(negotiation_status));
  }
  validate_api_table(native.api);

  mtt_runtime_desc_v1_t runtime_desc{};
  runtime_desc.struct_size = sizeof(runtime_desc);
  runtime_desc.abi_version = MTT_ABI_VERSION_1;
  runtime_desc.cuda_device_index = arguments.cuda_device_index;
  auto error = fresh_error();
  require_ok(
      "runtime_create",
      native.api.runtime_create(&runtime_desc, &native.runtime, &error),
      error);
  if (native.runtime == nullptr) {
    throw std::runtime_error(
        "runtime_create succeeded without returning a runtime handle");
  }

  const auto bundle_string = arguments.bundle.string();
  if (bundle_string.empty() ||
      bundle_string.size() > MTT_MAX_BUNDLE_PATH_BYTES) {
    throw std::runtime_error(
        "bundle path is outside the C ABI v1 byte limit");
  }
  mtt_model_desc_v1_t model_desc{};
  model_desc.struct_size = sizeof(model_desc);
  model_desc.abi_version = MTT_ABI_VERSION_1;
  model_desc.bundle_path = bundle_string.data();
  model_desc.bundle_path_length = bundle_string.size();
  std::copy(
      arguments.manifest_sha256.begin(),
      arguments.manifest_sha256.end(),
      model_desc.expected_manifest_sha256);
  error = fresh_error();
  require_ok(
      "model_load",
      native.api.model_load(
          native.runtime, &model_desc, &native.model, &error),
      error);
  if (native.model == nullptr) {
    throw std::runtime_error(
        "model_load succeeded without returning a model handle");
  }

  model_info = {};
  model_info.struct_size = sizeof(model_info);
  model_info.abi_version = MTT_ABI_VERSION_1;
  error = fresh_error();
  require_ok(
      "model_get_info",
      native.api.model_get_info(native.model, &model_info, &error), error);
  validate_model_info(model_info);

  mtt_session_desc_v1_t session_desc{};
  session_desc.struct_size = sizeof(session_desc);
  session_desc.abi_version = MTT_ABI_VERSION_1;
  error = fresh_error();
  require_ok(
      "session_create",
      native.api.session_create(
          native.model, &session_desc, &native.session, &error),
      error);
  if (native.session == nullptr) {
    throw std::runtime_error(
        "session_create succeeded without returning a session handle");
  }
}

void run_streaming_request(
    const Arguments& arguments, const GoldenInput& input,
    NativeObjects& native, const mtt_model_info_v1_t& model_info,
    const Clock::time_point request_started_at, StreamMetrics& metrics) {
  if (input.token_ids.size() > model_info.maximum_text_tokens) {
    throw std::runtime_error(
        "golden token count exceeds authenticated model limit");
  }
  if (input.token_ids.back() !=
      static_cast<std::int64_t>(model_info.eos_token_id)) {
    throw std::runtime_error(
        "golden input does not end in authenticated EOS");
  }
  for (std::size_t index = 0;
       index + 1 < input.token_ids.size();
       ++index) {
    const std::int64_t token = input.token_ids[index];
    if (token < 0 ||
        token >= static_cast<std::int64_t>(
                     model_info.tokenizer_vocabulary_size)) {
      throw std::runtime_error(
          "golden non-final token is not a normal tokenizer row");
    }
  }
  mtt_request_desc_v1_t request_desc{};
  request_desc.struct_size = sizeof(request_desc);
  request_desc.abi_version = MTT_ABI_VERSION_1;
  request_desc.text_token_ids = input.token_ids.data();
  request_desc.text_token_count = input.token_ids.size();
  request_desc.random_seed = input.seed;
  auto error = fresh_error();
  require_ok(
      "request_start",
      native.api.request_start(
          native.session, &request_desc, &native.request, &error),
      error);
  if (native.request == nullptr) {
    throw std::runtime_error(
        "request_start succeeded without returning a request handle");
  }

  std::uint64_t revision = 0;
  const auto deadline = request_started_at + arguments.request_timeout;
  for (;;) {
    drain_audio(
        native, model_info, request_desc.text_token_count, metrics);
    if (Clock::now() >= deadline) {
      throw std::runtime_error("streaming request exceeded its deadline");
    }
    mtt_request_snapshot_v1_t snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    snapshot.abi_version = MTT_ABI_VERSION_1;
    error = fresh_error();
    const auto status = validate_call_status(
        "request_wait",
        native.api.request_wait(
            native.request, revision, 100'000'000U, &snapshot, &error),
        error);
    if (status == MTT_STATUS_TIMEOUT) {
      continue;
    }
    require_ok("request_wait", status, error);
    validate_snapshot(
        "request_wait", snapshot, request_desc.text_token_count,
        model_info, revision, metrics);
    revision = snapshot.revision;
    drain_audio(
        native, model_info, request_desc.text_token_count, metrics);
    if (snapshot.state == MTT_REQUEST_STATE_RUNNING) {
      continue;
    }
    if (snapshot.state != MTT_REQUEST_STATE_COMPLETED) {
      throw std::runtime_error(
          "streaming request reached a non-success terminal state");
    }
    break;
  }
  drain_audio(
      native, model_info, request_desc.text_token_count, metrics);

  mtt_request_snapshot_v1_t final_snapshot{};
  final_snapshot.struct_size = sizeof(final_snapshot);
  final_snapshot.abi_version = MTT_ABI_VERSION_1;
  error = fresh_error();
  require_ok(
      "request_poll",
      native.api.request_poll(native.request, &final_snapshot, &error),
      error);
  validate_snapshot(
      "request_poll", final_snapshot, request_desc.text_token_count,
      model_info, std::nullopt, metrics);
  if (final_snapshot.state != MTT_REQUEST_STATE_COMPLETED ||
      final_snapshot.available_audio_leases != 0U ||
      final_snapshot.published_samples != metrics.sample_count ||
      final_snapshot.generated_codec_frames !=
          metrics.sample_count / kCodecFrameSamples ||
      final_snapshot.committed_text_tokens !=
          metrics.last_alignment_token) {
    throw std::runtime_error(
        "final request snapshot does not match the drained stream");
  }
  if (!metrics.first_seen || !metrics.final_seen ||
      metrics.chunk_count == 0U || metrics.sample_count == 0U) {
    throw std::runtime_error(
        "stream did not contain a complete FIRST-to-FINAL PCM sequence");
  }
  metrics.final_snapshot = final_snapshot;
}

[[nodiscard]] int run(const int argc, char** argv) {
  const auto arguments = parse_arguments(argc, argv);
  if (arguments.help) {
    std::cout << usage() << '\n';
    return 0;
  }
  const auto input = load_golden_input(arguments.bundle);
  NativeObjects native;
  try {
    const auto load_started_at = Clock::now();
    mtt_model_info_v1_t model_info{};
    load_native(arguments, native, model_info);
    const auto ready_at = Clock::now();
    StreamMetrics metrics;
    const auto request_started_at = Clock::now();
    run_streaming_request(
        arguments, input, native, model_info, request_started_at, metrics);
    const auto completed_at = Clock::now();
    const auto first_audio_at = metrics.first_audio_at.value();
    const auto audio_seconds =
        static_cast<double>(metrics.sample_count) /
        static_cast<double>(model_info.sample_rate_hz);
    const auto request_seconds =
        std::chrono::duration<double>(
            completed_at - request_started_at)
            .count();
    const auto clean = cleanup_native(native, true);
    if (!clean) {
      throw std::runtime_error(
          "stream succeeded but native cleanup failed");
    }
    const auto receipt = build_receipt(
        arguments, input, model_info, metrics,
        ready_at - load_started_at,
        first_audio_at - request_started_at,
        completed_at - request_started_at);
    if (arguments.receipt_json.has_value()) {
      write_new_receipt(*arguments.receipt_json, receipt);
    }
    std::cout << "status=accepted\n"
              << "bundle_manifest_sha256="
              << arguments.manifest_sha256_hex << '\n'
              << "cuda_device_index="
              << arguments.cuda_device_index << '\n'
              << "startup_gate_ms="
              << milliseconds(ready_at - load_started_at) << '\n'
              << "request_ttfa_ms="
              << milliseconds(first_audio_at - request_started_at) << '\n'
              << "request_total_ms="
              << milliseconds(completed_at - request_started_at) << '\n'
              << "audio_samples=" << metrics.sample_count << '\n'
              << "audio_seconds=" << audio_seconds << '\n'
              << "chunks=" << metrics.chunk_count << '\n'
              << "rtf=" << request_seconds / audio_seconds << '\n';
    if (arguments.receipt_json.has_value()) {
      std::cout << "receipt_json="
                << arguments.receipt_json->string() << '\n';
    }
    return 0;
  } catch (...) {
    const auto clean = cleanup_native(native);
    if (!clean) {
      std::cerr << "native cleanup also failed\n";
    }
    throw;
  }
}

}  // namespace

#ifndef MTT_RUNTIME_SMOKE_TESTING
int main(const int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "runtime smoke failed: " << error.what() << '\n';
    return 1;
  }
}
#endif

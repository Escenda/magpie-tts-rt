#include "manifest/manifest.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <unistd.h>

#include "validation/character_validation.hpp"

namespace magpie_tts_rt {
namespace {

using Json = nlohmann::json;

class ManifestFileDescriptor final {
 public:
  explicit ManifestFileDescriptor(const int value) : value_(value) {}
  ManifestFileDescriptor(const ManifestFileDescriptor&) = delete;
  ManifestFileDescriptor& operator=(const ManifestFileDescriptor&) = delete;

  ~ManifestFileDescriptor() {
    if (value_ >= 0) {
      static_cast<void>(::close(value_));
    }
  }

  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_;
};

[[nodiscard]] std::string build_error_message(
    const ManifestStage stage,
    const ManifestErrorCode code,
    const std::string_view json_pointer,
    const std::string_view detail) {
  std::ostringstream message;
  message << "runtime bundle manifest error"
          << " [stage=" << to_string(stage) << ", code=" << to_string(code)
          << ", path=" << (json_pointer.empty() ? "/" : json_pointer) << "]: "
          << detail;
  return message.str();
}

[[nodiscard]] std::string escape_json_pointer_token(const std::string_view token) {
  std::string escaped;
  escaped.reserve(token.size());
  for (const char value : token) {
    if (value == '~') {
      escaped += "~0";
    } else if (value == '/') {
      escaped += "~1";
    } else {
      escaped.push_back(value);
    }
  }
  return escaped;
}

[[nodiscard]] std::string child_path(
    const std::string_view parent,
    const std::string_view child) {
  std::string result(parent);
  result.push_back('/');
  result += escape_json_pointer_token(child);
  return result;
}

[[nodiscard]] std::string index_path(
    const std::string_view parent,
    const std::size_t index) {
  return child_path(parent, std::to_string(index));
}

[[noreturn]] void fail(
    const ManifestStage stage,
    const ManifestErrorCode code,
    const std::string& path,
    const std::string& detail) {
  throw ManifestError(stage, code, path, detail);
}

void require_exact_keys(
    const Json& value,
    const ManifestStage stage,
    const std::string& path,
    const std::initializer_list<std::string_view> required_keys) {
  if (!value.is_object()) {
    fail(stage, ManifestErrorCode::type_mismatch, path, "expected an object");
  }

  std::set<std::string, std::less<>> expected;
  for (const std::string_view key : required_keys) {
    expected.emplace(key);
  }

  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (!expected.contains(key)) {
      fail(
          stage,
          ManifestErrorCode::unknown_field,
          child_path(path, key),
          "field is not part of schema version 1");
    }
  }

  for (const std::string_view key : required_keys) {
    if (!value.contains(key)) {
      fail(
          stage,
          ManifestErrorCode::missing_field,
          child_path(path, key),
          "required field is missing");
    }
  }
}

[[nodiscard]] const Json& member(
    const Json& object,
    const std::string_view key) {
  return object.at(std::string(key));
}

[[nodiscard]] std::string parse_nonempty_string(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  if (!value.is_string()) {
    fail(stage, ManifestErrorCode::type_mismatch, path, "expected a string");
  }
  const std::string parsed = value.get<std::string>();
  if (parsed.empty()) {
    fail(stage, ManifestErrorCode::invalid_value, path, "string must not be empty");
  }
  return parsed;
}

[[nodiscard]] std::string parse_identifier(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  if (!character_validation::is_identifier(parsed)) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "expected 1..128 characters matching [A-Za-z0-9][A-Za-z0-9._-]*");
  }
  return parsed;
}

[[nodiscard]] std::string parse_sha256(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  if (!character_validation::is_lowercase_sha256(parsed)) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "SHA-256 must be exactly 64 lowercase hexadecimal characters");
  }
  return parsed;
}

[[nodiscard]] std::string parse_git_sha1(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  const bool valid =
      parsed.size() == 40U &&
      std::all_of(parsed.begin(), parsed.end(), [](const char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
      });
  if (!valid) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "Git revision must be exactly 40 lowercase hexadecimal characters");
  }
  return parsed;
}

[[nodiscard]] std::string parse_numeric_version(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  if (!character_validation::is_dotted_numeric_version(parsed)) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "expected a dotted numeric version");
  }
  return parsed;
}

[[nodiscard]] std::string parse_rfc3339_utc(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  if (!character_validation::is_rfc3339_utc_lexeme(parsed)) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "expected an RFC 3339 UTC timestamp ending in Z");
  }
  const auto digit = [&parsed](const std::size_t index) {
    return static_cast<unsigned int>(parsed.at(index) - '0');
  };
  const unsigned int year_value =
      digit(0) * 1000U + digit(1) * 100U + digit(2) * 10U + digit(3);
  const unsigned int month_value = digit(5) * 10U + digit(6);
  const unsigned int day_value = digit(8) * 10U + digit(9);
  const std::chrono::year_month_day calendar_date{
      std::chrono::year(static_cast<int>(year_value)),
      std::chrono::month(month_value),
      std::chrono::day(day_value)};
  if (year_value == 0 || !calendar_date.ok()) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "timestamp contains an invalid Gregorian calendar date");
  }
  return parsed;
}

[[nodiscard]] std::filesystem::path parse_relative_path(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  for (const char raw_byte : parsed) {
    const auto byte = static_cast<unsigned char>(raw_byte);
    if (byte <= 0x1FU || byte == 0x7FU) {
      fail(
          stage,
          ManifestErrorCode::invalid_value,
          path,
          "bundle path must not contain ASCII control bytes");
    }
  }
  if (parsed.find('\\') != std::string::npos) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "bundle paths must use forward slashes");
  }

  const std::filesystem::path artifact_path(parsed);
  if (artifact_path.is_absolute() || artifact_path.has_root_name() ||
      artifact_path.has_root_directory()) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "bundle path must be relative");
  }
  for (const auto& component : artifact_path) {
    if (component == "." || component == "..") {
      fail(
          stage,
          ManifestErrorCode::invalid_value,
          path,
          "bundle path must not contain '.' or '..' components");
    }
  }
  return artifact_path;
}

template <typename Integer>
[[nodiscard]] Integer parse_unsigned_integer(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  static_assert(std::is_unsigned_v<Integer>);
  if (!value.is_number_unsigned() && !value.is_number_integer()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        path,
        "expected a non-negative integer");
  }

  if (value.is_number_integer() && !value.is_number_unsigned()) {
    const auto signed_value = value.get<std::int64_t>();
    if (signed_value < 0) {
      fail(
          stage,
          ManifestErrorCode::invalid_value,
          path,
          "integer must be non-negative");
    }
    if (static_cast<std::uint64_t>(signed_value) >
        static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
      fail(stage, ManifestErrorCode::invalid_value, path, "integer is out of range");
    }
    return static_cast<Integer>(signed_value);
  }

  const auto unsigned_value = value.get<std::uint64_t>();
  if (unsigned_value >
      static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
    fail(stage, ManifestErrorCode::invalid_value, path, "integer is out of range");
  }
  return static_cast<Integer>(unsigned_value);
}

template <typename Integer>
[[nodiscard]] Integer parse_positive_integer(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const Integer parsed = parse_unsigned_integer<Integer>(value, stage, path);
  if (parsed == 0) {
    fail(stage, ManifestErrorCode::invalid_value, path, "integer must be positive");
  }
  return parsed;
}

[[nodiscard]] bool parse_boolean(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  if (!value.is_boolean()) {
    fail(stage, ManifestErrorCode::type_mismatch, path, "expected a boolean");
  }
  return value.get<bool>();
}

[[nodiscard]] double parse_positive_number(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  if (!value.is_number()) {
    fail(stage, ManifestErrorCode::type_mismatch, path, "expected a number");
  }
  const double parsed = value.get<double>();
  if (!std::isfinite(parsed) || parsed <= 0.0) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "number must be finite and positive");
  }
  return parsed;
}

[[nodiscard]] Endianness parse_endianness(
    const Json& value,
    const std::string& path) {
  const std::string parsed =
      parse_nonempty_string(value, ManifestStage::runtime_fingerprint, path);
  if (parsed == "little") {
    return Endianness::little;
  }
  if (parsed == "big") {
    return Endianness::big;
  }
  fail(
      ManifestStage::runtime_fingerprint,
      ManifestErrorCode::invalid_value,
      path,
      "endianness must be 'little' or 'big'");
}

[[nodiscard]] TensorDataType parse_dtype(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  static const std::unordered_map<std::string, TensorDataType> values{
      {"fp32", TensorDataType::fp32},
      {"fp16", TensorDataType::fp16},
      {"bf16", TensorDataType::bf16},
      {"int64", TensorDataType::int64},
      {"int32", TensorDataType::int32},
      {"int8", TensorDataType::int8},
      {"uint8", TensorDataType::uint8},
      {"bool", TensorDataType::boolean},
  };
  const auto found = values.find(parsed);
  if (found == values.end()) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "unsupported tensor dtype");
  }
  return found->second;
}

[[nodiscard]] TensorMemoryLocation parse_tensor_memory_location(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  const std::string parsed = parse_nonempty_string(value, stage, path);
  if (parsed == "device") {
    return TensorMemoryLocation::device;
  }
  if (parsed == "host") {
    return TensorMemoryLocation::host;
  }
  fail(
      stage,
      ManifestErrorCode::invalid_value,
      path,
      "tensor location must be 'device' or 'host'");
}

[[nodiscard]] EngineRole parse_engine_role(
    const Json& value,
    const std::string& path) {
  const std::string parsed =
      parse_nonempty_string(value, ManifestStage::engine, path);
  static const std::unordered_map<std::string, EngineRole> values{
      {"text_encoder", EngineRole::text_encoder},
      {"main_decoder_prefill", EngineRole::main_decoder_prefill},
      {"main_decoder_step", EngineRole::main_decoder_step},
      {"local_ar_16", EngineRole::local_ar_16},
      {"nanocodec_initial_4", EngineRole::nanocodec_initial_4},
      {"nanocodec_steady_8", EngineRole::nanocodec_steady_8},
      {"nanocodec_tail_1_8", EngineRole::nanocodec_tail_1_8},
  };
  const auto found = values.find(parsed);
  if (found == values.end()) {
    fail(
        ManifestStage::engine,
        ManifestErrorCode::invalid_value,
        path,
        "unsupported engine role");
  }
  return found->second;
}

[[nodiscard]] FileArtifact parse_file_artifact(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  require_exact_keys(value, stage, path, {"path", "sha256", "size_bytes"});
  return FileArtifact{
      .path = parse_relative_path(member(value, "path"), stage, child_path(path, "path")),
      .sha256 =
          parse_sha256(member(value, "sha256"), stage, child_path(path, "sha256")),
      .size_bytes = parse_positive_integer<std::uint64_t>(
          member(value, "size_bytes"),
          stage,
          child_path(path, "size_bytes")),
  };
}

[[nodiscard]] std::vector<LicenseArtifact> parse_licenses(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::licenses;
  constexpr std::array<std::string_view, 8> expected_roles{
      "project_license",
      "project_notice",
      "pytorch_license",
      "cutlass_license",
      "cuda_eula",
      "cuda_notice",
      "nvidia_open_model_license",
      "nvidia_model_notice",
  };
  if (!value.is_array()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        path,
        "licenses must be an array");
  }
  if (value.size() != expected_roles.size()) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "licenses must contain the canonical eight-role inventory");
  }
  std::vector<LicenseArtifact> licenses;
  licenses.reserve(expected_roles.size());
  for (std::size_t index = 0; index < expected_roles.size(); ++index) {
    const Json& item = value.at(index);
    const std::string item_path = index_path(path, index);
    require_exact_keys(item, stage, item_path, {"role", "file"});
    const std::string role = parse_identifier(
        member(item, "role"),
        stage,
        child_path(item_path, "role"));
    if (role != expected_roles.at(index)) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          child_path(item_path, "role"),
          "license role is absent, duplicated, or outside canonical order");
    }
    licenses.push_back(LicenseArtifact{
        .role = role,
        .file = parse_file_artifact(
            member(item, "file"),
            stage,
            child_path(item_path, "file")),
    });
  }
  return licenses;
}

[[nodiscard]] RuntimeFingerprint parse_runtime_fingerprint(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::runtime_fingerprint;
  require_exact_keys(
      value,
      stage,
      path,
      {"os_name",
       "os_version",
       "architecture",
       "endianness",
       "cuda_version",
       "tensorrt_version",
       "driver_version",
       "gpu_name",
       "gpu_compute_capability",
       "plugin_abi_version"});

  const std::string architecture = parse_nonempty_string(
      member(value, "architecture"), stage, child_path(path, "architecture"));
  if (architecture != "aarch64" && architecture != "x86_64") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "architecture"),
        "architecture must be 'aarch64' or 'x86_64'");
  }

  const std::string compute_capability = parse_nonempty_string(
      member(value, "gpu_compute_capability"),
      stage,
      child_path(path, "gpu_compute_capability"));
  if (!character_validation::is_major_minor_version(compute_capability)) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "gpu_compute_capability"),
        "GPU compute capability must use major.minor form");
  }

  return RuntimeFingerprint{
      .os_name =
          parse_identifier(member(value, "os_name"), stage, child_path(path, "os_name")),
      .os_version = parse_nonempty_string(
          member(value, "os_version"), stage, child_path(path, "os_version")),
      .architecture = architecture,
      .endianness =
          parse_endianness(member(value, "endianness"), child_path(path, "endianness")),
      .cuda_version = parse_numeric_version(
          member(value, "cuda_version"), stage, child_path(path, "cuda_version")),
      .tensorrt_version = parse_numeric_version(
          member(value, "tensorrt_version"),
          stage,
          child_path(path, "tensorrt_version")),
      .driver_version = parse_numeric_version(
          member(value, "driver_version"), stage, child_path(path, "driver_version")),
      .gpu_name = parse_nonempty_string(
          member(value, "gpu_name"), stage, child_path(path, "gpu_name")),
      .gpu_compute_capability = compute_capability,
      .plugin_abi_version = parse_positive_integer<std::uint32_t>(
          member(value, "plugin_abi_version"),
          stage,
          child_path(path, "plugin_abi_version")),
  };
}

[[nodiscard]] ArtifactsManifest parse_artifacts(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::artifacts;
  require_exact_keys(
      value, stage, path, {"source_model", "export", "tokenizer", "plugin"});

  const Json& model_json = member(value, "source_model");
  const std::string model_path = child_path(path, "source_model");
  require_exact_keys(
      model_json,
      stage,
      model_path,
      {"model_id",
       "version",
       "revision",
       "source_sha256",
       "acceptance_receipt"});
  SourceModelArtifact source_model{
      .model_id = parse_nonempty_string(
          member(model_json, "model_id"), stage, child_path(model_path, "model_id")),
      .version = parse_nonempty_string(
          member(model_json, "version"), stage, child_path(model_path, "version")),
      .revision = parse_git_sha1(
          member(model_json, "revision"), stage, child_path(model_path, "revision")),
      .source_sha256 = parse_sha256(
          member(model_json, "source_sha256"),
          stage,
          child_path(model_path, "source_sha256")),
      .acceptance_receipt = parse_file_artifact(
          member(model_json, "acceptance_receipt"),
          stage,
          child_path(model_path, "acceptance_receipt")),
  };

  const Json& export_json = member(value, "export");
  const std::string export_path = child_path(path, "export");
  require_exact_keys(
      export_json,
      stage,
      export_path,
      {"format",
       "source_revision",
       "voice_id",
       "baked_context_length",
       "baked_context_sha256",
       "audio_bos_baked",
       "export_receipt"});
  const std::string export_format = parse_identifier(
      member(export_json, "format"), stage, child_path(export_path, "format"));
  if (export_format != "magpie_tts_rt_export_v1") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(export_path, "format"),
        "schema version 1 requires export format 'magpie_tts_rt_export_v1'");
  }
  ExportArtifact export_artifact{
      .format = export_format,
      .source_revision = parse_nonempty_string(
          member(export_json, "source_revision"),
          stage,
          child_path(export_path, "source_revision")),
      .voice_id = parse_identifier(
          member(export_json, "voice_id"),
          stage,
          child_path(export_path, "voice_id")),
      .baked_context_length = parse_positive_integer<std::uint32_t>(
          member(export_json, "baked_context_length"),
          stage,
          child_path(export_path, "baked_context_length")),
      .baked_context_sha256 = parse_sha256(
          member(export_json, "baked_context_sha256"),
          stage,
          child_path(export_path, "baked_context_sha256")),
      .audio_bos_baked = parse_boolean(
          member(export_json, "audio_bos_baked"),
          stage,
          child_path(export_path, "audio_bos_baked")),
      .export_receipt = parse_file_artifact(
          member(export_json, "export_receipt"),
          stage,
          child_path(export_path, "export_receipt")),
  };
  if (export_artifact.voice_id != "sofia") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(export_path, "voice_id"),
        "schema version 1 is a voice-specific Sofia bundle");
  }
  if (export_artifact.baked_context_length != 217) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(export_path, "baked_context_length"),
        "schema version 1 requires exactly 217 baked context positions");
  }
  if (!export_artifact.audio_bos_baked) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(export_path, "audio_bos_baked"),
        "schema version 1 requires AUDIO_BOS to be baked into the prefill plan");
  }

  const Json& tokenizer_json = member(value, "tokenizer");
  const std::string tokenizer_path = child_path(path, "tokenizer");
  require_exact_keys(
      tokenizer_json,
      stage,
      tokenizer_path,
      {"kind",
       "tokenizer_vocabulary_size",
       "text_embedding_rows",
       "special_tokens",
       "identity_sha256",
       "identity_receipt"});
  const Json& special_tokens_json = member(tokenizer_json, "special_tokens");
  const std::string special_tokens_path =
      child_path(tokenizer_path, "special_tokens");
  require_exact_keys(
      special_tokens_json,
      stage,
      special_tokens_path,
      {"bos_token_id",
       "eos_token_id",
       "japanese_global_pad_token_id"});
  TokenizerArtifact tokenizer{
      .kind = parse_identifier(
          member(tokenizer_json, "kind"), stage, child_path(tokenizer_path, "kind")),
      .tokenizer_vocabulary_size = parse_positive_integer<std::uint32_t>(
          member(tokenizer_json, "tokenizer_vocabulary_size"),
          stage,
          child_path(tokenizer_path, "tokenizer_vocabulary_size")),
      .text_embedding_rows = parse_positive_integer<std::uint32_t>(
          member(tokenizer_json, "text_embedding_rows"),
          stage,
          child_path(tokenizer_path, "text_embedding_rows")),
      .bos_token_id = parse_unsigned_integer<std::uint32_t>(
          member(special_tokens_json, "bos_token_id"),
          stage,
          child_path(special_tokens_path, "bos_token_id")),
      .eos_token_id = parse_unsigned_integer<std::uint32_t>(
          member(special_tokens_json, "eos_token_id"),
          stage,
          child_path(special_tokens_path, "eos_token_id")),
      .japanese_global_pad_token_id =
          parse_unsigned_integer<std::uint32_t>(
              member(
                  special_tokens_json,
                  "japanese_global_pad_token_id"),
              stage,
              child_path(
                  special_tokens_path,
                  "japanese_global_pad_token_id")),
      .identity_sha256 = parse_sha256(
          member(tokenizer_json, "identity_sha256"),
          stage,
          child_path(tokenizer_path, "identity_sha256")),
      .identity_receipt = parse_file_artifact(
          member(tokenizer_json, "identity_receipt"),
          stage,
          child_path(tokenizer_path, "identity_receipt")),
  };
  if (tokenizer.tokenizer_vocabulary_size >=
      tokenizer.text_embedding_rows) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(tokenizer_path, "text_embedding_rows"),
        "text embedding rows must include authenticated special-token rows");
  }
  if (tokenizer.bos_token_id < tokenizer.tokenizer_vocabulary_size ||
      tokenizer.bos_token_id >= tokenizer.text_embedding_rows) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(special_tokens_path, "bos_token_id"),
        "BOS must identify a special text-embedding row");
  }
  if (tokenizer.eos_token_id < tokenizer.tokenizer_vocabulary_size ||
      tokenizer.eos_token_id >= tokenizer.text_embedding_rows) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(special_tokens_path, "eos_token_id"),
        "EOS must identify a special text-embedding row");
  }
  if (tokenizer.bos_token_id == tokenizer.eos_token_id) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        special_tokens_path,
        "BOS and EOS identifiers must differ");
  }
  if (tokenizer.japanese_global_pad_token_id >=
      tokenizer.tokenizer_vocabulary_size) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(
            special_tokens_path,
            "japanese_global_pad_token_id"),
        "the Japanese pad identifier must be a normal tokenizer row");
  }

  const Json& plugin_json = member(value, "plugin");
  const std::string plugin_path = child_path(path, "plugin");
  require_exact_keys(
      plugin_json,
      stage,
      plugin_path,
      {"name", "abi_version", "file", "build_receipt"});
  PluginArtifact plugin{
      .name = parse_identifier(
          member(plugin_json, "name"), stage, child_path(plugin_path, "name")),
      .abi_version = parse_positive_integer<std::uint32_t>(
          member(plugin_json, "abi_version"),
          stage,
          child_path(plugin_path, "abi_version")),
      .file = parse_file_artifact(
          member(plugin_json, "file"), stage, child_path(plugin_path, "file")),
      .build_receipt = parse_file_artifact(
          member(plugin_json, "build_receipt"),
          stage,
          child_path(plugin_path, "build_receipt")),
  };

  return ArtifactsManifest{
      .source_model = std::move(source_model),
      .export_artifact = std::move(export_artifact),
      .tokenizer = std::move(tokenizer),
      .plugin = std::move(plugin),
  };
}

[[nodiscard]] ClassifierFreeGuidanceManifest parse_classifier_free_guidance(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::classifier_free_guidance;
  require_exact_keys(
      value,
      stage,
      path,
      {"row_order",
       "conditional_row_index",
       "conditional_condition_source",
       "conditional_mask_source",
       "unconditional_row_index",
       "unconditional_condition_source",
       "unconditional_mask_source"});

  ClassifierFreeGuidanceManifest manifest{
      .row_order = parse_identifier(
          member(value, "row_order"), stage, child_path(path, "row_order")),
      .conditional_row_index = parse_unsigned_integer<std::uint32_t>(
          member(value, "conditional_row_index"),
          stage,
          child_path(path, "conditional_row_index")),
      .conditional_condition_source = parse_identifier(
          member(value, "conditional_condition_source"),
          stage,
          child_path(path, "conditional_condition_source")),
      .conditional_mask_source = parse_identifier(
          member(value, "conditional_mask_source"),
          stage,
          child_path(path, "conditional_mask_source")),
      .unconditional_row_index = parse_unsigned_integer<std::uint32_t>(
          member(value, "unconditional_row_index"),
          stage,
          child_path(path, "unconditional_row_index")),
      .unconditional_condition_source = parse_identifier(
          member(value, "unconditional_condition_source"),
          stage,
          child_path(path, "unconditional_condition_source")),
      .unconditional_mask_source = parse_identifier(
          member(value, "unconditional_mask_source"),
          stage,
          child_path(path, "unconditional_mask_source")),
  };

  if (manifest.row_order != "conditional_then_unconditional") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "row_order"),
        "CFG rows must be ordered conditional then unconditional");
  }
  if (manifest.conditional_row_index != 0 ||
      manifest.unconditional_row_index != 1) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "CFG conditional and unconditional rows must be indices 0 and 1");
  }
  if (manifest.conditional_condition_source != "text_encoder_output" ||
      manifest.conditional_mask_source != "text_mask" ||
      manifest.unconditional_condition_source != "all_zero" ||
      manifest.unconditional_mask_source != "first_true_rest_false") {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "CFG row construction must use text encoder output/text mask for row 0 and an all-zero condition with only mask position zero true for row 1");
  }
  return manifest;
}

[[nodiscard]] std::vector<std::int64_t> parse_declared_shape(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  if (!value.is_array()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        path,
        "shape must be an array");
  }
  std::vector<std::int64_t> shape;
  shape.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const Json& dimension = value.at(index);
    if (!dimension.is_number_integer()) {
      fail(
          stage,
          ManifestErrorCode::type_mismatch,
          index_path(path, index),
          "shape dimension must be an integer");
    }
    const std::int64_t parsed = dimension.get<std::int64_t>();
    if (parsed == 0 || parsed < -1) {
      fail(
          stage,
          ManifestErrorCode::invalid_value,
          index_path(path, index),
          "shape dimension must be -1 or a positive integer");
    }
    shape.push_back(parsed);
  }
  return shape;
}

[[nodiscard]] std::vector<std::int64_t> parse_concrete_shape(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  if (!value.is_array() || value.empty()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        path,
        "profile shape must be a non-empty array");
  }
  std::vector<std::int64_t> shape;
  shape.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const Json& dimension = value.at(index);
    if (!dimension.is_number_integer()) {
      fail(
          stage,
          ManifestErrorCode::type_mismatch,
          index_path(path, index),
          "profile dimension must be an integer");
    }
    const std::int64_t parsed = dimension.get<std::int64_t>();
    if (parsed <= 0) {
      fail(
          stage,
          ManifestErrorCode::invalid_value,
          index_path(path, index),
          "profile dimension must be positive");
    }
    shape.push_back(parsed);
  }
  return shape;
}

[[nodiscard]] std::vector<std::int64_t> parse_profile_values(
    const Json& value,
    const ManifestStage stage,
    const std::string& path) {
  if (!value.is_array() || value.empty()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        path,
        "profile values must be a non-empty array");
  }
  std::vector<std::int64_t> values;
  values.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const Json& item = value.at(index);
    if (!item.is_number_integer()) {
      fail(
          stage,
          ManifestErrorCode::type_mismatch,
          index_path(path, index),
          "profile value must be an integer");
    }
    values.push_back(item.get<std::int64_t>());
  }
  return values;
}

[[nodiscard]] TensorSpec parse_tensor_spec(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::tensor;
  require_exact_keys(
      value,
      stage,
      path,
      {"name", "dtype", "shape", "location", "shape_inference_io"});
  return TensorSpec{
      .name =
          parse_identifier(member(value, "name"), stage, child_path(path, "name")),
      .dtype = parse_dtype(member(value, "dtype"), stage, child_path(path, "dtype")),
      .shape =
          parse_declared_shape(member(value, "shape"), stage, child_path(path, "shape")),
      .location = parse_tensor_memory_location(
          member(value, "location"),
          stage,
          child_path(path, "location")),
      .shape_inference_io = parse_boolean(
          member(value, "shape_inference_io"),
          stage,
          child_path(path, "shape_inference_io")),
  };
}

[[nodiscard]] std::vector<TensorSpec> parse_tensor_specs(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::tensor;
  if (!value.is_array() || value.empty()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        path,
        "tensor list must be a non-empty array");
  }

  std::vector<TensorSpec> tensors;
  std::unordered_set<std::string> names;
  tensors.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    TensorSpec tensor = parse_tensor_spec(value.at(index), index_path(path, index));
    if (!names.emplace(tensor.name).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(index_path(path, index), "name"),
          "tensor name is duplicated");
    }
    tensors.emplace_back(std::move(tensor));
  }
  return tensors;
}

[[nodiscard]] TensorShapeRange parse_shape_range(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::tensor;
  require_exact_keys(value, stage, path, {"tensor_name", "min", "opt", "max"});
  TensorShapeRange range{
      .tensor_name = parse_identifier(
          member(value, "tensor_name"), stage, child_path(path, "tensor_name")),
      .minimum =
          parse_concrete_shape(member(value, "min"), stage, child_path(path, "min")),
      .optimum =
          parse_concrete_shape(member(value, "opt"), stage, child_path(path, "opt")),
      .maximum =
          parse_concrete_shape(member(value, "max"), stage, child_path(path, "max")),
  };

  if (range.minimum.size() != range.optimum.size() ||
      range.optimum.size() != range.maximum.size()) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "profile min/opt/max ranks must match");
  }
  for (std::size_t dimension = 0; dimension < range.minimum.size(); ++dimension) {
    if (range.minimum.at(dimension) > range.optimum.at(dimension) ||
        range.optimum.at(dimension) > range.maximum.at(dimension)) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          index_path(child_path(path, "opt"), dimension),
          "profile dimensions must satisfy min <= opt <= max");
    }
  }
  return range;
}

[[nodiscard]] TensorValueRange parse_value_range(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::tensor;
  require_exact_keys(value, stage, path, {"tensor_name", "min", "opt", "max"});
  TensorValueRange range{
      .tensor_name = parse_identifier(
          member(value, "tensor_name"), stage, child_path(path, "tensor_name")),
      .minimum = parse_profile_values(
          member(value, "min"), stage, child_path(path, "min")),
      .optimum = parse_profile_values(
          member(value, "opt"), stage, child_path(path, "opt")),
      .maximum = parse_profile_values(
          member(value, "max"), stage, child_path(path, "max")),
  };
  if (range.minimum.size() != range.optimum.size() ||
      range.optimum.size() != range.maximum.size()) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "profile value min/opt/max lengths must match");
  }
  for (std::size_t index = 0; index < range.minimum.size(); ++index) {
    if (range.minimum.at(index) > range.optimum.at(index) ||
        range.optimum.at(index) > range.maximum.at(index)) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          index_path(child_path(path, "opt"), index),
          "profile values must satisfy min <= opt <= max");
    }
  }
  return range;
}

[[nodiscard]] OptimizationProfile parse_profile(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::tensor;
  require_exact_keys(
      value, stage, path, {"name", "input_shapes", "input_values"});
  const Json& ranges_json = member(value, "input_shapes");
  const std::string ranges_path = child_path(path, "input_shapes");
  if (!ranges_json.is_array()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        ranges_path,
        "input_shapes must be an array");
  }

  OptimizationProfile profile{
      .name =
          parse_identifier(member(value, "name"), stage, child_path(path, "name")),
      .input_shapes = {},
      .input_values = {},
  };
  std::unordered_set<std::string> tensor_names;
  profile.input_shapes.reserve(ranges_json.size());
  for (std::size_t index = 0; index < ranges_json.size(); ++index) {
    TensorShapeRange range =
        parse_shape_range(ranges_json.at(index), index_path(ranges_path, index));
    if (!tensor_names.emplace(range.tensor_name).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(index_path(ranges_path, index), "tensor_name"),
          "profile tensor name is duplicated");
    }
    profile.input_shapes.emplace_back(std::move(range));
  }
  const Json& values_json = member(value, "input_values");
  const std::string values_path = child_path(path, "input_values");
  if (!values_json.is_array()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        values_path,
        "input_values must be an array");
  }
  tensor_names.clear();
  profile.input_values.reserve(values_json.size());
  for (std::size_t index = 0; index < values_json.size(); ++index) {
    TensorValueRange range =
        parse_value_range(values_json.at(index), index_path(values_path, index));
    if (!tensor_names.emplace(range.tensor_name).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(index_path(values_path, index), "tensor_name"),
          "profile tensor name is duplicated");
    }
    profile.input_values.emplace_back(std::move(range));
  }
  return profile;
}

void validate_profile_against_inputs(
    const OptimizationProfile& profile,
    const std::vector<TensorSpec>& inputs,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::tensor;
  std::unordered_map<std::string, const TensorSpec*> input_by_name;
  for (const TensorSpec& input : inputs) {
    input_by_name.emplace(input.name, &input);
  }

  const std::size_t dynamic_input_count = static_cast<std::size_t>(
      std::count_if(inputs.begin(), inputs.end(), [](const TensorSpec& input) {
        return std::find(input.shape.begin(), input.shape.end(), -1) !=
               input.shape.end();
      }));
  if (profile.input_shapes.size() != dynamic_input_count) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "input_shapes"),
        "every dynamic engine input must have exactly one shape range, and static inputs must not have one");
  }
  const std::size_t shape_input_count = static_cast<std::size_t>(
      std::count_if(inputs.begin(), inputs.end(), [](const TensorSpec& input) {
        return input.shape_inference_io;
      }));
  if (profile.input_values.size() != shape_input_count) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "input_values"),
        "every shape engine input must have exactly one value range, and ordinary inputs must not have one");
  }

  for (std::size_t range_index = 0; range_index < profile.input_shapes.size();
       ++range_index) {
    const TensorShapeRange& range = profile.input_shapes.at(range_index);
    const auto input = input_by_name.find(range.tensor_name);
    if (input == input_by_name.end()) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          child_path(
              index_path(child_path(path, "input_shapes"), range_index),
              "tensor_name"),
          "profile references a tensor that is not an engine input");
    }

    const TensorSpec& spec = *input->second;
    if (std::find(spec.shape.begin(), spec.shape.end(), -1) ==
        spec.shape.end()) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          child_path(
              index_path(child_path(path, "input_shapes"), range_index),
              "tensor_name"),
          "static engine inputs must not appear in an optimization profile");
    }
    if (range.minimum.size() != spec.shape.size()) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          index_path(child_path(path, "input_shapes"), range_index),
          "profile rank does not match declared input rank");
    }
    for (std::size_t dimension = 0; dimension < spec.shape.size(); ++dimension) {
      const std::int64_t declared = spec.shape.at(dimension);
      if (declared != -1 &&
          (range.minimum.at(dimension) != declared ||
           range.optimum.at(dimension) != declared ||
           range.maximum.at(dimension) != declared)) {
        fail(
            stage,
            ManifestErrorCode::invariant_violation,
            index_path(
                child_path(
                    index_path(child_path(path, "input_shapes"), range_index),
                    "opt"),
                dimension),
            "static input dimension must be identical in min/opt/max");
      }
    }
  }
  for (std::size_t range_index = 0;
       range_index < profile.input_values.size();
       ++range_index) {
    const TensorValueRange& range =
        profile.input_values.at(range_index);
    const auto input = input_by_name.find(range.tensor_name);
    if (input == input_by_name.end()) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          child_path(
              index_path(child_path(path, "input_values"), range_index),
              "tensor_name"),
          "profile references a tensor that is not an engine input");
    }
    const TensorSpec& spec = *input->second;
    if (!spec.shape_inference_io) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          child_path(
              index_path(child_path(path, "input_values"), range_index),
              "tensor_name"),
          "ordinary engine inputs must not appear in a value profile");
    }
    std::uint64_t expected_values = 1;
    for (const std::int64_t dimension : spec.shape) {
      if (dimension <= 0 ||
          expected_values >
              std::numeric_limits<std::uint64_t>::max() /
                  static_cast<std::uint64_t>(dimension)) {
        fail(
            stage,
            ManifestErrorCode::invariant_violation,
            child_path(
                index_path(child_path(path, "input_values"), range_index),
                "tensor_name"),
            "shape input must have a fixed, representable value count");
      }
      expected_values *= static_cast<std::uint64_t>(dimension);
    }
    if (expected_values != range.minimum.size()) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          index_path(child_path(path, "input_values"), range_index),
          "profile value count does not match the shape input");
    }
  }
}

[[nodiscard]] EngineManifest parse_engine(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::engine;
  require_exact_keys(
      value, stage, path, {"name", "role", "file", "inputs", "outputs", "profiles"});
  EngineManifest engine{
      .name =
          parse_identifier(member(value, "name"), stage, child_path(path, "name")),
      .role = parse_engine_role(member(value, "role"), child_path(path, "role")),
      .file =
          parse_file_artifact(member(value, "file"), stage, child_path(path, "file")),
      .inputs =
          parse_tensor_specs(member(value, "inputs"), child_path(path, "inputs")),
      .outputs =
          parse_tensor_specs(member(value, "outputs"), child_path(path, "outputs")),
      .profiles = {},
  };

  std::unordered_set<std::string> all_tensor_names;
  for (const TensorSpec& input : engine.inputs) {
    all_tensor_names.emplace(input.name);
  }
  for (const TensorSpec& output : engine.outputs) {
    if (!all_tensor_names.emplace(output.name).second) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::duplicate_value,
          child_path(path, "outputs"),
          "input and output tensor names must be disjoint");
    }
  }
  for (const TensorSpec& input : engine.inputs) {
    const bool is_position_shape_input =
        engine.role == EngineRole::main_decoder_step &&
        input.name == "position";
    if (is_position_shape_input) {
      if (input.dtype != TensorDataType::int64 ||
          !input.shape.empty() ||
          input.location != TensorMemoryLocation::host ||
          !input.shape_inference_io) {
        fail(
            ManifestStage::tensor,
            ManifestErrorCode::invariant_violation,
            child_path(path, "inputs"),
            "main_decoder_step/position must be the scalar int64 HOST shape input");
      }
    } else if (
        input.location != TensorMemoryLocation::device ||
        input.shape_inference_io) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::invariant_violation,
          child_path(path, "inputs"),
          "only main_decoder_step/position may be a HOST shape input");
    }
  }
  for (const TensorSpec& output : engine.outputs) {
    if (output.location != TensorMemoryLocation::device ||
        output.shape_inference_io) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::invariant_violation,
          child_path(path, "outputs"),
          "all engine outputs must be ordinary CUDA device tensors");
    }
  }

  const Json& profiles_json = member(value, "profiles");
  const std::string profiles_path = child_path(path, "profiles");
  if (!profiles_json.is_array() || profiles_json.empty()) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::type_mismatch,
        profiles_path,
        "profiles must be a non-empty array");
  }
  std::unordered_set<std::string> profile_names;
  engine.profiles.reserve(profiles_json.size());
  for (std::size_t index = 0; index < profiles_json.size(); ++index) {
    const std::string profile_path = index_path(profiles_path, index);
    OptimizationProfile profile =
        parse_profile(profiles_json.at(index), profile_path);
    if (!profile_names.emplace(profile.name).second) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::duplicate_value,
          child_path(profile_path, "name"),
          "optimization profile name is duplicated");
    }
    validate_profile_against_inputs(profile, engine.inputs, profile_path);
    engine.profiles.emplace_back(std::move(profile));
  }
  return engine;
}

[[nodiscard]] std::vector<EngineManifest> parse_engines(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::engine;
  if (!value.is_array() || value.empty()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        path,
        "engines must be a non-empty array");
  }

  std::vector<EngineManifest> engines;
  std::unordered_set<std::string> names;
  std::unordered_set<std::string> paths;
  std::set<EngineRole> roles;
  engines.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const std::string engine_path = index_path(path, index);
    EngineManifest engine = parse_engine(value.at(index), engine_path);
    if (!names.emplace(engine.name).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(engine_path, "name"),
          "engine name is duplicated");
    }
    if (!paths.emplace(engine.file.path.generic_string()).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(child_path(engine_path, "file"), "path"),
          "engine file path is duplicated");
    }
    if (!roles.emplace(engine.role).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(engine_path, "role"),
          "engine role is duplicated");
    }
    engines.emplace_back(std::move(engine));
  }

  constexpr std::array<EngineRole, 7> required_roles{
      EngineRole::text_encoder,
      EngineRole::main_decoder_prefill,
      EngineRole::main_decoder_step,
      EngineRole::local_ar_16,
      EngineRole::nanocodec_initial_4,
      EngineRole::nanocodec_steady_8,
      EngineRole::nanocodec_tail_1_8,
  };
  if (engines.size() != required_roles.size()) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "schema version 1 requires exactly seven engines");
  }
  for (const EngineRole required_role : required_roles) {
    if (!roles.contains(required_role)) {
      fail(
          stage,
          ManifestErrorCode::missing_field,
          path,
          "required engine role is missing: " +
              std::string(to_string(required_role)));
    }
  }
  return engines;
}

[[nodiscard]] KvCacheManifest parse_kv_cache(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::kv_cache;
  require_exact_keys(
      value,
      stage,
      path,
      {"layout",
       "dtype",
       "mask_dtype",
       "layers",
       "batch_size",
       "self_attention_heads",
       "self_attention_head_dimension",
       "cross_attention_heads",
       "cross_attention_head_dimension",
       "prefix_length",
       "maximum_generated_steps",
       "self_cache_capacity",
       "update_mode",
       "position_semantics",
       "first_step_position",
       "step_position_upper_bound_exclusive",
       "layer_bindings"});

  const std::string layout =
      parse_nonempty_string(member(value, "layout"), stage, child_path(path, "layout"));
  if (layout != "batch_sequence_head_dimension") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "layout"),
        "layout must be 'batch_sequence_head_dimension'");
  }
  const std::string update_mode = parse_nonempty_string(
      member(value, "update_mode"), stage, child_path(path, "update_mode"));
  if (update_mode != "in_place_explicit_bindings") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "update_mode"),
        "update_mode must be 'in_place_explicit_bindings'");
  }
  const std::string position_semantics = parse_nonempty_string(
      member(value, "position_semantics"),
      stage,
      child_path(path, "position_semantics"));
  if (position_semantics != "absolute_self_cache_write_index") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "position_semantics"),
        "step position must be an absolute self-cache write index");
  }

  const std::uint32_t layers = parse_positive_integer<std::uint32_t>(
      member(value, "layers"), stage, child_path(path, "layers"));
  if (layers != 12) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "layers"),
        "schema version 1 requires exactly 12 Main Decoder layers");
  }

  const Json& bindings_json = member(value, "layer_bindings");
  const std::string bindings_path = child_path(path, "layer_bindings");
  if (!bindings_json.is_array() || bindings_json.size() != layers) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        bindings_path,
        "layer_bindings must contain exactly one record for each of 12 layers");
  }

  std::vector<KvLayerBindings> layer_bindings;
  std::unordered_set<std::string> prefill_outputs;
  std::unordered_set<std::string> step_inputs;
  std::unordered_set<std::string> step_outputs;
  layer_bindings.reserve(layers);

  const auto parse_unique_binding = [&stage](
                                        const Json& binding_json,
                                        const std::string& binding_path,
                                        const std::string_view key,
                                        std::unordered_set<std::string>& names) {
    std::string name = parse_identifier(
        member(binding_json, key), stage, child_path(binding_path, key));
    if (!names.emplace(name).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(binding_path, key),
          "binding name is reused in the same engine I/O direction");
    }
    return name;
  };

  for (std::size_t index = 0; index < bindings_json.size(); ++index) {
    const Json& binding_json = bindings_json.at(index);
    const std::string binding_path = index_path(bindings_path, index);
    require_exact_keys(
        binding_json,
        stage,
        binding_path,
        {"layer_index",
         "prefill_self_key_output",
         "prefill_self_value_output",
         "prefill_self_mask_output",
         "prefill_cross_key_output",
         "prefill_cross_value_output",
         "step_self_key_input",
         "step_self_value_input",
         "step_self_mask_input",
         "step_cross_key_input",
         "step_cross_value_input",
         "step_self_key_output",
         "step_self_value_output",
         "step_self_mask_output"});
    const std::uint32_t layer_index = parse_unsigned_integer<std::uint32_t>(
        member(binding_json, "layer_index"),
        stage,
        child_path(binding_path, "layer_index"));
    if (layer_index != index) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          child_path(binding_path, "layer_index"),
          "layer bindings must be ordered with unique indices 0 through 11");
    }

    layer_bindings.push_back(KvLayerBindings{
        .layer_index = layer_index,
        .prefill_self_key_output = parse_unique_binding(
            binding_json,
            binding_path,
            "prefill_self_key_output",
            prefill_outputs),
        .prefill_self_value_output = parse_unique_binding(
            binding_json,
            binding_path,
            "prefill_self_value_output",
            prefill_outputs),
        .prefill_self_mask_output = parse_unique_binding(
            binding_json,
            binding_path,
            "prefill_self_mask_output",
            prefill_outputs),
        .prefill_cross_key_output = parse_unique_binding(
            binding_json,
            binding_path,
            "prefill_cross_key_output",
            prefill_outputs),
        .prefill_cross_value_output = parse_unique_binding(
            binding_json,
            binding_path,
            "prefill_cross_value_output",
            prefill_outputs),
        .step_self_key_input = parse_unique_binding(
            binding_json, binding_path, "step_self_key_input", step_inputs),
        .step_self_value_input = parse_unique_binding(
            binding_json, binding_path, "step_self_value_input", step_inputs),
        .step_self_mask_input = parse_unique_binding(
            binding_json, binding_path, "step_self_mask_input", step_inputs),
        .step_cross_key_input = parse_unique_binding(
            binding_json, binding_path, "step_cross_key_input", step_inputs),
        .step_cross_value_input = parse_unique_binding(
            binding_json, binding_path, "step_cross_value_input", step_inputs),
        .step_self_key_output = parse_unique_binding(
            binding_json, binding_path, "step_self_key_output", step_outputs),
        .step_self_value_output = parse_unique_binding(
            binding_json, binding_path, "step_self_value_output", step_outputs),
        .step_self_mask_output = parse_unique_binding(
            binding_json, binding_path, "step_self_mask_output", step_outputs),
    });
  }

  KvCacheManifest manifest{
      .layout = layout,
      .dtype = parse_dtype(member(value, "dtype"), stage, child_path(path, "dtype")),
      .mask_dtype =
          parse_dtype(member(value, "mask_dtype"), stage, child_path(path, "mask_dtype")),
      .layers = layers,
      .batch_size = parse_positive_integer<std::uint32_t>(
          member(value, "batch_size"), stage, child_path(path, "batch_size")),
      .self_attention_heads = parse_positive_integer<std::uint32_t>(
          member(value, "self_attention_heads"),
          stage,
          child_path(path, "self_attention_heads")),
      .self_attention_head_dimension = parse_positive_integer<std::uint32_t>(
          member(value, "self_attention_head_dimension"),
          stage,
          child_path(path, "self_attention_head_dimension")),
      .cross_attention_heads = parse_positive_integer<std::uint32_t>(
          member(value, "cross_attention_heads"),
          stage,
          child_path(path, "cross_attention_heads")),
      .cross_attention_head_dimension = parse_positive_integer<std::uint32_t>(
          member(value, "cross_attention_head_dimension"),
          stage,
          child_path(path, "cross_attention_head_dimension")),
      .prefix_length = parse_positive_integer<std::uint32_t>(
          member(value, "prefix_length"), stage, child_path(path, "prefix_length")),
      .maximum_generated_steps = parse_positive_integer<std::uint32_t>(
          member(value, "maximum_generated_steps"),
          stage,
          child_path(path, "maximum_generated_steps")),
      .self_cache_capacity = parse_positive_integer<std::uint32_t>(
          member(value, "self_cache_capacity"),
          stage,
          child_path(path, "self_cache_capacity")),
      .update_mode = update_mode,
      .position_semantics = position_semantics,
      .first_step_position = parse_positive_integer<std::uint32_t>(
          member(value, "first_step_position"),
          stage,
          child_path(path, "first_step_position")),
      .step_position_upper_bound_exclusive =
          parse_positive_integer<std::uint32_t>(
              member(value, "step_position_upper_bound_exclusive"),
              stage,
              child_path(path, "step_position_upper_bound_exclusive")),
      .layer_bindings = std::move(layer_bindings),
  };
  if (manifest.mask_dtype != TensorDataType::boolean) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "mask_dtype"),
        "KV key masks must use bool dtype");
  }
  if (manifest.batch_size != 2 || manifest.self_attention_heads != 12 ||
      manifest.self_attention_head_dimension != 64 ||
      manifest.cross_attention_heads != 1 ||
      manifest.cross_attention_head_dimension != 128) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "schema version 1 requires CFG batch 2, self attention 12x64, and cross attention 1x128");
  }
  if (manifest.prefix_length != 217 ||
      manifest.maximum_generated_steps != 250 ||
      manifest.self_cache_capacity != 467) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "schema version 1 requires prefix=217, generated steps=250, and self cache capacity=467");
  }
  if (manifest.prefix_length + manifest.maximum_generated_steps !=
      manifest.self_cache_capacity) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "self_cache_capacity"),
        "self_cache_capacity must equal prefix_length + maximum_generated_steps");
  }
  if (static_cast<std::uint64_t>(manifest.first_step_position) !=
      static_cast<std::uint64_t>(manifest.prefix_length) + 1U) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "first_step_position"),
        "the first main_decoder_step call must write at prefix_length + 1");
  }
  if (manifest.step_position_upper_bound_exclusive !=
      manifest.self_cache_capacity) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "step_position_upper_bound_exclusive"),
        "step position upper bound must be self_cache_capacity");
  }
  if (static_cast<std::uint64_t>(
          manifest.step_position_upper_bound_exclusive) -
          static_cast<std::uint64_t>(manifest.first_step_position) !=
      static_cast<std::uint64_t>(manifest.maximum_generated_steps) - 1U) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "prefill emits one generated step and the remaining step calls must exactly fill the declared position interval");
  }
  return manifest;
}

[[nodiscard]] AlignmentManifest parse_alignment(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::alignment;
  require_exact_keys(
      value,
      stage,
      path,
      {"dtype",
       "source_decoder_layers",
       "prefill_output_binding",
       "step_prior_input_binding",
       "step_alignment_output_binding",
       "prior_epsilon",
       "initial_attended",
       "ignored_terminal_tokens",
       "short_text_no_prior_max_tokens",
       "lookahead",
       "sink_threshold",
       "source_position_policy"});
  const Json& source_layers_json = member(value, "source_decoder_layers");
  const std::string source_layers_path =
      child_path(path, "source_decoder_layers");
  if (!source_layers_json.is_array() || source_layers_json.empty()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        source_layers_path,
        "source_decoder_layers must be a non-empty array");
  }
  std::vector<std::uint32_t> source_layers;
  std::unordered_set<std::uint32_t> unique_source_layers;
  source_layers.reserve(source_layers_json.size());
  for (std::size_t index = 0; index < source_layers_json.size(); ++index) {
    const std::uint32_t layer = parse_unsigned_integer<std::uint32_t>(
        source_layers_json.at(index),
        stage,
        index_path(source_layers_path, index));
    if (!unique_source_layers.emplace(layer).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          index_path(source_layers_path, index),
          "alignment source decoder layer is duplicated");
    }
    source_layers.push_back(layer);
  }
  if (source_layers != std::vector<std::uint32_t>{4, 5, 8, 9}) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        source_layers_path,
        "schema version 1 requires alignment layers [4, 5, 8, 9]");
  }
  const double prior_epsilon = parse_positive_number(
      member(value, "prior_epsilon"),
      stage,
      child_path(path, "prior_epsilon"));
  if (prior_epsilon != 0.1) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "prior_epsilon"),
        "canonical v1 alignment requires prior_epsilon exactly equal to JSON number 0.1");
  }
  const std::uint32_t initial_attended =
      parse_positive_integer<std::uint32_t>(
          member(value, "initial_attended"),
          stage,
          child_path(path, "initial_attended"));
  if (initial_attended != 1) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "initial_attended"),
        "canonical v1 alignment requires initial_attended=1");
  }
  const std::uint32_t ignored_terminal_tokens =
      parse_positive_integer<std::uint32_t>(
          member(value, "ignored_terminal_tokens"),
          stage,
          child_path(path, "ignored_terminal_tokens"));
  if (ignored_terminal_tokens != 3) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "ignored_terminal_tokens"),
        "canonical v1 alignment requires ignored_terminal_tokens=3");
  }
  const std::uint32_t short_text_no_prior_max_tokens =
      parse_positive_integer<std::uint32_t>(
          member(value, "short_text_no_prior_max_tokens"),
          stage,
          child_path(path, "short_text_no_prior_max_tokens"));
  if (short_text_no_prior_max_tokens != 5) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "short_text_no_prior_max_tokens"),
        "canonical v1 alignment requires short_text_no_prior_max_tokens=5");
  }
  const std::uint32_t lookahead = parse_positive_integer<std::uint32_t>(
      member(value, "lookahead"),
      stage,
      child_path(path, "lookahead"));
  if (lookahead != 6) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "lookahead"),
        "canonical v1 alignment requires lookahead=6");
  }
  const std::uint32_t sink_threshold =
      parse_positive_integer<std::uint32_t>(
          member(value, "sink_threshold"),
          stage,
          child_path(path, "sink_threshold"));
  if (sink_threshold != 4) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "sink_threshold"),
        "canonical v1 alignment requires sink_threshold=4");
  }
  const std::string source_position_policy = parse_nonempty_string(
      member(value, "source_position_policy"),
      stage,
      child_path(path, "source_position_policy"));
  if (source_position_policy != "exact_frontend_span_only") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "source_position_policy"),
        "source position policy must be 'exact_frontend_span_only'");
  }
  AlignmentManifest manifest{
      .dtype = parse_dtype(member(value, "dtype"), stage, child_path(path, "dtype")),
      .source_decoder_layers = std::move(source_layers),
      .prefill_output_binding = parse_identifier(
          member(value, "prefill_output_binding"),
          stage,
          child_path(path, "prefill_output_binding")),
      .step_prior_input_binding = parse_identifier(
          member(value, "step_prior_input_binding"),
          stage,
          child_path(path, "step_prior_input_binding")),
      .step_alignment_output_binding = parse_identifier(
          member(value, "step_alignment_output_binding"),
          stage,
          child_path(path, "step_alignment_output_binding")),
      .prior_epsilon = prior_epsilon,
      .initial_attended = initial_attended,
      .ignored_terminal_tokens = ignored_terminal_tokens,
      .short_text_no_prior_max_tokens = short_text_no_prior_max_tokens,
      .lookahead = lookahead,
      .sink_threshold = sink_threshold,
      .source_position_policy = source_position_policy,
  };
  if (manifest.dtype != TensorDataType::bf16 ||
      manifest.prefill_output_binding != "alignment" ||
      manifest.step_prior_input_binding != "alignment_prior" ||
      manifest.step_alignment_output_binding != "alignment") {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "canonical v1 alignment requires BF16 bindings alignment/alignment_prior/alignment");
  }
  return manifest;
}

[[nodiscard]] RngManifest parse_rng(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::sampling;
  require_exact_keys(
      value,
      stage,
      path,
      {"algorithm",
       "seed_bits",
       "counter_bits",
       "state_location",
       "ownership",
       "deterministic"});

  const std::string algorithm = parse_nonempty_string(
      member(value, "algorithm"), stage, child_path(path, "algorithm"));
  if (algorithm != "philox4x32-10") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "algorithm"),
        "RNG algorithm must be 'philox4x32-10'");
  }
  const std::uint32_t seed_bits = parse_positive_integer<std::uint32_t>(
      member(value, "seed_bits"), stage, child_path(path, "seed_bits"));
  const std::uint32_t counter_bits = parse_positive_integer<std::uint32_t>(
      member(value, "counter_bits"), stage, child_path(path, "counter_bits"));
  if (seed_bits != 64 || counter_bits != 64) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        path,
        "schema version 1 requires a 64-bit seed and 64-bit counter");
  }
  const std::string state_location = parse_nonempty_string(
      member(value, "state_location"), stage, child_path(path, "state_location"));
  if (state_location != "device") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "state_location"),
        "RNG state must remain on the device");
  }
  const std::string ownership = parse_nonempty_string(
      member(value, "ownership"), stage, child_path(path, "ownership"));
  if (ownership != "session") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "ownership"),
        "RNG state ownership must be 'session'");
  }
  const bool deterministic =
      parse_boolean(member(value, "deterministic"), stage, child_path(path, "deterministic"));
  if (!deterministic) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "deterministic"),
        "golden receipt validation requires deterministic RNG");
  }

  return RngManifest{
      .algorithm = algorithm,
      .seed_bits = seed_bits,
      .counter_bits = counter_bits,
      .state_location = state_location,
      .ownership = ownership,
      .deterministic = deterministic,
  };
}

[[nodiscard]] SamplingManifest parse_sampling(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::sampling;
  require_exact_keys(
      value,
      stage,
      path,
      {"algorithm",
       "top_k",
       "temperature",
       "eos_token_id",
       "forbidden_token_ids",
       "invalid_distribution_policy",
       "next_embedding_location",
       "rng"});

  const std::string algorithm = parse_nonempty_string(
      member(value, "algorithm"), stage, child_path(path, "algorithm"));
  if (algorithm != "top_k_gumbel_max") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "algorithm"),
        "sampling algorithm must be 'top_k_gumbel_max'");
  }
  constexpr std::uint32_t local_ar_codebook_size = 2016;
  constexpr std::uint32_t local_ar_vocabulary_size = 2024;
  constexpr std::uint32_t canonical_audio_eos_id = 2017;
  constexpr std::array<std::uint32_t, 7>
      canonical_static_forbidden_token_ids{
          2016, 2018, 2019, 2020, 2021, 2022, 2023};
  static_assert(
      canonical_static_forbidden_token_ids.front() ==
      local_ar_codebook_size);
  constexpr std::uint32_t canonical_top_k = 80;
  constexpr double canonical_temperature = 0.6;
  const std::uint32_t top_k = parse_positive_integer<std::uint32_t>(
      member(value, "top_k"), stage, child_path(path, "top_k"));
  if (top_k != canonical_top_k) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "top_k"),
        "canonical v1 sampling requires top_k=80");
  }
  const double temperature = parse_positive_number(
      member(value, "temperature"), stage, child_path(path, "temperature"));
  // The JSON number is parsed once as IEEE-754 binary64 and compared exactly
  // with the v1 literal. An epsilon would admit a different sampling
  // distribution and invalidate deterministic golden receipts.
  if (temperature != canonical_temperature) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "temperature"),
        "canonical v1 sampling requires temperature exactly equal to JSON number 0.6");
  }
  const std::uint32_t eos_token_id = parse_unsigned_integer<std::uint32_t>(
      member(value, "eos_token_id"), stage, child_path(path, "eos_token_id"));
  if (eos_token_id != canonical_audio_eos_id) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "eos_token_id"),
        "canonical Sofia v1 sampling requires AUDIO_EOS=2017");
  }

  const Json& forbidden_json = member(value, "forbidden_token_ids");
  const std::string forbidden_path = child_path(path, "forbidden_token_ids");
  if (!forbidden_json.is_array()) {
    fail(
        stage,
        ManifestErrorCode::type_mismatch,
        forbidden_path,
        "forbidden_token_ids must be an array");
  }
  std::vector<std::uint32_t> forbidden;
  std::unordered_set<std::uint32_t> forbidden_unique;
  forbidden.reserve(forbidden_json.size());
  for (std::size_t index = 0; index < forbidden_json.size(); ++index) {
    const std::uint32_t token = parse_unsigned_integer<std::uint32_t>(
        forbidden_json.at(index), stage, index_path(forbidden_path, index));
    if (token >= local_ar_vocabulary_size) {
      fail(
          stage,
          ManifestErrorCode::invalid_value,
          index_path(forbidden_path, index),
          "forbidden token id must be less than 2024");
    }
    if (!forbidden_unique.emplace(token).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          index_path(forbidden_path, index),
          "forbidden token id is duplicated");
    }
    if (token == eos_token_id) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          index_path(forbidden_path, index),
          "EOS token id must not appear in the static forbidden token set");
    }
    forbidden.push_back(token);
  }
  if (local_ar_vocabulary_size - forbidden.size() < top_k) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        forbidden_path,
        "static forbidden tokens leave fewer than top_k=80 eligible candidates");
  }
  if (!std::equal(
          forbidden.begin(),
          forbidden.end(),
          canonical_static_forbidden_token_ids.begin(),
          canonical_static_forbidden_token_ids.end())) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        forbidden_path,
        "canonical Sofia v1 sampling must permit codec IDs 0..2015, reserve AUDIO_EOS=2017, and forbid exactly [2016,2018,2019,2020,2021,2022,2023]");
  }

  const std::string invalid_policy = parse_nonempty_string(
      member(value, "invalid_distribution_policy"),
      stage,
      child_path(path, "invalid_distribution_policy"));
  if (invalid_policy != "fail_closed") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "invalid_distribution_policy"),
        "invalid distribution policy must be 'fail_closed'");
  }
  const std::string embedding_location = parse_nonempty_string(
      member(value, "next_embedding_location"),
      stage,
      child_path(path, "next_embedding_location"));
  if (embedding_location != "device") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "next_embedding_location"),
        "next embedding must remain on the device");
  }

  return SamplingManifest{
      .algorithm = algorithm,
      .top_k = top_k,
      .temperature = temperature,
      .eos_token_id = eos_token_id,
      .forbidden_token_ids = std::move(forbidden),
      .invalid_distribution_policy = invalid_policy,
      .next_embedding_location = embedding_location,
      .rng = parse_rng(member(value, "rng"), child_path(path, "rng")),
  };
}

[[nodiscard]] LocalArManifest parse_local_ar(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::local_ar;
  require_exact_keys(
      value,
      stage,
      path,
      {"engine_name",
       "execution",
       "iterations",
       "positions",
       "position_embedding",
       "codebooks_per_frame",
       "frames_per_decoder_step",
       "sampling_plugin_name",
       "invalid_rows_encoding",
       "no_eos_frame_index"});

  const std::string execution = parse_nonempty_string(
      member(value, "execution"), stage, child_path(path, "execution"));
  if (execution != "fixed_unrolled") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "execution"),
        "Local AR execution must be 'fixed_unrolled'");
  }
  const std::uint32_t iterations = parse_positive_integer<std::uint32_t>(
      member(value, "iterations"), stage, child_path(path, "iterations"));
  if (iterations != 16) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "iterations"),
        "Local AR must contain exactly 16 positions");
  }

  const std::string invalid_rows_encoding = parse_nonempty_string(
      member(value, "invalid_rows_encoding"),
      stage,
      child_path(path, "invalid_rows_encoding"));
  if (invalid_rows_encoding != "cfg_row_bitmask_lsb") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "invalid_rows_encoding"),
        "invalid_rows must be encoded as a least-significant-bit CFG row bitmask");
  }
  const Json& no_eos_frame_index_json = member(value, "no_eos_frame_index");
  if (!no_eos_frame_index_json.is_number_integer() ||
      no_eos_frame_index_json.get<std::int64_t>() != -1) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "no_eos_frame_index"),
        "the no-EOS frame index sentinel must be -1");
  }

  const Json& positions_json = member(value, "positions");
  const std::string positions_path = child_path(path, "positions");
  if (!positions_json.is_array() || positions_json.size() != 16) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        positions_path,
        "positions must contain exactly 16 entries");
  }
  std::vector<std::uint32_t> positions;
  positions.reserve(16);
  for (std::size_t index = 0; index < positions_json.size(); ++index) {
    const std::uint32_t position = parse_unsigned_integer<std::uint32_t>(
        positions_json.at(index), stage, index_path(positions_path, index));
    if (position != index) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          index_path(positions_path, index),
          "positions must be the ordered sequence 0 through 15");
    }
    positions.push_back(position);
  }

  const Json& position_embedding_json = member(value, "position_embedding");
  const std::string position_embedding_path =
      child_path(path, "position_embedding");
  require_exact_keys(
      position_embedding_json,
      stage,
      position_embedding_path,
      {"kind", "positions", "source_shape", "dtype", "source_table_sha256"});
  const std::string position_embedding_kind = parse_nonempty_string(
      member(position_embedding_json, "kind"),
      stage,
      child_path(position_embedding_path, "kind"));
  if (position_embedding_kind != "learned_absolute") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(position_embedding_path, "kind"),
        "Local AR position embedding must be learned absolute");
  }
  const Json& embedding_positions_json =
      member(position_embedding_json, "positions");
  const std::string embedding_positions_path =
      child_path(position_embedding_path, "positions");
  if (!embedding_positions_json.is_array() ||
      embedding_positions_json.size() != 16) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        embedding_positions_path,
        "position embedding positions must contain exactly 16 entries");
  }
  std::vector<std::uint32_t> embedding_positions;
  embedding_positions.reserve(16);
  for (std::size_t index = 0; index < embedding_positions_json.size(); ++index) {
    const std::uint32_t position = parse_unsigned_integer<std::uint32_t>(
        embedding_positions_json.at(index),
        stage,
        index_path(embedding_positions_path, index));
    if (position != index) {
      fail(
          stage,
          ManifestErrorCode::invariant_violation,
          index_path(embedding_positions_path, index),
          "position embedding rows must be the ordered sequence 0 through 15");
    }
    embedding_positions.push_back(position);
  }
  const Json& source_shape_json = member(position_embedding_json, "source_shape");
  const std::string source_shape_path =
      child_path(position_embedding_path, "source_shape");
  if (!source_shape_json.is_array() || source_shape_json.size() != 2 ||
      !source_shape_json.at(0).is_number_integer() ||
      !source_shape_json.at(1).is_number_integer() ||
      source_shape_json.at(0).get<std::int64_t>() != 18 ||
      source_shape_json.at(1).get<std::int64_t>() != 768) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        source_shape_path,
        "Local AR position embedding source shape must be [18, 768]");
  }
  const TensorDataType position_embedding_dtype = parse_dtype(
      member(position_embedding_json, "dtype"),
      stage,
      child_path(position_embedding_path, "dtype"));
  if (position_embedding_dtype != TensorDataType::bf16) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(position_embedding_path, "dtype"),
        "Local AR position embedding must use BF16");
  }
  constexpr std::string_view canonical_position_table_sha256 =
      "1db63ebd4ceffba52e03cf67c9d186f3b7e38bb0c6eb9056a93ceeb55a4a695e";
  const std::string position_table_sha256 = parse_sha256(
      member(position_embedding_json, "source_table_sha256"),
      stage,
      child_path(position_embedding_path, "source_table_sha256"));
  if (position_table_sha256 != canonical_position_table_sha256) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(position_embedding_path, "source_table_sha256"),
        "canonical Sofia v1 Local AR requires the accepted [18,768] BF16 position-embedding table");
  }

  return LocalArManifest{
      .engine_name = parse_identifier(
          member(value, "engine_name"), stage, child_path(path, "engine_name")),
      .execution = execution,
      .iterations = iterations,
      .positions = std::move(positions),
      .position_embedding_kind = position_embedding_kind,
      .position_embedding_positions = std::move(embedding_positions),
      .position_embedding_source_shape = {18, 768},
      .position_embedding_dtype = position_embedding_dtype,
      .position_embedding_source_table_sha256 =
          std::move(position_table_sha256),
      .codebooks_per_frame = parse_positive_integer<std::uint32_t>(
          member(value, "codebooks_per_frame"),
          stage,
          child_path(path, "codebooks_per_frame")),
      .frames_per_decoder_step = parse_positive_integer<std::uint32_t>(
          member(value, "frames_per_decoder_step"),
          stage,
          child_path(path, "frames_per_decoder_step")),
      .sampling_plugin_name = parse_identifier(
          member(value, "sampling_plugin_name"),
          stage,
          child_path(path, "sampling_plugin_name")),
      .invalid_rows_encoding = invalid_rows_encoding,
      .no_eos_frame_index = -1,
  };
}

[[nodiscard]] CodecStateBinding canonical_codec_state_binding(
    std::string logical_name,
    const std::int64_t channels,
    const std::int64_t history_samples) {
  const std::string input_binding = "state_in." + logical_name;
  const std::string output_binding = "state_out." + logical_name;
  return CodecStateBinding{
      .logical_name = std::move(logical_name),
      .dtype = TensorDataType::fp32,
      .shape = {1, channels, history_samples},
      .initial_output_binding = output_binding,
      .steady_input_binding = input_binding,
      .steady_output_binding = output_binding,
      .tail_input_binding = input_binding,
      .tail_output_binding = output_binding,
  };
}

[[nodiscard]] std::vector<CodecStateBinding>
canonical_codec_state_bindings() {
  constexpr std::array<std::int64_t, 5> stage_channels{
      432, 216, 108, 54, 27};
  constexpr std::array<std::int64_t, 5> stage_overlaps{8, 8, 4, 2, 2};
  constexpr std::array<std::int64_t, 3> branch_kernels{3, 7, 11};
  constexpr std::array<std::int64_t, 3> block_dilations{1, 3, 5};

  std::vector<CodecStateBinding> result;
  result.reserve(97);
  result.push_back(
      canonical_codec_state_binding("pre_conv.input_history", 32, 6));
  for (std::size_t stage = 0; stage < stage_channels.size(); ++stage) {
    result.push_back(canonical_codec_state_binding(
        "upsample_convs." + std::to_string(stage) + ".pending_overlap",
        stage_channels.at(stage),
        stage_overlaps.at(stage)));
  }
  for (std::size_t stage = 0; stage < stage_channels.size(); ++stage) {
    for (std::size_t branch = 0; branch < branch_kernels.size(); ++branch) {
      const std::int64_t kernel_size = branch_kernels.at(branch);
      for (std::size_t block = 0; block < block_dilations.size(); ++block) {
        const std::string prefix =
            "residual_layers." + std::to_string(stage) + ".branches." +
            std::to_string(branch) + ".residual_blocks." +
            std::to_string(block);
        result.push_back(canonical_codec_state_binding(
            prefix + ".input_conv.input_history",
            stage_channels.at(stage),
            (kernel_size - 1) * block_dilations.at(block)));
        result.push_back(canonical_codec_state_binding(
            prefix + ".skip_conv.input_history",
            stage_channels.at(stage),
            kernel_size - 1));
      }
    }
  }
  result.push_back(
      canonical_codec_state_binding("post_conv.input_history", 27, 2));
  if (result.size() != 97) {
    fail(
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        "/codec/state_bindings",
        "internal canonical NanoCodec state registry is incomplete");
  }
  return result;
}

void require_canonical_codec_state_bindings(
    const std::vector<CodecStateBinding>& actual,
    const std::string& path) {
  const std::vector<CodecStateBinding> expected =
      canonical_codec_state_bindings();
  if (actual.size() != expected.size()) {
    fail(
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        path,
        "canonical v1 NanoCodec requires exactly 97 explicit state bindings");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const CodecStateBinding& actual_state = actual.at(index);
    const CodecStateBinding& expected_state = expected.at(index);
    const std::string binding_path = index_path(path, index);
    const auto require_equal =
        [&binding_path](const bool equal, const std::string_view field) {
          if (!equal) {
            fail(
                ManifestStage::codec,
                ManifestErrorCode::invariant_violation,
                child_path(binding_path, field),
                "NanoCodec state binding differs from the canonical v1 registry");
          }
        };
    require_equal(
        actual_state.logical_name == expected_state.logical_name,
        "logical_name");
    require_equal(actual_state.dtype == expected_state.dtype, "dtype");
    require_equal(actual_state.shape == expected_state.shape, "shape");
    require_equal(
        actual_state.initial_output_binding ==
            expected_state.initial_output_binding,
        "initial_output_binding");
    require_equal(
        actual_state.steady_input_binding ==
            expected_state.steady_input_binding,
        "steady_input_binding");
    require_equal(
        actual_state.steady_output_binding ==
            expected_state.steady_output_binding,
        "steady_output_binding");
    require_equal(
        actual_state.tail_input_binding ==
            expected_state.tail_input_binding,
        "tail_input_binding");
    require_equal(
        actual_state.tail_output_binding ==
            expected_state.tail_output_binding,
        "tail_output_binding");
  }
}

[[nodiscard]] CodecManifest parse_codec(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::codec;
  require_exact_keys(
      value,
      stage,
      path,
      {"initial_engine_name",
       "steady_engine_name",
       "tail_engine_name",
       "sample_rate_hz",
       "hop_length_samples",
       "channels",
       "pcm_format",
       "stateful",
       "initial_frames",
       "steady_frames",
       "tail_min_frames",
       "tail_max_frames",
       "eos_frame_is_audio",
       "zero_frame_finalization",
       "state_bindings"});

  const std::string pcm_format = parse_nonempty_string(
      member(value, "pcm_format"), stage, child_path(path, "pcm_format"));
  if (pcm_format != "f32le") {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "pcm_format"),
        "PCM format must be 'f32le'");
  }
  const bool stateful =
      parse_boolean(member(value, "stateful"), stage, child_path(path, "stateful"));
  if (!stateful) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "stateful"),
        "NanoCodec must use stateful decoding");
  }
  const std::uint32_t initial_frames = parse_positive_integer<std::uint32_t>(
      member(value, "initial_frames"), stage, child_path(path, "initial_frames"));
  const std::uint32_t steady_frames = parse_positive_integer<std::uint32_t>(
      member(value, "steady_frames"), stage, child_path(path, "steady_frames"));
  const std::uint32_t tail_min_frames = parse_positive_integer<std::uint32_t>(
      member(value, "tail_min_frames"), stage, child_path(path, "tail_min_frames"));
  const std::uint32_t tail_max_frames = parse_positive_integer<std::uint32_t>(
      member(value, "tail_max_frames"), stage, child_path(path, "tail_max_frames"));
  if (initial_frames != 4 || steady_frames != 8 || tail_min_frames != 1 ||
      tail_max_frames != 8) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "NanoCodec schedule must be initial=4, steady=8, tail=1..8");
  }
  const bool eos_frame_is_audio = parse_boolean(
      member(value, "eos_frame_is_audio"),
      stage,
      child_path(path, "eos_frame_is_audio"));
  if (eos_frame_is_audio) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "eos_frame_is_audio"),
        "AUDIO_EOS is a control token and must not be decoded as an audio frame");
  }
  const std::string zero_frame_finalization = parse_identifier(
      member(value, "zero_frame_finalization"),
      stage,
      child_path(path, "zero_frame_finalization"));
  if (zero_frame_finalization !=
      "control_marker_without_codec_invocation") {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "zero_frame_finalization"),
        "residual zero frames must emit a FINAL control marker without invoking NanoCodec");
  }
  const std::uint32_t channels = parse_positive_integer<std::uint32_t>(
      member(value, "channels"), stage, child_path(path, "channels"));
  if (channels != 1) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "channels"),
        "schema version 1 supports exactly one audio channel");
  }

  const Json& state_bindings_json = member(value, "state_bindings");
  const std::string state_bindings_path = child_path(path, "state_bindings");
  if (!state_bindings_json.is_array() || state_bindings_json.size() != 97) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        state_bindings_path,
        "canonical v1 NanoCodec requires exactly 97 explicit state bindings");
  }
  std::vector<CodecStateBinding> state_bindings;
  std::unordered_set<std::string> logical_names;
  std::array<std::unordered_set<std::string>, 5> engine_binding_names;
  state_bindings.reserve(state_bindings_json.size());
  for (std::size_t index = 0; index < state_bindings_json.size(); ++index) {
    const Json& binding_json = state_bindings_json.at(index);
    const std::string binding_path = index_path(state_bindings_path, index);
    require_exact_keys(
        binding_json,
        stage,
        binding_path,
        {"logical_name",
         "dtype",
         "shape",
         "initial_output_binding",
         "steady_input_binding",
         "steady_output_binding",
         "tail_input_binding",
         "tail_output_binding"});
    const std::string logical_name = parse_identifier(
        member(binding_json, "logical_name"),
        stage,
        child_path(binding_path, "logical_name"));
    if (!logical_names.emplace(logical_name).second) {
      fail(
          stage,
          ManifestErrorCode::duplicate_value,
          child_path(binding_path, "logical_name"),
          "codec state logical name is duplicated");
    }

    const std::array<std::string_view, 5> binding_fields{
        "initial_output_binding",
        "steady_input_binding",
        "steady_output_binding",
        "tail_input_binding",
        "tail_output_binding",
    };
    std::array<std::string, 5> names;
    for (std::size_t field_index = 0; field_index < binding_fields.size();
         ++field_index) {
      names.at(field_index) = parse_identifier(
          member(binding_json, binding_fields.at(field_index)),
          stage,
          child_path(binding_path, binding_fields.at(field_index)));
      if (!engine_binding_names.at(field_index)
               .emplace(names.at(field_index))
               .second) {
        fail(
            stage,
            ManifestErrorCode::duplicate_value,
            child_path(binding_path, binding_fields.at(field_index)),
            "codec state binding name is duplicated in one engine direction");
      }
    }

    state_bindings.push_back(CodecStateBinding{
        .logical_name = logical_name,
        .dtype =
            parse_dtype(member(binding_json, "dtype"), stage, child_path(binding_path, "dtype")),
        .shape = parse_declared_shape(
            member(binding_json, "shape"), stage, child_path(binding_path, "shape")),
        .initial_output_binding = std::move(names.at(0)),
        .steady_input_binding = std::move(names.at(1)),
        .steady_output_binding = std::move(names.at(2)),
        .tail_input_binding = std::move(names.at(3)),
        .tail_output_binding = std::move(names.at(4)),
    });
  }
  require_canonical_codec_state_bindings(
      state_bindings,
      state_bindings_path);

  CodecManifest manifest{
      .initial_engine_name = parse_identifier(
          member(value, "initial_engine_name"),
          stage,
          child_path(path, "initial_engine_name")),
      .steady_engine_name = parse_identifier(
          member(value, "steady_engine_name"),
          stage,
          child_path(path, "steady_engine_name")),
      .tail_engine_name = parse_identifier(
          member(value, "tail_engine_name"),
          stage,
          child_path(path, "tail_engine_name")),
      .sample_rate_hz = parse_positive_integer<std::uint32_t>(
          member(value, "sample_rate_hz"), stage, child_path(path, "sample_rate_hz")),
      .hop_length_samples = parse_positive_integer<std::uint32_t>(
          member(value, "hop_length_samples"),
          stage,
          child_path(path, "hop_length_samples")),
      .channels = channels,
      .pcm_format = pcm_format,
      .stateful = stateful,
      .initial_frames = initial_frames,
      .steady_frames = steady_frames,
      .tail_min_frames = tail_min_frames,
      .tail_max_frames = tail_max_frames,
      .eos_frame_is_audio = eos_frame_is_audio,
      .zero_frame_finalization = zero_frame_finalization,
      .state_bindings = std::move(state_bindings),
  };
  if (manifest.sample_rate_hz != 22050 ||
      manifest.hop_length_samples != 1024) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "canonical v1 NanoCodec requires 22050 Hz and a 1024-sample hop");
  }
  return manifest;
}

[[nodiscard]] LimitsManifest parse_limits(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::limits;
  require_exact_keys(
      value,
      stage,
      path,
      {"maximum_text_tokens",
       "maximum_decoder_steps",
       "maximum_audio_frames",
       "maximum_sessions",
       "maximum_concurrent_requests",
       "pcm_ring_capacity_frames",
       "maximum_workspace_bytes",
       "maximum_device_memory_bytes",
       "maximum_bundle_snapshot_bytes"});

  LimitsManifest limits{
      .maximum_text_tokens = parse_positive_integer<std::uint32_t>(
          member(value, "maximum_text_tokens"),
          stage,
          child_path(path, "maximum_text_tokens")),
      .maximum_decoder_steps = parse_positive_integer<std::uint32_t>(
          member(value, "maximum_decoder_steps"),
          stage,
          child_path(path, "maximum_decoder_steps")),
      .maximum_audio_frames = parse_positive_integer<std::uint32_t>(
          member(value, "maximum_audio_frames"),
          stage,
          child_path(path, "maximum_audio_frames")),
      .maximum_sessions = parse_positive_integer<std::uint32_t>(
          member(value, "maximum_sessions"),
          stage,
          child_path(path, "maximum_sessions")),
      .maximum_concurrent_requests = parse_positive_integer<std::uint32_t>(
          member(value, "maximum_concurrent_requests"),
          stage,
          child_path(path, "maximum_concurrent_requests")),
      .pcm_ring_capacity_frames = parse_positive_integer<std::uint32_t>(
          member(value, "pcm_ring_capacity_frames"),
          stage,
          child_path(path, "pcm_ring_capacity_frames")),
      .maximum_workspace_bytes = parse_positive_integer<std::uint64_t>(
          member(value, "maximum_workspace_bytes"),
          stage,
          child_path(path, "maximum_workspace_bytes")),
      .maximum_device_memory_bytes = parse_positive_integer<std::uint64_t>(
          member(value, "maximum_device_memory_bytes"),
          stage,
          child_path(path, "maximum_device_memory_bytes")),
      .maximum_bundle_snapshot_bytes =
          parse_positive_integer<std::uint64_t>(
              member(value, "maximum_bundle_snapshot_bytes"),
              stage,
              child_path(path, "maximum_bundle_snapshot_bytes")),
  };
  if (limits.maximum_concurrent_requests !=
      kMaximumActiveRequestsPerSession) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "maximum_concurrent_requests"),
        "schema version 1 requires one active request per session");
  }
  if (limits.pcm_ring_capacity_frames != 8) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "pcm_ring_capacity_frames"),
        "schema version 1 requires an 8-frame PCM emission ring");
  }
  if (limits.maximum_audio_frames < limits.pcm_ring_capacity_frames) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        child_path(path, "maximum_audio_frames"),
        "maximum_audio_frames must cover the PCM emission ring");
  }
  if (limits.maximum_bundle_snapshot_bytes >
      kMaximumBundleSnapshotBytes) {
    fail(
        stage,
        ManifestErrorCode::size_limit_exceeded,
        child_path(path, "maximum_bundle_snapshot_bytes"),
        "bundle snapshot budget exceeds the runtime hard limit");
  }
  return limits;
}

[[nodiscard]] GoldenReceiptManifest parse_golden_receipt(
    const Json& value,
    const std::string& path) {
  constexpr ManifestStage stage = ManifestStage::golden_receipt;
  require_exact_keys(
      value,
      stage,
      path,
      {"receipt_version",
       "path",
       "sha256",
       "size_bytes",
       "created_at_utc",
       "normalized_text_sha256",
       "token_ids_sha256",
       "baked_context_sha256",
       "seed",
       "decoder_tokens_sha256",
       "codec_codes_sha256",
       "codec_frame_count",
       "pcm_f32le_sha256",
       "sample_count",
       "initial_frames",
       "steady_frames",
       "tail_min_frames",
       "tail_max_frames",
       "eos_frame_is_audio",
       "zero_frame_finalization"});
  const std::uint32_t receipt_version = parse_positive_integer<std::uint32_t>(
      member(value, "receipt_version"), stage, child_path(path, "receipt_version"));
  if (receipt_version != 1) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        child_path(path, "receipt_version"),
        "receipt_version must be 1");
  }

  return GoldenReceiptManifest{
      .receipt_version = receipt_version,
      .path =
          parse_relative_path(member(value, "path"), stage, child_path(path, "path")),
      .sha256 =
          parse_sha256(member(value, "sha256"), stage, child_path(path, "sha256")),
      .size_bytes = parse_positive_integer<std::uint64_t>(
          member(value, "size_bytes"),
          stage,
          child_path(path, "size_bytes")),
      .created_at_utc = parse_rfc3339_utc(
          member(value, "created_at_utc"), stage, child_path(path, "created_at_utc")),
      .normalized_text_sha256 = parse_sha256(
          member(value, "normalized_text_sha256"),
          stage,
          child_path(path, "normalized_text_sha256")),
      .token_ids_sha256 = parse_sha256(
          member(value, "token_ids_sha256"),
          stage,
          child_path(path, "token_ids_sha256")),
      .baked_context_sha256 = parse_sha256(
          member(value, "baked_context_sha256"),
          stage,
          child_path(path, "baked_context_sha256")),
      .seed = parse_unsigned_integer<std::uint32_t>(
          member(value, "seed"), stage, child_path(path, "seed")),
      .decoder_tokens_sha256 = parse_sha256(
          member(value, "decoder_tokens_sha256"),
          stage,
          child_path(path, "decoder_tokens_sha256")),
      .codec_codes_sha256 = parse_sha256(
          member(value, "codec_codes_sha256"),
          stage,
          child_path(path, "codec_codes_sha256")),
      .codec_frame_count = parse_positive_integer<std::uint64_t>(
          member(value, "codec_frame_count"),
          stage,
          child_path(path, "codec_frame_count")),
      .pcm_f32le_sha256 = parse_sha256(
          member(value, "pcm_f32le_sha256"),
          stage,
          child_path(path, "pcm_f32le_sha256")),
      .sample_count = parse_positive_integer<std::uint64_t>(
          member(value, "sample_count"), stage, child_path(path, "sample_count")),
      .initial_frames = parse_positive_integer<std::uint32_t>(
          member(value, "initial_frames"), stage, child_path(path, "initial_frames")),
      .steady_frames = parse_positive_integer<std::uint32_t>(
          member(value, "steady_frames"), stage, child_path(path, "steady_frames")),
      .tail_min_frames = parse_positive_integer<std::uint32_t>(
          member(value, "tail_min_frames"), stage, child_path(path, "tail_min_frames")),
      .tail_max_frames = parse_positive_integer<std::uint32_t>(
          member(value, "tail_max_frames"), stage, child_path(path, "tail_max_frames")),
      .eos_frame_is_audio = parse_boolean(
          member(value, "eos_frame_is_audio"),
          stage,
          child_path(path, "eos_frame_is_audio")),
      .zero_frame_finalization = parse_identifier(
          member(value, "zero_frame_finalization"),
          stage,
          child_path(path, "zero_frame_finalization")),
  };
}

[[nodiscard]] const TensorSpec* require_tensor(
    const EngineManifest& engine,
    const bool input,
    const std::string& name,
    const ManifestStage stage,
    const std::string& path) {
  const std::vector<TensorSpec>* tensors =
      input ? &engine.inputs : &engine.outputs;
  const auto found = std::find_if(
      tensors->begin(), tensors->end(), [&name](const TensorSpec& tensor) {
        return tensor.name == name;
      });
  if (found == tensors->end()) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        std::string(input ? "input" : "output") +
            " binding is absent from engine '" + engine.name + "'");
  }
  return &*found;
}

[[nodiscard]] std::string shape_text(
    const std::vector<std::int64_t>& shape) {
  std::ostringstream text;
  text << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      text << ',';
    }
    text << shape.at(index);
  }
  text << ']';
  return text.str();
}

void require_tensor_contract(
    const EngineManifest& engine,
    const bool input,
    const std::string& name,
    const TensorDataType dtype,
    const std::vector<std::vector<std::int64_t>>& accepted_shapes,
    const ManifestStage stage,
    const std::string& path) {
  const TensorSpec* tensor =
      require_tensor(engine, input, name, stage, path);
  if (tensor->dtype != dtype) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "binding '" + name + "' has dtype '" +
            std::string(to_string(tensor->dtype)) + "', expected '" +
            std::string(to_string(dtype)) + "'");
  }
  if (std::find(accepted_shapes.begin(), accepted_shapes.end(), tensor->shape) ==
      accepted_shapes.end()) {
    std::ostringstream expected;
    for (std::size_t index = 0; index < accepted_shapes.size(); ++index) {
      if (index != 0) {
        expected << " or ";
      }
      expected << shape_text(accepted_shapes.at(index));
    }
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "binding '" + name + "' has shape " + shape_text(tensor->shape) +
            ", expected " + expected.str());
  }
}

void require_exact_tensor_names(
    const EngineManifest& engine,
    const bool input,
    const std::vector<std::string>& expected_names,
    const std::string& path) {
  const std::vector<TensorSpec>& tensors =
      input ? engine.inputs : engine.outputs;
  std::unordered_set<std::string> expected(
      expected_names.begin(), expected_names.end());
  if (expected.size() != expected_names.size()) {
    fail(
        ManifestStage::engine,
        ManifestErrorCode::invariant_violation,
        path,
        "canonical binding list itself contains a duplicate name");
  }
  for (const TensorSpec& tensor : tensors) {
    if (!expected.contains(tensor.name)) {
      fail(
          ManifestStage::engine,
          ManifestErrorCode::unknown_field,
          path,
          "engine '" + engine.name + "' exposes non-canonical " +
              std::string(input ? "input" : "output") + " binding '" +
              tensor.name + "'");
    }
  }
  for (const std::string& expected_name : expected_names) {
    const auto found = std::find_if(
        tensors.begin(),
        tensors.end(),
        [&expected_name](const TensorSpec& tensor) {
          return tensor.name == expected_name;
        });
    if (found == tensors.end()) {
      fail(
          ManifestStage::engine,
          ManifestErrorCode::missing_field,
          path,
          "engine '" + engine.name + "' is missing canonical " +
              std::string(input ? "input" : "output") + " binding '" +
              expected_name + "'");
    }
  }
}

void require_named_engine_role(
    const RuntimeBundleManifest& manifest,
    const std::string& engine_name,
    const EngineRole role,
    const ManifestStage stage,
    const std::string& path) {
  const auto found = std::find_if(
      manifest.engines.begin(),
      manifest.engines.end(),
      [&engine_name](const EngineManifest& engine) {
        return engine.name == engine_name;
      });
  if (found == manifest.engines.end()) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "referenced engine does not exist");
  }
  if (found->role != role) {
    fail(
        stage,
        ManifestErrorCode::invariant_violation,
        path,
        "referenced engine has role '" + std::string(to_string(found->role)) +
            "', expected '" + std::string(to_string(role)) + "'");
  }
}

void validate_canonical_engine_contracts(
    const RuntimeBundleManifest& manifest) {
  const EngineManifest& text_encoder =
      require_engine(manifest, EngineRole::text_encoder);
  require_exact_tensor_names(
      text_encoder,
      true,
      {"text_token_ids", "text_mask"},
      "/engines/text_encoder/inputs");
  require_exact_tensor_names(
      text_encoder,
      false,
      {"text_condition"},
      "/engines/text_encoder/outputs");
  require_tensor_contract(
      text_encoder,
      true,
      "text_token_ids",
      TensorDataType::int32,
      {{{1, -1}}},
      ManifestStage::engine,
      "/engines/text_encoder/inputs/text_token_ids");
  require_tensor_contract(
      text_encoder,
      true,
      "text_mask",
      TensorDataType::boolean,
      {{{1, -1}}},
      ManifestStage::engine,
      "/engines/text_encoder/inputs/text_mask");
  require_tensor_contract(
      text_encoder,
      false,
      "text_condition",
      TensorDataType::bf16,
      {{{1, -1, 768}}},
      ManifestStage::engine,
      "/engines/text_encoder/outputs/text_condition");

  const EngineManifest& prefill =
      require_engine(manifest, EngineRole::main_decoder_prefill);
  std::vector<std::string> prefill_outputs{"last_hidden", "alignment"};
  for (const KvLayerBindings& layer : manifest.kv_cache.layer_bindings) {
    prefill_outputs.push_back(layer.prefill_self_key_output);
    prefill_outputs.push_back(layer.prefill_self_value_output);
    prefill_outputs.push_back(layer.prefill_self_mask_output);
    prefill_outputs.push_back(layer.prefill_cross_key_output);
    prefill_outputs.push_back(layer.prefill_cross_value_output);
  }
  require_exact_tensor_names(
      prefill,
      true,
      {"condition", "condition_mask"},
      "/engines/main_decoder_prefill/inputs");
  require_exact_tensor_names(
      prefill,
      false,
      prefill_outputs,
      "/engines/main_decoder_prefill/outputs");
  require_tensor_contract(
      prefill,
      true,
      "condition",
      TensorDataType::bf16,
      {{{2, -1, 768}}},
      ManifestStage::engine,
      "/engines/main_decoder_prefill/inputs/condition");
  require_tensor_contract(
      prefill,
      true,
      "condition_mask",
      TensorDataType::boolean,
      {{{2, -1}}},
      ManifestStage::engine,
      "/engines/main_decoder_prefill/inputs/condition_mask");
  require_tensor_contract(
      prefill,
      false,
      "last_hidden",
      TensorDataType::bf16,
      {{{2, 1, 768}}},
      ManifestStage::engine,
      "/engines/main_decoder_prefill/outputs/last_hidden");
  require_tensor_contract(
      prefill,
      false,
      "alignment",
      TensorDataType::bf16,
      {{{2, -1}}},
      ManifestStage::engine,
      "/engines/main_decoder_prefill/outputs/alignment");

  const EngineManifest& step =
      require_engine(manifest, EngineRole::main_decoder_step);
  std::vector<std::string> step_inputs{
      "previous_codec_tokens", "position", "condition_mask", "alignment_prior"};
  std::vector<std::string> step_outputs{"decoder_hidden", "alignment"};
  for (const KvLayerBindings& layer : manifest.kv_cache.layer_bindings) {
    step_inputs.push_back(layer.step_self_key_input);
    step_inputs.push_back(layer.step_self_value_input);
    step_inputs.push_back(layer.step_self_mask_input);
    step_inputs.push_back(layer.step_cross_key_input);
    step_inputs.push_back(layer.step_cross_value_input);
    step_outputs.push_back(layer.step_self_key_output);
    step_outputs.push_back(layer.step_self_value_output);
    step_outputs.push_back(layer.step_self_mask_output);
  }
  require_exact_tensor_names(
      step, true, step_inputs, "/engines/main_decoder_step/inputs");
  require_exact_tensor_names(
      step, false, step_outputs, "/engines/main_decoder_step/outputs");
  require_tensor_contract(
      step,
      true,
      "previous_codec_tokens",
      TensorDataType::int64,
      {{{1, 8, 2}}},
      ManifestStage::engine,
      "/engines/main_decoder_step/inputs/previous_codec_tokens");
  require_tensor_contract(
      step,
      true,
      "position",
      TensorDataType::int64,
      {std::vector<std::int64_t>{}},
      ManifestStage::engine,
      "/engines/main_decoder_step/inputs/position");
  require_tensor_contract(
      step,
      true,
      "condition_mask",
      TensorDataType::boolean,
      {{{2, -1}}},
      ManifestStage::engine,
      "/engines/main_decoder_step/inputs/condition_mask");
  require_tensor_contract(
      step,
      true,
      "alignment_prior",
      TensorDataType::bf16,
      {{{2, 1, -1}}},
      ManifestStage::engine,
      "/engines/main_decoder_step/inputs/alignment_prior");
  require_tensor_contract(
      step,
      false,
      "decoder_hidden",
      TensorDataType::bf16,
      {{{2, 768}}},
      ManifestStage::engine,
      "/engines/main_decoder_step/outputs/decoder_hidden");
  require_tensor_contract(
      step,
      false,
      "alignment",
      TensorDataType::bf16,
      {{{2, -1}}},
      ManifestStage::engine,
      "/engines/main_decoder_step/outputs/alignment");

  const EngineManifest& local_ar =
      require_engine(manifest, EngineRole::local_ar_16);
  require_exact_tensor_names(
      local_ar,
      true,
      {"decoder_hidden",
       "unfinished",
       "finished",
       "forbid_eos",
       "rng_seed",
       "rng_counter"},
      "/engines/local_ar_16/inputs");
  require_exact_tensor_names(
      local_ar,
      false,
      {"codec_tokens",
       "updated_rng_counter",
       "invalid_rows",
       "end_frame_index"},
      "/engines/local_ar_16/outputs");
  const std::array<std::tuple<std::string, TensorDataType, std::vector<std::int64_t>>, 6>
      local_inputs{
          std::tuple{"decoder_hidden", TensorDataType::bf16, std::vector<std::int64_t>{2, 768}},
          std::tuple{"unfinished", TensorDataType::boolean, std::vector<std::int64_t>{1}},
          std::tuple{"finished", TensorDataType::boolean, std::vector<std::int64_t>{1}},
          std::tuple{"forbid_eos", TensorDataType::boolean, std::vector<std::int64_t>{1}},
          std::tuple{"rng_seed", TensorDataType::int64, std::vector<std::int64_t>{1}},
          std::tuple{"rng_counter", TensorDataType::int64, std::vector<std::int64_t>{1}},
      };
  for (const auto& [name, dtype, shape] : local_inputs) {
    require_tensor_contract(
        local_ar,
        true,
        name,
        dtype,
        {shape},
        ManifestStage::engine,
        "/engines/local_ar_16/inputs/" + name);
  }
  const std::array<std::tuple<std::string, TensorDataType, std::vector<std::int64_t>>, 4>
      local_outputs{
          std::tuple{"codec_tokens", TensorDataType::int64, std::vector<std::int64_t>{1, 8, 2}},
          std::tuple{"updated_rng_counter", TensorDataType::int64, std::vector<std::int64_t>{1}},
          std::tuple{"invalid_rows", TensorDataType::int32, std::vector<std::int64_t>{1}},
          std::tuple{"end_frame_index", TensorDataType::int32, std::vector<std::int64_t>{1}},
      };
  for (const auto& [name, dtype, shape] : local_outputs) {
    require_tensor_contract(
        local_ar,
        false,
        name,
        dtype,
        {shape},
        ManifestStage::engine,
        "/engines/local_ar_16/outputs/" + name);
  }

  const auto validate_codec_engine =
      [&manifest](
          const EngineRole role,
          const std::uint32_t frames,
          const bool dynamic_frames,
          const bool has_state_inputs,
          const std::string_view route,
          const auto input_binding,
          const auto output_binding) {
        const EngineManifest& engine = require_engine(manifest, role);
        std::vector<std::string> inputs{"codec_tokens"};
        std::vector<std::string> outputs{"pcm", "valid_sample_length"};
        for (const CodecStateBinding& state : manifest.codec.state_bindings) {
          if (has_state_inputs) {
            inputs.push_back(input_binding(state));
          }
          outputs.push_back(output_binding(state));
        }
        const std::string base = "/engines/" + std::string(route);
        require_exact_tensor_names(engine, true, inputs, base + "/inputs");
        require_exact_tensor_names(engine, false, outputs, base + "/outputs");
        const std::int64_t frame_dimension =
            dynamic_frames ? -1 : static_cast<std::int64_t>(frames);
        const std::int64_t sample_dimension =
            dynamic_frames
                ? -1
                : static_cast<std::int64_t>(
                      frames * manifest.codec.hop_length_samples);
        require_tensor_contract(
            engine,
            true,
            "codec_tokens",
            TensorDataType::int64,
            {{{1, 8, frame_dimension}}},
            ManifestStage::engine,
            base + "/inputs/codec_tokens");
        require_tensor_contract(
            engine,
            false,
            "pcm",
            TensorDataType::fp32,
            {{{1, sample_dimension}}},
            ManifestStage::engine,
            base + "/outputs/pcm");
        require_tensor_contract(
            engine,
            false,
            "valid_sample_length",
            TensorDataType::int64,
            {{{1}}},
            ManifestStage::engine,
            base + "/outputs/valid_sample_length");
        for (const CodecStateBinding& state :
             manifest.codec.state_bindings) {
          if (has_state_inputs) {
            require_tensor_contract(
                engine,
                true,
                input_binding(state),
                state.dtype,
                {state.shape},
                ManifestStage::engine,
                base + "/inputs/" + input_binding(state));
          }
          require_tensor_contract(
              engine,
              false,
              output_binding(state),
              state.dtype,
              {state.shape},
              ManifestStage::engine,
              base + "/outputs/" + output_binding(state));
        }
      };
  validate_codec_engine(
      EngineRole::nanocodec_initial_4,
      4,
      false,
      false,
      "nanocodec_initial_4",
      [](const CodecStateBinding&) { return std::string{}; },
      [](const CodecStateBinding& state) {
        return state.initial_output_binding;
      });
  validate_codec_engine(
      EngineRole::nanocodec_steady_8,
      8,
      false,
      true,
      "nanocodec_steady_8",
      [](const CodecStateBinding& state) {
        return state.steady_input_binding;
      },
      [](const CodecStateBinding& state) {
        return state.steady_output_binding;
      });
  validate_codec_engine(
      EngineRole::nanocodec_tail_1_8,
      0,
      true,
      true,
      "nanocodec_tail_1_8",
      [](const CodecStateBinding& state) {
        return state.tail_input_binding;
      },
      [](const CodecStateBinding& state) {
        return state.tail_output_binding;
      });
}

struct CanonicalProfileRange {
  std::string tensor_name;
  std::vector<std::int64_t> minimum;
  std::vector<std::int64_t> optimum;
  std::vector<std::int64_t> maximum;
};

[[nodiscard]] const TensorShapeRange& require_profile_range(
    const OptimizationProfile& profile,
    const std::string& tensor_name,
    const std::string& path) {
  const auto found = std::find_if(
      profile.input_shapes.begin(),
      profile.input_shapes.end(),
      [&tensor_name](const TensorShapeRange& range) {
        return range.tensor_name == tensor_name;
      });
  if (found == profile.input_shapes.end()) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::missing_field,
        path,
        "canonical optimization-profile shape range is missing");
  }
  return *found;
}

[[nodiscard]] const TensorValueRange& require_profile_value_range(
    const OptimizationProfile& profile,
    const std::string& tensor_name,
    const std::string& path) {
  const auto found = std::find_if(
      profile.input_values.begin(),
      profile.input_values.end(),
      [&tensor_name](const TensorValueRange& range) {
        return range.tensor_name == tensor_name;
      });
  if (found == profile.input_values.end()) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::missing_field,
        path,
        "canonical optimization-profile value range is missing");
  }
  return *found;
}

void require_canonical_profile(
    const EngineManifest& engine,
    const std::string_view expected_name,
    const std::vector<CanonicalProfileRange>& expected_ranges,
    const std::string& path) {
  if (engine.profiles.size() != 1) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        path,
        "schema version 1 requires exactly one optimization profile per engine");
  }
  const OptimizationProfile& profile = engine.profiles.front();
  if (profile.name != expected_name) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        child_path(index_path(path, 0), "name"),
        "optimization profile has a non-canonical role-specific name");
  }
  const std::string ranges_path =
      child_path(index_path(path, 0), "input_shapes");
  if (profile.input_shapes.size() != expected_ranges.size()) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        ranges_path,
        "optimization profile does not contain the canonical dynamic-input set");
  }
  for (const TensorShapeRange& actual : profile.input_shapes) {
    const auto known = std::find_if(
        expected_ranges.begin(),
        expected_ranges.end(),
        [&actual](const CanonicalProfileRange& expected) {
          return expected.tensor_name == actual.tensor_name;
        });
    if (known == expected_ranges.end()) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::unknown_field,
          child_path(ranges_path, actual.tensor_name),
          "optimization profile contains a non-canonical tensor range");
    }
  }
  for (const CanonicalProfileRange& expected : expected_ranges) {
    const std::string range_path =
        child_path(ranges_path, expected.tensor_name);
    const TensorShapeRange& actual =
        require_profile_range(profile, expected.tensor_name, range_path);
    if (actual.minimum != expected.minimum ||
        actual.optimum != expected.optimum ||
        actual.maximum != expected.maximum) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::invariant_violation,
          range_path,
          "optimization-profile min/opt/max does not match the canonical v1 range");
    }
  }
  if (engine.role == EngineRole::main_decoder_step) {
    if (profile.input_values.size() != 1) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::invariant_violation,
          child_path(index_path(path, 0), "input_values"),
          "Main Decoder step must authenticate one position value range");
    }
    const std::string position_name = "position";
    const std::string position_path = child_path(
        child_path(index_path(path, 0), "input_values"),
        position_name);
    const TensorValueRange& position = require_profile_value_range(
        profile,
        position_name,
        position_path);
    if (position.minimum != std::vector<std::int64_t>{218} ||
        position.optimum != std::vector<std::int64_t>{342} ||
        position.maximum != std::vector<std::int64_t>{466}) {
      fail(
          ManifestStage::tensor,
          ManifestErrorCode::invariant_violation,
          position_path,
          "Main Decoder step position values must be min/opt/max 218/342/466");
    }
  } else if (!profile.input_values.empty()) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        child_path(index_path(path, 0), "input_values"),
        "only Main Decoder step may expose a shape-input value range");
  }
}

struct ProfileAxisBounds {
  std::int64_t minimum;
  std::int64_t optimum;
  std::int64_t maximum;

  [[nodiscard]] bool operator==(const ProfileAxisBounds&) const = default;
};

[[nodiscard]] ProfileAxisBounds profile_axis_bounds(
    const EngineManifest& engine,
    const std::string& tensor_name,
    const std::size_t axis,
    const std::string& path) {
  if (engine.profiles.empty()) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::missing_field,
        path,
        "engine has no optimization profile");
  }
  const TensorShapeRange& range =
      require_profile_range(engine.profiles.front(), tensor_name, path);
  if (axis >= range.minimum.size() || axis >= range.optimum.size() ||
      axis >= range.maximum.size()) {
    fail(
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        path,
        "profile axis is outside the declared shape rank");
  }
  return ProfileAxisBounds{
      .minimum = range.minimum.at(axis),
      .optimum = range.optimum.at(axis),
      .maximum = range.maximum.at(axis),
  };
}

void validate_connected_text_profile_axes(
    const RuntimeBundleManifest& manifest) {
  const ProfileAxisBounds canonical_text_span = profile_axis_bounds(
      require_engine(manifest, EngineRole::text_encoder),
      "text_token_ids",
      1,
      "/engines/text_encoder/profiles/0/input_shapes/text_token_ids");
  const auto require_same_text_span =
      [&canonical_text_span](
          const EngineManifest& engine,
          const std::string& tensor_name,
          const std::size_t axis,
          const std::string& path) {
        if (profile_axis_bounds(engine, tensor_name, axis, path) !=
            canonical_text_span) {
          fail(
              ManifestStage::tensor,
              ManifestErrorCode::invariant_violation,
              path,
              "min/opt/max text length T must exactly match the Text Encoder profile");
        }
      };

  const EngineManifest& text_encoder =
      require_engine(manifest, EngineRole::text_encoder);
  require_same_text_span(
      text_encoder,
      "text_mask",
      1,
      "/engines/text_encoder/profiles/0/input_shapes/text_mask");

  const EngineManifest& prefill =
      require_engine(manifest, EngineRole::main_decoder_prefill);
  require_same_text_span(
      prefill,
      "condition",
      1,
      "/engines/main_decoder_prefill/profiles/0/input_shapes/condition");
  require_same_text_span(
      prefill,
      "condition_mask",
      1,
      "/engines/main_decoder_prefill/profiles/0/input_shapes/condition_mask");

  const EngineManifest& step =
      require_engine(manifest, EngineRole::main_decoder_step);
  require_same_text_span(
      step,
      "condition_mask",
      1,
      "/engines/main_decoder_step/profiles/0/input_shapes/condition_mask");
  require_same_text_span(
      step,
      "alignment_prior",
      2,
      "/engines/main_decoder_step/profiles/0/input_shapes/alignment_prior");
  for (const KvLayerBindings& layer : manifest.kv_cache.layer_bindings) {
    require_same_text_span(
        step,
        layer.step_cross_key_input,
        1,
        "/engines/main_decoder_step/profiles/0/input_shapes/" +
            layer.step_cross_key_input);
    require_same_text_span(
        step,
        layer.step_cross_value_input,
        1,
        "/engines/main_decoder_step/profiles/0/input_shapes/" +
            layer.step_cross_value_input);
  }
  if (canonical_text_span.maximum !=
      static_cast<std::int64_t>(manifest.limits.maximum_text_tokens)) {
    fail(
        ManifestStage::limits,
        ManifestErrorCode::invariant_violation,
        "/limits/maximum_text_tokens",
        "maximum_text_tokens must exactly match the connected engine profile maximum T");
  }
}

void validate_canonical_profile_contracts(
    const RuntimeBundleManifest& manifest) {
  require_canonical_profile(
      require_engine(manifest, EngineRole::text_encoder),
      "text_1_512",
      {
          {"text_token_ids", {1, 1}, {1, 64}, {1, 512}},
          {"text_mask", {1, 1}, {1, 64}, {1, 512}},
      },
      "/engines/text_encoder/profiles");
  require_canonical_profile(
      require_engine(manifest, EngineRole::main_decoder_prefill),
      "text_1_512",
      {
          {"condition", {2, 1, 768}, {2, 64, 768}, {2, 512, 768}},
          {"condition_mask", {2, 1}, {2, 64}, {2, 512}},
      },
      "/engines/main_decoder_prefill/profiles");

  std::vector<CanonicalProfileRange> step_ranges{
      {"condition_mask", {2, 1}, {2, 64}, {2, 512}},
      {"alignment_prior", {2, 1, 1}, {2, 1, 64}, {2, 1, 512}},
  };
  step_ranges.reserve(
      2U + (manifest.kv_cache.layer_bindings.size() * 2U));
  for (const KvLayerBindings& layer : manifest.kv_cache.layer_bindings) {
    step_ranges.push_back(
        {layer.step_cross_key_input,
         {2, 1, 1, 128},
         {2, 64, 1, 128},
         {2, 512, 1, 128}});
    step_ranges.push_back(
        {layer.step_cross_value_input,
         {2, 1, 1, 128},
         {2, 64, 1, 128},
         {2, 512, 1, 128}});
  }
  require_canonical_profile(
      require_engine(manifest, EngineRole::main_decoder_step),
      "text_1_512",
      step_ranges,
      "/engines/main_decoder_step/profiles");

  require_canonical_profile(
      require_engine(manifest, EngineRole::local_ar_16),
      "fixed",
      {},
      "/engines/local_ar_16/profiles");
  require_canonical_profile(
      require_engine(manifest, EngineRole::nanocodec_initial_4),
      "fixed",
      {},
      "/engines/nanocodec_initial_4/profiles");
  require_canonical_profile(
      require_engine(manifest, EngineRole::nanocodec_steady_8),
      "fixed",
      {},
      "/engines/nanocodec_steady_8/profiles");
  require_canonical_profile(
      require_engine(manifest, EngineRole::nanocodec_tail_1_8),
      "tail_1_8",
      {
          {"codec_tokens", {1, 8, 1}, {1, 8, 4}, {1, 8, 8}},
      },
      "/engines/nanocodec_tail_1_8/profiles");
}

void validate_cross_field_invariants(const RuntimeBundleManifest& manifest) {
  std::uint64_t total_snapshot_bytes = 0;
  const auto add_snapshot_size =
      [&total_snapshot_bytes](
          const std::uint64_t size_bytes,
          const std::string& json_pointer) {
        if (size_bytes > kMaximumBundleSnapshotBytes -
                             total_snapshot_bytes) {
          fail(
              ManifestStage::limits,
              ManifestErrorCode::size_limit_exceeded,
              json_pointer,
              "declared artifact sizes exceed the runtime bundle snapshot hard limit");
        }
        total_snapshot_bytes += size_bytes;
      };
  add_snapshot_size(
      manifest.artifacts.source_model.acceptance_receipt.size_bytes,
      "/artifacts/source_model/acceptance_receipt/size_bytes");
  add_snapshot_size(
      manifest.artifacts.export_artifact.export_receipt.size_bytes,
      "/artifacts/export/export_receipt/size_bytes");
  add_snapshot_size(
      manifest.artifacts.tokenizer.identity_receipt.size_bytes,
      "/artifacts/tokenizer/identity_receipt/size_bytes");
  add_snapshot_size(
      manifest.artifacts.plugin.file.size_bytes,
      "/artifacts/plugin/file/size_bytes");
  add_snapshot_size(
      manifest.artifacts.plugin.build_receipt.size_bytes,
      "/artifacts/plugin/build_receipt/size_bytes");
  for (std::size_t index = 0; index < manifest.engines.size(); ++index) {
    add_snapshot_size(
        manifest.engines.at(index).file.size_bytes,
        index_path("/engines", index) + "/file/size_bytes");
  }
  for (std::size_t index = 0; index < manifest.licenses.size(); ++index) {
    add_snapshot_size(
        manifest.licenses.at(index).file.size_bytes,
        index_path("/licenses", index) + "/file/size_bytes");
  }
  add_snapshot_size(
      manifest.golden_receipt.size_bytes,
      "/golden_receipt/size_bytes");
  add_snapshot_size(
      manifest.golden_fixture.size_bytes,
      "/golden_fixture/size_bytes");
  if (total_snapshot_bytes !=
      manifest.limits.maximum_bundle_snapshot_bytes) {
    fail(
        ManifestStage::limits,
        ManifestErrorCode::invariant_violation,
        "/limits/maximum_bundle_snapshot_bytes",
        "bundle snapshot budget must exactly equal the sum of all declared artifact sizes");
  }

  if (manifest.artifacts.export_artifact.baked_context_length !=
      manifest.kv_cache.prefix_length) {
    fail(
        ManifestStage::artifacts,
        ManifestErrorCode::invariant_violation,
        "/artifacts/export/baked_context_length",
        "baked export context length must exactly match KV prefix_length");
  }
  if (manifest.golden_receipt.baked_context_sha256 !=
      manifest.artifacts.export_artifact.baked_context_sha256) {
    fail(
        ManifestStage::golden_receipt,
        ManifestErrorCode::invariant_violation,
        "/golden_receipt/baked_context_sha256",
        "golden receipt baked context hash must exactly match the exported Sofia context hash");
  }
  if (manifest.artifacts.plugin.abi_version !=
      manifest.runtime.plugin_abi_version) {
    fail(
        ManifestStage::artifacts,
        ManifestErrorCode::invariant_violation,
        "/artifacts/plugin/abi_version",
        "plugin ABI must exactly match runtime fingerprint plugin ABI");
  }

  require_named_engine_role(
      manifest,
      manifest.local_ar.engine_name,
      EngineRole::local_ar_16,
      ManifestStage::local_ar,
      "/local_ar/engine_name");
  if (manifest.local_ar.sampling_plugin_name !=
      "MagpieLocalARSampling") {
    fail(
        ManifestStage::local_ar,
        ManifestErrorCode::invariant_violation,
        "/local_ar/sampling_plugin_name",
        "sampling plugin must select the canonical authenticated creator");
  }
  if (manifest.local_ar.codebooks_per_frame != 8 ||
      manifest.local_ar.frames_per_decoder_step != 2) {
    fail(
        ManifestStage::local_ar,
        ManifestErrorCode::invariant_violation,
        "/local_ar",
        "schema version 1 requires 8 codebooks and 2 frames per decoder step");
  }

  require_named_engine_role(
      manifest,
      manifest.codec.initial_engine_name,
      EngineRole::nanocodec_initial_4,
      ManifestStage::codec,
      "/codec/initial_engine_name");
  require_named_engine_role(
      manifest,
      manifest.codec.steady_engine_name,
      EngineRole::nanocodec_steady_8,
      ManifestStage::codec,
      "/codec/steady_engine_name");
  require_named_engine_role(
      manifest,
      manifest.codec.tail_engine_name,
      EngineRole::nanocodec_tail_1_8,
      ManifestStage::codec,
      "/codec/tail_engine_name");

  if (manifest.golden_receipt.initial_frames != manifest.codec.initial_frames ||
      manifest.golden_receipt.steady_frames != manifest.codec.steady_frames ||
      manifest.golden_receipt.tail_min_frames != manifest.codec.tail_min_frames ||
      manifest.golden_receipt.tail_max_frames != manifest.codec.tail_max_frames ||
      manifest.golden_receipt.eos_frame_is_audio !=
          manifest.codec.eos_frame_is_audio ||
      manifest.golden_receipt.zero_frame_finalization !=
          manifest.codec.zero_frame_finalization) {
    fail(
        ManifestStage::golden_receipt,
        ManifestErrorCode::invariant_violation,
        "/golden_receipt",
        "golden receipt stream contract must exactly match the codec contract");
  }
  if (manifest.golden_receipt.codec_frame_count >
          std::numeric_limits<std::uint64_t>::max() /
              manifest.codec.hop_length_samples ||
      manifest.golden_receipt.codec_frame_count *
              manifest.codec.hop_length_samples !=
          manifest.golden_receipt.sample_count) {
    fail(
        ManifestStage::golden_receipt,
        ManifestErrorCode::invariant_violation,
        "/golden_receipt/sample_count",
        "golden PCM sample count must equal codec_frame_count times hop_length_samples");
  }
  if (manifest.golden_receipt.codec_frame_count >
      manifest.limits.maximum_audio_frames) {
    fail(
        ManifestStage::golden_receipt,
        ManifestErrorCode::invariant_violation,
        "/golden_receipt/codec_frame_count",
        "golden codec frame count exceeds the runtime audio-frame limit");
  }
  if (manifest.limits.maximum_text_tokens != 512 ||
      manifest.limits.maximum_decoder_steps !=
          manifest.kv_cache.maximum_generated_steps ||
      static_cast<std::uint64_t>(manifest.limits.maximum_audio_frames) !=
          static_cast<std::uint64_t>(
              manifest.limits.maximum_decoder_steps) *
              static_cast<std::uint64_t>(
                  manifest.local_ar.frames_per_decoder_step)) {
    fail(
        ManifestStage::limits,
        ManifestErrorCode::invariant_violation,
        "/limits",
        "v1 limits must be text=512, decoder steps=250, and audio frames=500");
  }

  const EngineManifest& prefill =
      require_engine(manifest, EngineRole::main_decoder_prefill);
  const EngineManifest& step =
      require_engine(manifest, EngineRole::main_decoder_step);
  const std::vector<std::int64_t> self_shape{
      manifest.kv_cache.batch_size,
      manifest.kv_cache.self_cache_capacity,
      manifest.kv_cache.self_attention_heads,
      manifest.kv_cache.self_attention_head_dimension,
  };
  const std::vector<std::int64_t> self_mask_shape{
      manifest.kv_cache.batch_size,
      manifest.kv_cache.self_cache_capacity,
  };
  const std::vector<std::int64_t> cross_dynamic_shape{
      manifest.kv_cache.batch_size,
      -1,
      manifest.kv_cache.cross_attention_heads,
      manifest.kv_cache.cross_attention_head_dimension,
  };
  const std::vector<std::vector<std::int64_t>> self_shapes{self_shape};
  const std::vector<std::vector<std::int64_t>> self_mask_shapes{
      self_mask_shape};
  const std::vector<std::vector<std::int64_t>> cross_shapes{
      cross_dynamic_shape};

  for (std::size_t index = 0;
       index < manifest.kv_cache.layer_bindings.size();
       ++index) {
    const KvLayerBindings& bindings =
        manifest.kv_cache.layer_bindings.at(index);
    const std::string base =
        index_path("/kv_cache/layer_bindings", index);
    const auto check_prefill_value =
        [&](const std::string& name, const std::string_view field) {
          require_tensor_contract(
              prefill,
              false,
              name,
              manifest.kv_cache.dtype,
              self_shapes,
              ManifestStage::kv_cache,
              child_path(base, field));
        };
    const auto check_prefill_mask =
        [&](const std::string& name, const std::string_view field) {
          require_tensor_contract(
              prefill,
              false,
              name,
              manifest.kv_cache.mask_dtype,
              self_mask_shapes,
              ManifestStage::kv_cache,
              child_path(base, field));
        };
    const auto check_prefill_cross =
        [&](const std::string& name, const std::string_view field) {
          require_tensor_contract(
              prefill,
              false,
              name,
              manifest.kv_cache.dtype,
              cross_shapes,
              ManifestStage::kv_cache,
              child_path(base, field));
        };
    const auto check_step_value =
        [&](const bool input,
            const std::string& name,
            const std::string_view field) {
          require_tensor_contract(
              step,
              input,
              name,
              manifest.kv_cache.dtype,
              self_shapes,
              ManifestStage::kv_cache,
              child_path(base, field));
        };
    const auto check_step_mask =
        [&](const bool input,
            const std::string& name,
            const std::string_view field) {
          require_tensor_contract(
              step,
              input,
              name,
              manifest.kv_cache.mask_dtype,
              self_mask_shapes,
              ManifestStage::kv_cache,
              child_path(base, field));
        };
    const auto check_step_cross =
        [&](const std::string& name, const std::string_view field) {
          require_tensor_contract(
              step,
              true,
              name,
              manifest.kv_cache.dtype,
              cross_shapes,
              ManifestStage::kv_cache,
              child_path(base, field));
        };

    check_prefill_value(
        bindings.prefill_self_key_output, "prefill_self_key_output");
    check_prefill_value(
        bindings.prefill_self_value_output, "prefill_self_value_output");
    check_prefill_mask(
        bindings.prefill_self_mask_output, "prefill_self_mask_output");
    check_prefill_cross(
        bindings.prefill_cross_key_output, "prefill_cross_key_output");
    check_prefill_cross(
        bindings.prefill_cross_value_output, "prefill_cross_value_output");
    check_step_value(
        true, bindings.step_self_key_input, "step_self_key_input");
    check_step_value(
        true, bindings.step_self_value_input, "step_self_value_input");
    check_step_mask(
        true, bindings.step_self_mask_input, "step_self_mask_input");
    check_step_cross(bindings.step_cross_key_input, "step_cross_key_input");
    check_step_cross(
        bindings.step_cross_value_input, "step_cross_value_input");
    check_step_value(
        false, bindings.step_self_key_output, "step_self_key_output");
    check_step_value(
        false, bindings.step_self_value_output, "step_self_value_output");
    check_step_mask(
        false, bindings.step_self_mask_output, "step_self_mask_output");
  }

  const TensorSpec* prefill_alignment = require_tensor(
      prefill,
      false,
      manifest.alignment.prefill_output_binding,
      ManifestStage::alignment,
      "/alignment/prefill_output_binding");
  const TensorSpec* prior_input = require_tensor(
      step,
      true,
      manifest.alignment.step_prior_input_binding,
      ManifestStage::alignment,
      "/alignment/step_prior_input_binding");
  const TensorSpec* alignment_output = require_tensor(
      step,
      false,
      manifest.alignment.step_alignment_output_binding,
      ManifestStage::alignment,
      "/alignment/step_alignment_output_binding");
  if (prefill_alignment->dtype != manifest.alignment.dtype ||
      prior_input->dtype != manifest.alignment.dtype ||
      alignment_output->dtype != manifest.alignment.dtype) {
    fail(
        ManifestStage::alignment,
        ManifestErrorCode::invariant_violation,
        "/alignment/dtype",
        "all declared alignment bindings must use alignment.dtype");
  }
  validate_canonical_engine_contracts(manifest);
  validate_connected_text_profile_axes(manifest);
  validate_canonical_profile_contracts(manifest);
}

template <typename Value>
void require_equal_fingerprint_field(
    const Value& expected,
    const Value& actual,
    const std::string_view field,
    const std::string& expected_text,
    const std::string& actual_text) {
  if (expected != actual) {
    fail(
        ManifestStage::runtime_compatibility,
        ManifestErrorCode::fingerprint_mismatch,
        child_path("/runtime", field),
        "expected '" + expected_text + "', actual '" + actual_text + "'");
  }
}

}  // namespace

std::string_view to_string(const ManifestStage stage) noexcept {
  switch (stage) {
    case ManifestStage::io:
      return "io";
    case ManifestStage::json:
      return "json";
    case ManifestStage::top_level:
      return "top_level";
    case ManifestStage::runtime_fingerprint:
      return "runtime_fingerprint";
    case ManifestStage::artifacts:
      return "artifacts";
    case ManifestStage::classifier_free_guidance:
      return "classifier_free_guidance";
    case ManifestStage::licenses:
      return "licenses";
    case ManifestStage::engine:
      return "engine";
    case ManifestStage::tensor:
      return "tensor";
    case ManifestStage::kv_cache:
      return "kv_cache";
    case ManifestStage::alignment:
      return "alignment";
    case ManifestStage::sampling:
      return "sampling";
    case ManifestStage::local_ar:
      return "local_ar";
    case ManifestStage::codec:
      return "codec";
    case ManifestStage::limits:
      return "limits";
    case ManifestStage::golden_fixture:
      return "golden_fixture";
    case ManifestStage::golden_receipt:
      return "golden_receipt";
    case ManifestStage::runtime_compatibility:
      return "runtime_compatibility";
  }
  return "unknown";
}

std::string_view to_string(const ManifestErrorCode code) noexcept {
  switch (code) {
    case ManifestErrorCode::io_error:
      return "io_error";
    case ManifestErrorCode::size_limit_exceeded:
      return "size_limit_exceeded";
    case ManifestErrorCode::json_syntax_error:
      return "json_syntax_error";
    case ManifestErrorCode::missing_field:
      return "missing_field";
    case ManifestErrorCode::unknown_field:
      return "unknown_field";
    case ManifestErrorCode::type_mismatch:
      return "type_mismatch";
    case ManifestErrorCode::invalid_value:
      return "invalid_value";
    case ManifestErrorCode::duplicate_value:
      return "duplicate_value";
    case ManifestErrorCode::invariant_violation:
      return "invariant_violation";
    case ManifestErrorCode::fingerprint_mismatch:
      return "fingerprint_mismatch";
  }
  return "unknown";
}

ManifestError::ManifestError(
    const ManifestStage stage,
    const ManifestErrorCode code,
    std::string json_pointer,
    std::string detail)
    : std::runtime_error(
          build_error_message(stage, code, json_pointer, detail)),
      stage_(stage),
      code_(code),
      json_pointer_(std::move(json_pointer)),
      detail_(std::move(detail)) {}

ManifestStage ManifestError::stage() const noexcept {
  return stage_;
}

ManifestErrorCode ManifestError::code() const noexcept {
  return code_;
}

const std::string& ManifestError::json_pointer() const noexcept {
  return json_pointer_;
}

const std::string& ManifestError::detail() const noexcept {
  return detail_;
}

std::string_view to_string(const Endianness value) noexcept {
  switch (value) {
    case Endianness::little:
      return "little";
    case Endianness::big:
      return "big";
  }
  return "unknown";
}

std::string_view to_string(const TensorDataType value) noexcept {
  switch (value) {
    case TensorDataType::fp32:
      return "fp32";
    case TensorDataType::fp16:
      return "fp16";
    case TensorDataType::bf16:
      return "bf16";
    case TensorDataType::int64:
      return "int64";
    case TensorDataType::int32:
      return "int32";
    case TensorDataType::int8:
      return "int8";
    case TensorDataType::uint8:
      return "uint8";
    case TensorDataType::boolean:
      return "bool";
  }
  return "unknown";
}

std::string_view to_string(
    const TensorMemoryLocation value) noexcept {
  switch (value) {
    case TensorMemoryLocation::device:
      return "device";
    case TensorMemoryLocation::host:
      return "host";
  }
  return "unknown";
}

std::string_view to_string(const EngineRole value) noexcept {
  switch (value) {
    case EngineRole::text_encoder:
      return "text_encoder";
    case EngineRole::main_decoder_prefill:
      return "main_decoder_prefill";
    case EngineRole::main_decoder_step:
      return "main_decoder_step";
    case EngineRole::local_ar_16:
      return "local_ar_16";
    case EngineRole::nanocodec_initial_4:
      return "nanocodec_initial_4";
    case EngineRole::nanocodec_steady_8:
      return "nanocodec_steady_8";
    case EngineRole::nanocodec_tail_1_8:
      return "nanocodec_tail_1_8";
  }
  return "unknown";
}

RuntimeBundleManifest parse_runtime_bundle_manifest(
    const std::string_view json_text) {
  Json root;
  std::unordered_map<int, std::unordered_set<std::string>> object_keys;
  const Json::parser_callback_t reject_duplicate_keys =
      [&object_keys](
          const int depth,
          const Json::parse_event_t event,
          Json& parsed) {
        if (event == Json::parse_event_t::object_start) {
          object_keys[depth].clear();
        } else if (event == Json::parse_event_t::key) {
          const int object_depth = depth - 1;
          const std::string key = parsed.get<std::string>();
          if (!object_keys[object_depth].emplace(key).second) {
            fail(
                ManifestStage::json,
                ManifestErrorCode::duplicate_value,
                child_path("", key),
                "duplicate JSON object key is not permitted");
          }
        } else if (event == Json::parse_event_t::object_end) {
          object_keys.erase(depth);
        }
        return true;
      };
  try {
    root = Json::parse(
        json_text.begin(),
        json_text.end(),
        reject_duplicate_keys,
        true,
        false);
  } catch (const Json::parse_error& error) {
    throw ManifestError(
        ManifestStage::json,
        ManifestErrorCode::json_syntax_error,
        "/",
        error.what());
  }

  constexpr ManifestStage stage = ManifestStage::top_level;
  require_exact_keys(
      root,
      stage,
      "",
      {"schema_version",
       "bundle_id",
       "created_at_utc",
       "runtime",
       "artifacts",
       "classifier_free_guidance",
       "licenses",
       "engines",
       "kv_cache",
       "alignment",
       "sampling",
       "local_ar",
       "codec",
       "limits",
       "golden_fixture",
       "golden_receipt"});

  const std::uint32_t schema_version = parse_positive_integer<std::uint32_t>(
      member(root, "schema_version"), stage, "/schema_version");
  if (schema_version != kRuntimeBundleManifestSchemaVersion) {
    fail(
        stage,
        ManifestErrorCode::invalid_value,
        "/schema_version",
        "unsupported schema version");
  }

  RuntimeBundleManifest manifest{
      .schema_version = schema_version,
      .bundle_id =
          parse_identifier(member(root, "bundle_id"), stage, "/bundle_id"),
      .created_at_utc =
          parse_rfc3339_utc(member(root, "created_at_utc"), stage, "/created_at_utc"),
      .runtime = parse_runtime_fingerprint(member(root, "runtime"), "/runtime"),
      .artifacts = parse_artifacts(member(root, "artifacts"), "/artifacts"),
      .classifier_free_guidance = parse_classifier_free_guidance(
          member(root, "classifier_free_guidance"),
          "/classifier_free_guidance"),
      .licenses = parse_licenses(member(root, "licenses"), "/licenses"),
      .engines = parse_engines(member(root, "engines"), "/engines"),
      .kv_cache = parse_kv_cache(member(root, "kv_cache"), "/kv_cache"),
      .alignment = parse_alignment(member(root, "alignment"), "/alignment"),
      .sampling = parse_sampling(member(root, "sampling"), "/sampling"),
      .local_ar = parse_local_ar(member(root, "local_ar"), "/local_ar"),
      .codec = parse_codec(member(root, "codec"), "/codec"),
      .limits = parse_limits(member(root, "limits"), "/limits"),
      .golden_fixture = parse_file_artifact(
          member(root, "golden_fixture"),
          ManifestStage::golden_fixture,
          "/golden_fixture"),
      .golden_receipt =
          parse_golden_receipt(member(root, "golden_receipt"), "/golden_receipt"),
  };
  validate_cross_field_invariants(manifest);
  return manifest;
}

RuntimeBundleManifest load_runtime_bundle_manifest(
    const std::filesystem::path& manifest_path) {
  const std::string path_bytes = manifest_path.native();
  if (path_bytes.empty() ||
      std::any_of(
          path_bytes.begin(),
          path_bytes.end(),
          [](const unsigned char byte) {
            return byte <= 0x1FU || byte == 0x7FU;
          })) {
    throw ManifestError(
        ManifestStage::io,
        ManifestErrorCode::io_error,
        "/",
        "manifest path must not be empty or contain ASCII control bytes");
  }

  const int descriptor =
      ::open(path_bytes.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW);
  if (descriptor < 0) {
    const int open_error = errno;
    throw ManifestError(
        ManifestStage::io,
        ManifestErrorCode::io_error,
        "/",
        "unable to open manifest file '" + manifest_path.string() +
            "': errno=" + std::to_string(open_error));
  }
  const ManifestFileDescriptor file(descriptor);

  struct stat file_status {};
  if (::fstat(file.get(), &file_status) != 0) {
    const int stat_error = errno;
    throw ManifestError(
        ManifestStage::io,
        ManifestErrorCode::io_error,
        "/",
        "unable to inspect opened manifest file '" + manifest_path.string() +
            "': errno=" + std::to_string(stat_error));
  }
  if (!S_ISREG(file_status.st_mode)) {
    throw ManifestError(
        ManifestStage::io,
        ManifestErrorCode::io_error,
        "/",
        "manifest path must identify a regular file");
  }
  if (file_status.st_size < 0 ||
      static_cast<std::uint64_t>(file_status.st_size) >
          kMaximumRuntimeBundleManifestBytes) {
    throw ManifestError(
        ManifestStage::io,
        ManifestErrorCode::size_limit_exceeded,
        "/",
        "manifest file exceeds the 16 MiB input limit");
  }

  std::string contents;
  contents.reserve(static_cast<std::size_t>(file_status.st_size));
  std::array<char, 64U * 1024U> buffer{};
  while (true) {
    const ssize_t bytes_read =
        ::read(file.get(), buffer.data(), buffer.size());
    if (bytes_read == 0) {
      break;
    }
    if (bytes_read < 0) {
      const int read_error = errno;
      if (read_error == EINTR) {
        continue;
      }
      throw ManifestError(
          ManifestStage::io,
          ManifestErrorCode::io_error,
          "/",
          "failed while reading manifest file '" + manifest_path.string() +
              "': errno=" + std::to_string(read_error));
    }
    const auto bytes_read_unsigned = static_cast<std::uint64_t>(bytes_read);
    if (contents.size() >
        kMaximumRuntimeBundleManifestBytes - bytes_read_unsigned) {
      throw ManifestError(
          ManifestStage::io,
          ManifestErrorCode::size_limit_exceeded,
          "/",
          "manifest file exceeded the 16 MiB input limit while being read");
    }
    contents.append(buffer.data(), static_cast<std::size_t>(bytes_read));
  }
  return parse_runtime_bundle_manifest(contents);
}

void require_exact_runtime_fingerprint(
    const RuntimeFingerprint& expected,
    const RuntimeFingerprint& actual) {
  require_equal_fingerprint_field(
      expected.os_name,
      actual.os_name,
      "os_name",
      expected.os_name,
      actual.os_name);
  require_equal_fingerprint_field(
      expected.os_version,
      actual.os_version,
      "os_version",
      expected.os_version,
      actual.os_version);
  require_equal_fingerprint_field(
      expected.architecture,
      actual.architecture,
      "architecture",
      expected.architecture,
      actual.architecture);
  require_equal_fingerprint_field(
      expected.endianness,
      actual.endianness,
      "endianness",
      std::string(to_string(expected.endianness)),
      std::string(to_string(actual.endianness)));
  require_equal_fingerprint_field(
      expected.cuda_version,
      actual.cuda_version,
      "cuda_version",
      expected.cuda_version,
      actual.cuda_version);
  require_equal_fingerprint_field(
      expected.tensorrt_version,
      actual.tensorrt_version,
      "tensorrt_version",
      expected.tensorrt_version,
      actual.tensorrt_version);
  require_equal_fingerprint_field(
      expected.driver_version,
      actual.driver_version,
      "driver_version",
      expected.driver_version,
      actual.driver_version);
  require_equal_fingerprint_field(
      expected.gpu_name,
      actual.gpu_name,
      "gpu_name",
      expected.gpu_name,
      actual.gpu_name);
  require_equal_fingerprint_field(
      expected.gpu_compute_capability,
      actual.gpu_compute_capability,
      "gpu_compute_capability",
      expected.gpu_compute_capability,
      actual.gpu_compute_capability);
  require_equal_fingerprint_field(
      expected.plugin_abi_version,
      actual.plugin_abi_version,
      "plugin_abi_version",
      std::to_string(expected.plugin_abi_version),
      std::to_string(actual.plugin_abi_version));
}

const EngineManifest& require_engine(
    const RuntimeBundleManifest& manifest,
    const EngineRole role) {
  const auto found = std::find_if(
      manifest.engines.begin(),
      manifest.engines.end(),
      [role](const EngineManifest& engine) { return engine.role == role; });
  if (found == manifest.engines.end()) {
    throw ManifestError(
        ManifestStage::engine,
        ManifestErrorCode::missing_field,
        "/engines",
        "required engine role is missing: " + std::string(to_string(role)));
  }
  return *found;
}

}  // namespace magpie_tts_rt

#include "manifest/manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <sys/stat.h>

#include "validation/character_validation.hpp"

namespace {

using magpie_tts_rt::ManifestError;
using magpie_tts_rt::ManifestErrorCode;
using magpie_tts_rt::ManifestStage;
using nlohmann::json;

[[nodiscard]] std::string read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("unable to open fixture: " + path);
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (stream.bad()) {
    throw std::runtime_error("unable to read fixture: " + path);
  }
  return contents.str();
}

[[noreturn]] void test_failure(const std::string& detail) {
  throw std::runtime_error("test assertion failed: " + detail);
}

void require(const bool condition, const std::string& detail) {
  if (!condition) {
    test_failure(detail);
  }
}

void expect_manifest_error(
    const std::string_view name,
    const ManifestStage expected_stage,
    const ManifestErrorCode expected_code,
    const std::string_view expected_path,
    const std::function<void()>& operation,
    const std::string_view expected_detail = {}) {
  try {
    operation();
  } catch (const ManifestError& error) {
    require(
        error.stage() == expected_stage,
        std::string(name) + ": unexpected stage " +
            std::string(magpie_tts_rt::to_string(error.stage())));
    require(
        error.code() == expected_code,
        std::string(name) + ": unexpected code " +
            std::string(magpie_tts_rt::to_string(error.code())));
    require(
        error.json_pointer() == expected_path,
        std::string(name) + ": unexpected path " + error.json_pointer());
    if (!expected_detail.empty()) {
      require(
          error.detail() == expected_detail,
          std::string(name) + ": unexpected detail " + error.detail());
    }
    return;
  }
  test_failure(std::string(name) + ": expected ManifestError");
}

void test_valid_manifest(const std::string& valid_text) {
  const magpie_tts_rt::RuntimeBundleManifest manifest =
      magpie_tts_rt::parse_runtime_bundle_manifest(valid_text);
  require(manifest.schema_version == 1, "schema version");
  require(manifest.licenses.size() == 8, "license artifact count");
  require(
      manifest.licenses.front().role == "project_license" &&
          manifest.licenses.back().role == "nvidia_model_notice",
      "canonical license role order");
  require(manifest.engines.size() == 7, "engine count");
  require(
      manifest.runtime.cublas.api_version_integer == 130400,
      "cuBLAS API version");
  require(
      manifest.runtime.cublas.library.soname == "libcublas.so.13" &&
          manifest.runtime.cublas.library.size_bytes == 67751616U &&
          manifest.runtime.cublas.library.sha256 ==
              "826486b8869144621e3a477cddcd28f56733c7c80c6f998b898384fc09e10f91",
      "cuBLAS loaded artifact identity");
  require(
      manifest.runtime.cublas.lt_library.soname ==
              "libcublasLt.so.13" &&
          manifest.runtime.cublas.lt_library.size_bytes == 606744240U &&
          manifest.runtime.cublas.lt_library.sha256 ==
              "b7aa42c190c2e7490abd6ea987883e05678e26222b7f9f1c9b96374fcbddbf04",
      "cuBLASLt loaded artifact identity");
  require(manifest.kv_cache.layer_bindings.size() == 12, "KV layer count");
  require(manifest.local_ar.positions.size() == 16, "Local AR positions");
  require(
      manifest.local_ar.position_embedding_kind == "learned_absolute",
      "Local AR learned absolute position embedding");
  require(
      manifest.local_ar.position_embedding_positions ==
          manifest.local_ar.positions,
      "Local AR position embedding row ordering");
  require(
      manifest.local_ar.position_embedding_source_shape ==
          std::vector<std::int64_t>({18, 768}),
      "Local AR position embedding source shape");
  require(
      manifest.local_ar.position_embedding_dtype ==
          magpie_tts_rt::TensorDataType::bf16,
      "Local AR position embedding dtype");
  require(
      manifest.local_ar.position_embedding_source_table_sha256 ==
          "1db63ebd4ceffba52e03cf67c9d186f3b7e38bb0c6eb9056a93ceeb55a4a695e",
      "accepted Sofia Local AR position embedding source table");
  require(
      manifest.local_ar.invalid_rows_encoding == "cfg_row_bitmask_lsb",
      "Local AR invalid-row encoding");
  require(
      manifest.local_ar.no_eos_frame_index == -1,
      "Local AR no-EOS sentinel");
  require(
      manifest.alignment.prior_epsilon == 0.1,
      "alignment prior epsilon");
  require(
      manifest.alignment.initial_attended == 1,
      "alignment initial attended position");
  require(
      manifest.alignment.ignored_terminal_tokens == 3,
      "alignment ignored terminal tokens");
  require(
      manifest.alignment.short_text_no_prior_max_tokens == 5,
      "alignment short-text prior boundary");
  require(manifest.alignment.lookahead == 6, "alignment lookahead");
  require(
      manifest.alignment.sink_threshold == 4,
      "alignment sink threshold");
  require(manifest.codec.initial_frames == 4, "initial codec frames");
  require(manifest.codec.steady_frames == 8, "steady codec frames");
  require(manifest.codec.tail_min_frames == 1, "minimum tail codec frames");
  require(manifest.codec.tail_max_frames == 8, "maximum tail codec frames");
  require(
      !manifest.codec.eos_frame_is_audio,
      "AUDIO_EOS is excluded from codec frames");
  require(
      manifest.codec.zero_frame_finalization ==
          "control_marker_without_codec_invocation",
      "zero-frame FINAL marker contract");
  require(
      manifest.artifacts.export_artifact.voice_id == "sofia",
      "Sofia voice-specific export");
  require(manifest.runtime.gpu_name == "NVIDIA Thor", "runtime GPU identity");
  require(
      manifest.artifacts.source_model.acceptance_receipt.size_bytes == 3,
      "authenticated artifact size");
  require(
      manifest.artifacts.source_model.source_sha256 ==
          "1010101010101010101010101010101010101010101010101010101010101010",
      "source-model identity hash");
  require(
      manifest.artifacts.source_model.version == "v2607",
      "source-model version");
  require(
      manifest.artifacts.source_model.revision ==
          "5023df68bd3f5b5ce6d666a50979bc501af145cc",
      "immutable source-model revision");
  require(
      manifest.artifacts.tokenizer.identity_sha256 ==
          "3030303030303030303030303030303030303030303030303030303030303030",
      "tokenizer identity hash");
  require(
      manifest.artifacts.tokenizer.tokenizer_vocabulary_size == 3357,
      "normal tokenizer row count");
  require(
      manifest.artifacts.tokenizer.text_embedding_rows == 3359,
      "text embedding row count");
  require(
      manifest.artifacts.tokenizer.bos_token_id == 3357 &&
          manifest.artifacts.tokenizer.eos_token_id == 3358,
      "authenticated text special-token rows");
  require(
      manifest.golden_receipt.size_bytes == 3,
      "authenticated golden receipt size");
  require(
      manifest.golden_fixture.size_bytes == 3,
      "authenticated golden fixture size");
  require(
      manifest.limits.maximum_bundle_snapshot_bytes == 66,
      "exact aggregate snapshot budget");
  require(
      manifest.limits.maximum_concurrent_requests ==
          magpie_tts_rt::kMaximumActiveRequestsPerSession,
      "one active request per session");
  require(
      manifest.artifacts.export_artifact.baked_context_length == 217,
      "baked prefill context length");
  require(
      manifest.artifacts.export_artifact.audio_bos_baked,
      "baked AUDIO_BOS contract");
  require(
      manifest.golden_receipt.baked_context_sha256 ==
          manifest.artifacts.export_artifact.baked_context_sha256,
      "golden Sofia context hash");
  require(
      manifest.golden_receipt.codec_frame_count == 12,
      "golden codec frame count");
  require(
      manifest.classifier_free_guidance.row_order ==
          "conditional_then_unconditional",
      "CFG row order");
  require(
      manifest.sampling.eos_token_id == 2017,
      "canonical Sofia AUDIO_EOS token");
  require(
      manifest.local_ar.sampling_plugin_name ==
          "MagpieLocalARSampling",
      "authenticated Local AR sampling creator");
  require(
      manifest.sampling.forbidden_token_ids ==
          std::vector<std::uint32_t>(
              {2016, 2018, 2019, 2020, 2021, 2022, 2023}),
      "canonical Sofia static forbidden-token set");
  require(
      manifest.kv_cache.first_step_position == 218,
      "first absolute step position");
  require(
      manifest.kv_cache.step_position_upper_bound_exclusive == 467,
      "exclusive step position bound");
  require(
      magpie_tts_rt::require_engine(
          manifest, magpie_tts_rt::EngineRole::main_decoder_step)
              .name == "main_decoder_step",
      "required engine lookup");
}

void test_load_manifest(const std::string& fixture_path) {
  const magpie_tts_rt::RuntimeBundleManifest manifest =
      magpie_tts_rt::load_runtime_bundle_manifest(fixture_path);
  require(
      manifest.bundle_id == "test-fixture-magpie-v1",
      "loaded bundle identifier");
}

void test_load_manifest_size_limit() {
  const auto unique_suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path oversized_path =
      std::filesystem::temp_directory_path() /
      ("mtt-oversized-manifest-" + std::to_string(unique_suffix) + ".json");

  {
    std::ofstream stream(oversized_path, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
      test_failure("unable to create oversized manifest fixture");
    }
    stream.seekp(
        static_cast<std::streamoff>(
            magpie_tts_rt::kMaximumRuntimeBundleManifestBytes));
    stream.put('x');
    if (!stream) {
      test_failure("unable to write oversized manifest fixture");
    }
  }

  try {
    expect_manifest_error(
        "manifest file size limit",
        ManifestStage::io,
        ManifestErrorCode::size_limit_exceeded,
        "/",
        [&oversized_path] {
          static_cast<void>(
              magpie_tts_rt::load_runtime_bundle_manifest(oversized_path));
        });
  } catch (...) {
    std::filesystem::remove(oversized_path);
    throw;
  }
  std::filesystem::remove(oversized_path);
}

void test_load_manifest_rejects_fifo_without_blocking() {
  const auto unique_suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path fifo_path =
      std::filesystem::temp_directory_path() /
      ("mtt-manifest-fifo-" + std::to_string(unique_suffix));
  if (::mkfifo(fifo_path.c_str(), 0600) != 0) {
    test_failure("unable to create manifest FIFO fixture");
  }
  try {
    expect_manifest_error(
        "manifest FIFO",
        ManifestStage::io,
        ManifestErrorCode::io_error,
        "/",
        [&fifo_path] {
          static_cast<void>(
              magpie_tts_rt::load_runtime_bundle_manifest(fifo_path));
        });
  } catch (...) {
    std::filesystem::remove(fifo_path);
    throw;
  }
  std::filesystem::remove(fifo_path);
}

void test_missing_field(const json& valid) {
  json candidate = valid;
  candidate.at("runtime").erase("driver_version");
  expect_manifest_error(
      "missing field",
      ManifestStage::runtime_fingerprint,
      ManifestErrorCode::missing_field,
      "/runtime/driver_version",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_unknown_field(const json& valid) {
  json candidate = valid;
  candidate.at("sampling").at("rng")["silent_fallback"] = true;
  expect_manifest_error(
      "unknown field",
      ManifestStage::sampling,
      ManifestErrorCode::unknown_field,
      "/sampling/rng/silent_fallback",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_license_inventory(const json& valid) {
  {
    json candidate = valid;
    candidate.at("licenses").erase(candidate.at("licenses").begin());
    expect_manifest_error(
        "missing license role",
        ManifestStage::licenses,
        ManifestErrorCode::invariant_violation,
        "/licenses",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("licenses").at(0).at("role") = "project_notice";
    expect_manifest_error(
        "duplicated license role",
        ManifestStage::licenses,
        ManifestErrorCode::invariant_violation,
        "/licenses/0/role",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_malformed_sha256(const json& valid) {
  {
    json candidate = valid;
    candidate.at("artifacts")
        .at("source_model")
        .at("acceptance_receipt")
        .at("sha256") =
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    expect_manifest_error(
        "malformed receipt SHA-256",
        ManifestStage::artifacts,
        ManifestErrorCode::invalid_value,
        "/artifacts/source_model/acceptance_receipt/sha256",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("artifacts").at("source_model").at("source_sha256") =
        "not-a-source-hash";
    expect_manifest_error(
        "malformed source-model SHA-256",
        ManifestStage::artifacts,
        ManifestErrorCode::invalid_value,
        "/artifacts/source_model/source_sha256",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("artifacts").at("source_model").at("revision") = "v2607";
    expect_manifest_error(
        "mutable source-model revision",
        ManifestStage::artifacts,
        ManifestErrorCode::invalid_value,
        "/artifacts/source_model/revision",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("artifacts").at("tokenizer").at("identity_sha256") =
        "not-a-tokenizer-identity";
    expect_manifest_error(
        "malformed tokenizer identity SHA-256",
        ManifestStage::artifacts,
        ManifestErrorCode::invalid_value,
        "/artifacts/tokenizer/identity_sha256",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("golden_fixture").at("sha256") =
        "not-a-golden-fixture-hash";
    expect_manifest_error(
        "malformed golden fixture SHA-256",
        ManifestStage::golden_fixture,
        ManifestErrorCode::invalid_value,
        "/golden_fixture/sha256",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_character_validation_boundaries(const json& valid) {
  using magpie_tts_rt::character_validation::is_dotted_numeric_version;
  using magpie_tts_rt::character_validation::is_identifier;
  using magpie_tts_rt::character_validation::is_lowercase_sha256;
  using magpie_tts_rt::character_validation::is_major_minor_version;
  using magpie_tts_rt::character_validation::is_rfc3339_utc_lexeme;

  require(is_identifier("A"), "single-character identifier");
  require(is_identifier("A.z_9-"), "identifier tail alphabet");
  require(
      is_identifier(std::string(128U, 'A')),
      "128-character identifier boundary");
  for (const std::string& value :
       std::vector<std::string>{
           "",
           std::string(129U, 'A'),
           ".bundle",
           "bundle/name",
           "bundlé",
       }) {
    require(!is_identifier(value), "rejected identifier " + value);
  }

  require(
      is_lowercase_sha256(std::string(64U, '0')),
      "all-digit SHA-256");
  require(
      is_lowercase_sha256(std::string(64U, 'f')),
      "lowercase hexadecimal SHA-256 boundary");
  for (const std::string& value :
       std::vector<std::string>{
           std::string(63U, 'a'),
           std::string(65U, 'a'),
           std::string(64U, 'A'),
           std::string(63U, 'a') + "g",
       }) {
    require(!is_lowercase_sha256(value), "rejected SHA-256 " + value);
  }

  for (const std::string_view value :
       {"0.0", "13.2+sbsa_1-rc.2", "10.16.2.10--"}) {
    require(
        is_dotted_numeric_version(value),
        "accepted dotted numeric version " + std::string(value));
  }
  for (const std::string_view value :
       {"13",
        ".2",
        "13.",
        "13..2",
        "13.2.",
        "13.2+",
        "13.2+cuda+13",
        "13.2/rc"}) {
    require(
        !is_dotted_numeric_version(value),
        "rejected dotted numeric version " + std::string(value));
  }

  require(
      is_major_minor_version("0011.000"),
      "accepted major.minor compute capability");
  for (const std::string_view value :
       {"11", ".0", "11.", "11.0.0", "11.a", "+11.0"}) {
    require(
        !is_major_minor_version(value),
        "rejected compute capability " + std::string(value));
  }

  for (const std::string_view value :
       {"2024-02-29T23:59:59Z",
        "2024-02-29T23:59:59.0Z",
        "9999-12-31T00:00:00.000001Z"}) {
    require(
        is_rfc3339_utc_lexeme(value),
        "accepted RFC 3339 UTC lexeme " + std::string(value));
  }
  for (const std::string_view value :
       {"2026-01-01T00:00:00",
        "2026-01-01T00:00:00z",
        "2026-01-01T00:00:00.Z",
        "2026-01-01T00:00:00+00:00",
        "2026-00-01T00:00:00Z",
        "2026-13-01T00:00:00Z",
        "2026-01-00T00:00:00Z",
        "2026-01-32T00:00:00Z",
        "2026-01-01T24:00:00Z",
        "2026-01-01T00:60:00Z",
        "2026-01-01T00:00:60Z"}) {
    require(
        !is_rfc3339_utc_lexeme(value),
        "rejected RFC 3339 UTC lexeme " + std::string(value));
  }
  require(
      is_rfc3339_utc_lexeme("0000-01-01T00:00:00Z"),
      "year zero remains a lexically valid timestamp");
  require(
      is_rfc3339_utc_lexeme("2023-02-29T00:00:00Z"),
      "calendar validation remains separate from lexical validation");

  {
    json candidate = valid;
    candidate["bundle_id"] = ".bundle";
    expect_manifest_error(
        "identifier parser preserves error contract",
        ManifestStage::top_level,
        ManifestErrorCode::invalid_value,
        "/bundle_id",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(
                  candidate.dump()));
        },
        "expected 1..128 characters matching [A-Za-z0-9][A-Za-z0-9._-]*");
  }
  {
    json candidate = valid;
    candidate["runtime"]["cuda_version"] = "13.2+cuda+13";
    expect_manifest_error(
        "numeric version parser preserves error contract",
        ManifestStage::runtime_fingerprint,
        ManifestErrorCode::invalid_value,
        "/runtime/cuda_version",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(
                  candidate.dump()));
        },
        "expected a dotted numeric version");
  }
  {
    json candidate = valid;
    candidate["runtime"]["cublas"]["lt_library"]["soname"] =
        "libcublasLt.so.12";
    expect_manifest_error(
        "cuBLASLt SONAME is fixed",
        ManifestStage::runtime_fingerprint,
        ManifestErrorCode::invalid_value,
        "/runtime/cublas/lt_library/soname",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(
                  candidate.dump()));
        },
        "loaded library SONAME must be 'libcublasLt.so.13'");
  }
  {
    json candidate = valid;
    candidate["runtime"]["gpu_compute_capability"] = "11.0.0";
    expect_manifest_error(
        "compute capability parser preserves error contract",
        ManifestStage::runtime_fingerprint,
        ManifestErrorCode::invalid_value,
        "/runtime/gpu_compute_capability",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(
                  candidate.dump()));
        },
        "GPU compute capability must use major.minor form");
  }
  {
    json candidate = valid;
    candidate["created_at_utc"] = "2026-01-01T00:00:00.Z";
    expect_manifest_error(
        "timestamp parser preserves lexical error contract",
        ManifestStage::top_level,
        ManifestErrorCode::invalid_value,
        "/created_at_utc",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(
                  candidate.dump()));
        },
        "expected an RFC 3339 UTC timestamp ending in Z");
  }
}

void test_rejected_tokenizer_row_contract(const json& valid) {
  for (const auto& [label, path, mutate] :
       std::vector<std::tuple<
           std::string,
           std::string,
           std::function<void(json&)>>>{
           {
               "embedding rows exclude special tokens",
               "/artifacts/tokenizer/text_embedding_rows",
               [](json& candidate) {
                 candidate.at("artifacts")
                     .at("tokenizer")
                     .at("text_embedding_rows") = 3357;
               },
           },
           {
               "EOS is a normal row",
               "/artifacts/tokenizer/special_tokens/eos_token_id",
               [](json& candidate) {
                 candidate.at("artifacts")
                     .at("tokenizer")
                     .at("special_tokens")
                     .at("eos_token_id") = 1;
               },
           },
           {
               "BOS and EOS alias",
               "/artifacts/tokenizer/special_tokens",
               [](json& candidate) {
                 candidate.at("artifacts")
                     .at("tokenizer")
                     .at("special_tokens")
                     .at("bos_token_id") = 3358;
               },
           },
           {
               "Japanese pad is not a normal row",
               "/artifacts/tokenizer/special_tokens/japanese_global_pad_token_id",
               [](json& candidate) {
                 candidate.at("artifacts")
                     .at("tokenizer")
                     .at("special_tokens")
                     .at("japanese_global_pad_token_id") = 3357;
               },
           },
       }) {
    json candidate = valid;
    mutate(candidate);
    expect_manifest_error(
        label,
        ManifestStage::artifacts,
        ManifestErrorCode::invariant_violation,
        path,
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(
                  candidate.dump()));
        });
  }
}

void test_rejected_plugin_library_name_as_creator(const json& valid) {
  json candidate = valid;
  candidate.at("local_ar").at("sampling_plugin_name") =
      candidate.at("artifacts").at("plugin").at("name");
  expect_manifest_error(
      "plugin library logical name is not a creator identity",
      ManifestStage::local_ar,
      ManifestErrorCode::invariant_violation,
      "/local_ar/sampling_plugin_name",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(
                candidate.dump()));
      });
}

void test_rejected_control_byte_in_artifact_path(const json& valid) {
  json candidate = valid;
  std::string path_with_nul = "model/model.nemo";
  path_with_nul.push_back('\0');
  path_with_nul += "suffix";
  candidate.at("artifacts").at("source_model").at("acceptance_receipt").at("path") =
      path_with_nul;
  expect_manifest_error(
      "NUL in artifact path",
      ManifestStage::artifacts,
      ManifestErrorCode::invalid_value,
      "/artifacts/source_model/acceptance_receipt/path",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_impossible_timestamp(const json& valid) {
  json candidate = valid;
  candidate.at("created_at_utc") = "2026-02-31T12:00:00Z";
  expect_manifest_error(
      "impossible manifest timestamp",
      ManifestStage::top_level,
      ManifestErrorCode::invalid_value,
      "/created_at_utc",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });

  candidate = valid;
  candidate.at("golden_receipt").at("created_at_utc") =
      "2025-04-31T12:00:00Z";
  expect_manifest_error(
      "impossible receipt timestamp",
      ManifestStage::golden_receipt,
      ManifestErrorCode::invalid_value,
      "/golden_receipt/created_at_utc",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_non_sofia_export(const json& valid) {
  json candidate = valid;
  candidate.at("artifacts").at("export").at("voice_id") = "aria";
  expect_manifest_error(
      "non-Sofia export",
      ManifestStage::artifacts,
      ManifestErrorCode::invalid_value,
      "/artifacts/export/voice_id",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_noncanonical_export_format(const json& valid) {
  json candidate = valid;
  candidate.at("artifacts").at("export").at("format") = "another_export";
  expect_manifest_error(
      "noncanonical export format",
      ManifestStage::artifacts,
      ManifestErrorCode::invalid_value,
      "/artifacts/export/format",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_unbaked_audio_bos(const json& valid) {
  json candidate = valid;
  candidate.at("artifacts").at("export").at("audio_bos_baked") = false;
  expect_manifest_error(
      "unbaked AUDIO_BOS",
      ManifestStage::artifacts,
      ManifestErrorCode::invariant_violation,
      "/artifacts/export/audio_bos_baked",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_golden_context_hash_mismatch(const json& valid) {
  {
    json candidate = valid;
    candidate.at("golden_receipt").at("baked_context_sha256") =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    expect_manifest_error(
        "golden baked context mismatch",
        ManifestStage::golden_receipt,
        ManifestErrorCode::invariant_violation,
        "/golden_receipt/baked_context_sha256",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& receipt = candidate.at("golden_receipt");
    receipt["speaker_embedding_sha256"] =
        receipt.at("baked_context_sha256");
    receipt.erase("baked_context_sha256");
    expect_manifest_error(
        "removed speaker hash alias",
        ManifestStage::golden_receipt,
        ManifestErrorCode::unknown_field,
        "/golden_receipt/speaker_embedding_sha256",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_golden_sample_count_mismatch(const json& valid) {
  json candidate = valid;
  candidate.at("golden_receipt").at("sample_count") = 12287;
  expect_manifest_error(
      "golden sample count mismatch",
      ManifestStage::golden_receipt,
      ManifestErrorCode::invariant_violation,
      "/golden_receipt/sample_count",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_golden_seed_and_frame_limit(const json& valid) {
  {
    json candidate = valid;
    candidate.at("golden_receipt").at("seed") = 4294967296ULL;
    expect_manifest_error(
        "golden seed exceeds public uint32 contract",
        ManifestStage::golden_receipt,
        ManifestErrorCode::invalid_value,
        "/golden_receipt/seed",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("golden_receipt").at("codec_frame_count") = 501;
    candidate.at("golden_receipt").at("sample_count") = 501 * 1024;
    expect_manifest_error(
        "golden frame count exceeds runtime limit",
        ManifestStage::golden_receipt,
        ManifestErrorCode::invariant_violation,
        "/golden_receipt/codec_frame_count",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_cfg_row_contract(const json& valid) {
  {
    json candidate = valid;
    candidate.at("classifier_free_guidance").at("row_order") =
        "unconditional_then_conditional";
    expect_manifest_error(
        "reversed CFG row order",
        ManifestStage::classifier_free_guidance,
        ManifestErrorCode::invalid_value,
        "/classifier_free_guidance/row_order",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("classifier_free_guidance").at(
        "unconditional_mask_source") = "all_false";
    expect_manifest_error(
        "all-false unconditional CFG mask",
        ManifestStage::classifier_free_guidance,
        ManifestErrorCode::invariant_violation,
        "/classifier_free_guidance",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_step_position_contract(const json& valid) {
  const auto find_step = [](json& candidate) -> json& {
    json& engines = candidate.at("engines");
    const auto step = std::find_if(
        engines.begin(), engines.end(), [](const json& engine) {
          return engine.at("role") == "main_decoder_step";
        });
    require(step != engines.end(), "step position fixture engine");
    return *step;
  };
  {
    json candidate = valid;
    candidate.at("kv_cache").at("position_semantics") = "relative";
    expect_manifest_error(
        "relative step position",
        ManifestStage::kv_cache,
        ManifestErrorCode::invalid_value,
        "/kv_cache/position_semantics",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("kv_cache").at("first_step_position") = 217;
    expect_manifest_error(
        "wrong first step position",
        ManifestStage::kv_cache,
        ManifestErrorCode::invariant_violation,
        "/kv_cache/first_step_position",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("kv_cache").at("step_position_upper_bound_exclusive") = 468;
    expect_manifest_error(
        "wrong step position bound",
        ManifestStage::kv_cache,
        ManifestErrorCode::invariant_violation,
        "/kv_cache/step_position_upper_bound_exclusive",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& step = find_step(candidate);
    step.at("inputs").at(1).at("location") = "host";
    expect_manifest_error(
        "HOST step position",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/2/inputs/1",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& step = find_step(candidate);
    step.at("inputs").at(1).at("shape_inference_io") = true;
    expect_manifest_error(
        "shape-inference step position",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/2/inputs/1",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& step = find_step(candidate);
    step.at("profiles").at(0).at("input_values") = json::array(
        {{{"tensor_name", "position"},
          {"min", json::array({218})},
          {"opt", json::array({342})},
          {"max", json::array({466})}}});
    expect_manifest_error(
        "legacy position value profile",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/2/profiles/0/input_values",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& step = find_step(candidate);
    json& inputs = step.at("inputs");
    const auto status = std::find_if(
        inputs.begin(), inputs.end(), [](const json& input) {
          return input.at("name") == "execution_status_in";
        });
    require(status != inputs.end(), "step status fixture input");
    inputs.erase(status);
    expect_manifest_error(
        "missing step execution status input",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/2",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& step = find_step(candidate);
    json& outputs = step.at("outputs");
    const auto status = std::find_if(
        outputs.begin(), outputs.end(), [](const json& output) {
          return output.at("name") == "execution_status_out";
        });
    require(status != outputs.end(), "step status fixture output");
    status->at("dtype") = "int64";
    expect_manifest_error(
        "wrong step execution status output dtype",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/2/outputs/2",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_malformed_json() {
  expect_manifest_error(
      "malformed JSON",
      ManifestStage::json,
      ManifestErrorCode::json_syntax_error,
      "/",
      [] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest("{\"schema_version\":"));
      });
}

void test_duplicate_json_key() {
  expect_manifest_error(
      "duplicate JSON key",
      ManifestStage::json,
      ManifestErrorCode::duplicate_value,
      "/schema_version",
      [] {
        static_cast<void>(magpie_tts_rt::parse_runtime_bundle_manifest(
            "{\"schema_version\":1,\"schema_version\":1}"));
      });
}

void test_rejected_four_twelve_schedule(const json& valid) {
  json candidate = valid;
  candidate.at("codec").at("steady_frames") = 12;
  expect_manifest_error(
      "4/12 codec schedule",
      ManifestStage::codec,
      ManifestErrorCode::invariant_violation,
      "/codec",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_eos_frame_contract(const json& valid) {
  {
    json candidate = valid;
    candidate.at("codec").at("eos_frame_is_audio") = true;
    expect_manifest_error(
        "AUDIO_EOS declared as audio",
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        "/codec/eos_frame_is_audio",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("codec").at("zero_frame_finalization") =
        "invoke_tail_with_zero";
    expect_manifest_error(
        "zero-frame tail invocation",
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        "/codec/zero_frame_finalization",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("golden_receipt").at("zero_frame_finalization") =
        "invoke_tail_with_zero";
    expect_manifest_error(
        "golden zero-frame contract mismatch",
        ManifestStage::golden_receipt,
        ManifestErrorCode::invariant_violation,
        "/golden_receipt",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_local_ar_position(const json& valid) {
  {
    json candidate = valid;
    candidate.at("local_ar").at("positions").at(15) = 14;
    expect_manifest_error(
        "Local AR position ordering",
        ManifestStage::local_ar,
        ManifestErrorCode::invariant_violation,
        "/local_ar/positions/15",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("local_ar").at("position_embedding").at("kind") =
        "sinusoidal";
    expect_manifest_error(
        "Local AR position embedding kind",
        ManifestStage::local_ar,
        ManifestErrorCode::invalid_value,
        "/local_ar/position_embedding/kind",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("local_ar").at("position_embedding").at("positions").at(0) =
        1;
    expect_manifest_error(
        "Local AR position embedding row ordering",
        ManifestStage::local_ar,
        ManifestErrorCode::invariant_violation,
        "/local_ar/position_embedding/positions/0",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("local_ar").at("position_embedding").at("source_shape") =
        json::array({17, 768});
    expect_manifest_error(
        "Local AR position embedding source shape",
        ManifestStage::local_ar,
        ManifestErrorCode::invalid_value,
        "/local_ar/position_embedding/source_shape",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_duplicate_kv_layer(const json& valid) {
  json candidate = valid;
  candidate.at("kv_cache").at("layer_bindings").at(11).at("layer_index") = 10;
  expect_manifest_error(
      "duplicate KV layer",
      ManifestStage::kv_cache,
      ManifestErrorCode::invariant_violation,
      "/kv_cache/layer_bindings/11/layer_index",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_missing_engine_binding(const json& valid) {
  json candidate = valid;
  json& engines = candidate.at("engines");
  auto prefill = std::find_if(
      engines.begin(), engines.end(), [](const json& engine) {
        return engine.at("role") == "main_decoder_prefill";
      });
  require(prefill != engines.end(), "prefill fixture engine");
  json& outputs = prefill->at("outputs");
  const auto tensor = std::find_if(
      outputs.begin(), outputs.end(), [](const json& output) {
        return output.at("name") == "prefill_self_key_7";
      });
  require(tensor != outputs.end(), "prefill KV fixture tensor");
  outputs.erase(tensor);

  expect_manifest_error(
      "missing explicit engine KV binding",
      ManifestStage::kv_cache,
      ManifestErrorCode::invariant_violation,
      "/kv_cache/layer_bindings/7/prefill_self_key_output",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_non_f32_emission(const json& valid) {
  json candidate = valid;
  candidate.at("codec").at("pcm_format") = "s16le";
  expect_manifest_error(
      "non-FP32 runtime emission",
      ManifestStage::codec,
      ManifestErrorCode::invalid_value,
      "/codec/pcm_format",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_missing_semantic_binding(const json& valid) {
  json candidate = valid;
  json& engines = candidate.at("engines");
  auto text_encoder = std::find_if(
      engines.begin(), engines.end(), [](const json& engine) {
        return engine.at("role") == "text_encoder";
      });
  require(text_encoder != engines.end(), "text encoder fixture engine");
  json& inputs = text_encoder->at("inputs");
  const auto text_mask = std::find_if(
      inputs.begin(), inputs.end(), [](const json& input) {
        return input.at("name") == "text_mask";
      });
  require(text_mask != inputs.end(), "text mask fixture input");
  inputs.erase(text_mask);
  json& profile_shapes =
      text_encoder->at("profiles").at(0).at("input_shapes");
  const auto text_mask_profile = std::find_if(
      profile_shapes.begin(), profile_shapes.end(), [](const json& shape) {
        return shape.at("tensor_name") == "text_mask";
      });
  require(
      text_mask_profile != profile_shapes.end(),
      "text mask fixture profile");
  profile_shapes.erase(text_mask_profile);

  expect_manifest_error(
      "missing canonical text mask",
      ManifestStage::engine,
      ManifestErrorCode::missing_field,
      "/engines/text_encoder/inputs",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_wrong_semantic_dtype(const json& valid) {
  json candidate = valid;
  json& engines = candidate.at("engines");
  const auto local_ar = std::find_if(
      engines.begin(), engines.end(), [](const json& engine) {
        return engine.at("role") == "local_ar_16";
      });
  require(local_ar != engines.end(), "Local AR fixture engine");
  json& inputs = local_ar->at("inputs");
  const auto unfinished = std::find_if(
      inputs.begin(), inputs.end(), [](const json& input) {
        return input.at("name") == "unfinished";
      });
  require(unfinished != inputs.end(), "unfinished fixture input");
  unfinished->at("dtype") = "int32";

  expect_manifest_error(
      "wrong canonical Local AR dtype",
      ManifestStage::engine,
      ManifestErrorCode::invariant_violation,
      "/engines/local_ar_16/inputs/unfinished",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_empty_codec_state_contract(const json& valid) {
  json candidate = valid;
  candidate.at("codec").at("state_bindings") = json::array();
  expect_manifest_error(
      "empty codec state contract",
      ManifestStage::codec,
      ManifestErrorCode::invariant_violation,
      "/codec/state_bindings",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_initial_codec_state_input(const json& valid) {
  json candidate = valid;
  json& engines = candidate.at("engines");
  const auto initial = std::find_if(
      engines.begin(), engines.end(), [](const json& engine) {
        return engine.at("role") == "nanocodec_initial_4";
      });
  require(initial != engines.end(), "initial codec fixture engine");
  initial->at("inputs").push_back(
      {{"name", "implicit_zero_state"},
       {"dtype", "fp32"},
       {"shape", json::array({1, 1})},
       {"location", "device"},
       {"shape_inference_io", false}});
  expect_manifest_error(
      "initial codec state input",
      ManifestStage::engine,
      ManifestErrorCode::unknown_field,
      "/engines/nanocodec_initial_4/inputs",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_missing_initial_codec_state_output(const json& valid) {
  json candidate = valid;
  json& engines = candidate.at("engines");
  const auto initial = std::find_if(
      engines.begin(), engines.end(), [](const json& engine) {
        return engine.at("role") == "nanocodec_initial_4";
      });
  require(initial != engines.end(), "initial codec fixture engine");
  json& outputs = initial->at("outputs");
  const auto state_output = std::find_if(
      outputs.begin(), outputs.end(), [](const json& output) {
        return output.at("name") ==
               "state_out.pre_conv.input_history";
      });
  require(state_output != outputs.end(), "initial codec state output fixture");
  outputs.erase(state_output);
  expect_manifest_error(
      "missing initial codec state output",
      ManifestStage::engine,
      ManifestErrorCode::missing_field,
      "/engines/nanocodec_initial_4/outputs",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_noncanonical_codec_state_registry(const json& valid) {
  {
    json candidate = valid;
    candidate.at("codec")
        .at("state_bindings")
        .at(0)
        .at("shape") = json::array({1, 32, 5});
    expect_manifest_error(
        "wrong canonical codec state shape",
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        "/codec/state_bindings/0/shape",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& states = candidate.at("codec").at("state_bindings");
    std::swap(states.at(0), states.at(1));
    expect_manifest_error(
        "reordered canonical codec states",
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        "/codec/state_bindings/0/logical_name",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("codec").at("state_bindings").erase(
        candidate.at("codec").at("state_bindings").begin());
    expect_manifest_error(
        "missing canonical codec state",
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        "/codec/state_bindings",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("codec")
        .at("state_bindings")
        .at(0)
        .at("steady_input_binding") =
        "state_in.opaque_aggregate";
    expect_manifest_error(
        "opaque codec state binding",
        ManifestStage::codec,
        ManifestErrorCode::invariant_violation,
        "/codec/state_bindings/0/steady_input_binding",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_fixed_cross_attention_shape(const json& valid) {
  json candidate = valid;
  json& engines = candidate.at("engines");
  const auto prefill = std::find_if(
      engines.begin(), engines.end(), [](const json& engine) {
        return engine.at("role") == "main_decoder_prefill";
      });
  require(prefill != engines.end(), "prefill fixture engine");
  json& outputs = prefill->at("outputs");
  const auto cross_key = std::find_if(
      outputs.begin(), outputs.end(), [](const json& output) {
        return output.at("name") == "prefill_cross_key_0";
      });
  require(cross_key != outputs.end(), "cross key fixture output");
  cross_key->at("shape") = json::array({2, 512, 1, 128});
  expect_manifest_error(
      "fixed cross attention padding",
      ManifestStage::kv_cache,
      ManifestErrorCode::invariant_violation,
      "/kv_cache/layer_bindings/0/prefill_cross_key_output",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_canonical_profile_mutations(const json& valid) {
  {
    json candidate = valid;
    json& engines = candidate.at("engines");
    const auto text_encoder = std::find_if(
        engines.begin(), engines.end(), [](const json& engine) {
          return engine.at("role") == "text_encoder";
        });
    require(text_encoder != engines.end(), "text encoder profile fixture");
    text_encoder->at("profiles").at(0).at("name") = "text_flexible";
    expect_manifest_error(
        "noncanonical profile name",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/text_encoder/profiles/0/name",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& engines = candidate.at("engines");
    const auto text_encoder = std::find_if(
        engines.begin(), engines.end(), [](const json& engine) {
          return engine.at("role") == "text_encoder";
        });
    require(text_encoder != engines.end(), "text encoder profile fixture");
    text_encoder->at("profiles")
        .at(0)
        .at("input_shapes")
        .at(1)
        .at("opt")
        .at(1) = 65;
    expect_manifest_error(
        "text input T mismatch",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/text_encoder/profiles/0/input_shapes/text_mask",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& engines = candidate.at("engines");
    const auto prefill = std::find_if(
        engines.begin(), engines.end(), [](const json& engine) {
          return engine.at("role") == "main_decoder_prefill";
        });
    require(prefill != engines.end(), "prefill profile fixture");
    prefill->at("profiles")
        .at(0)
        .at("input_shapes")
        .at(0)
        .at("max")
        .at(1) = 511;
    expect_manifest_error(
        "prefill T mismatch",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/main_decoder_prefill/profiles/0/input_shapes/condition",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& engines = candidate.at("engines");
    const auto step = std::find_if(
        engines.begin(), engines.end(), [](const json& engine) {
          return engine.at("role") == "main_decoder_step";
        });
    require(step != engines.end(), "step profile fixture");
    step->at("profiles")
        .at(0)
        .at("input_shapes")
        .at(2)
        .at("opt")
        .at(1) = 65;
    expect_manifest_error(
        "step cross-attention T mismatch",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/main_decoder_step/profiles/0/input_shapes/step_cross_key_in_0",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& engines = candidate.at("engines");
    const auto tail = std::find_if(
        engines.begin(), engines.end(), [](const json& engine) {
          return engine.at("role") == "nanocodec_tail_1_8";
        });
    require(tail != engines.end(), "tail profile fixture");
    tail->at("profiles")
        .at(0)
        .at("input_shapes")
        .at(0)
        .at("max")
        .at(2) = 7;
    expect_manifest_error(
        "tail frame range mismatch",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/nanocodec_tail_1_8/profiles/0/input_shapes/codec_tokens",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json& engines = candidate.at("engines");
    const auto local_ar = std::find_if(
        engines.begin(), engines.end(), [](const json& engine) {
          return engine.at("role") == "local_ar_16";
        });
    require(local_ar != engines.end(), "Local AR profile fixture");
    json extra = local_ar->at("profiles").at(0);
    extra.at("name") = "fixed_extra";
    local_ar->at("profiles").push_back(std::move(extra));
    expect_manifest_error(
        "extra role profile",
        ManifestStage::tensor,
        ManifestErrorCode::invariant_violation,
        "/engines/local_ar_16/profiles",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("limits").at("maximum_text_tokens") = 511;
    expect_manifest_error(
        "text profile limit mismatch",
        ManifestStage::limits,
        ManifestErrorCode::invariant_violation,
        "/limits",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_sampling_contract_mutations(const json& valid) {
  {
    json candidate = valid;
    candidate.at("sampling").at("top_k") = 79;
    expect_manifest_error(
        "noncanonical top-k",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/top_k",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("sampling").at("temperature") = 0.6000000000000001;
    expect_manifest_error(
        "noncanonical temperature",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/temperature",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("sampling").at("eos_token_id") = 2;
    expect_manifest_error(
        "noncanonical AUDIO_EOS",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/eos_token_id",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("sampling").at("forbidden_token_ids").push_back(2024);
    expect_manifest_error(
        "out-of-range sampling token",
        ManifestStage::sampling,
        ManifestErrorCode::invalid_value,
        "/sampling/forbidden_token_ids/7",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("sampling").at("forbidden_token_ids").push_back(2017);
    expect_manifest_error(
        "EOS in forbidden set",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/forbidden_token_ids/7",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("sampling").at("forbidden_token_ids").erase(0);
    expect_manifest_error(
        "special token 2016 must stay forbidden",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/forbidden_token_ids",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("sampling").at("forbidden_token_ids").push_back(0);
    expect_manifest_error(
        "codec token IDs 0 through 2015 remain eligible",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/forbidden_token_ids",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    std::swap(
        candidate.at("sampling").at("forbidden_token_ids").at(0),
        candidate.at("sampling").at("forbidden_token_ids").at(1));
    expect_manifest_error(
        "static forbidden-token order is canonical",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/forbidden_token_ids",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    json forbidden = json::array();
    for (std::uint32_t token = 0; token < 2024 && forbidden.size() < 1945;
         ++token) {
      if (token != 2017) {
        forbidden.push_back(token);
      }
    }
    candidate.at("sampling").at("forbidden_token_ids") =
        std::move(forbidden);
    expect_manifest_error(
        "insufficient sampling candidates",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/forbidden_token_ids",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_local_ar_output_encoding_mutations(const json& valid) {
  {
    json candidate = valid;
    candidate.at("local_ar")
        .at("position_embedding")
        .at("source_table_sha256") =
        "abababababababababababababababababababababababababababababababab";
    expect_manifest_error(
        "wrong Sofia Local AR position table",
        ManifestStage::local_ar,
        ManifestErrorCode::invariant_violation,
        "/local_ar/position_embedding/source_table_sha256",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("local_ar").at("invalid_rows_encoding") = "row_index";
    expect_manifest_error(
        "noncanonical invalid-row encoding",
        ManifestStage::local_ar,
        ManifestErrorCode::invalid_value,
        "/local_ar/invalid_rows_encoding",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("local_ar").at("no_eos_frame_index") = 2;
    expect_manifest_error(
        "noncanonical no-EOS sentinel",
        ManifestStage::local_ar,
        ManifestErrorCode::invalid_value,
        "/local_ar/no_eos_frame_index",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_alignment_controller_mutations(const json& valid) {
  const auto expect_noncanonical =
      [&valid](
          const std::string_view name,
          const std::string_view field,
          const json& replacement) {
        json candidate = valid;
        candidate.at("alignment").at(field) = replacement;
        const std::string path = "/alignment/" + std::string(field);
        expect_manifest_error(
            name,
            ManifestStage::alignment,
            ManifestErrorCode::invariant_violation,
            path,
            [&candidate] {
              static_cast<void>(
                  magpie_tts_rt::parse_runtime_bundle_manifest(
                      candidate.dump()));
            });
      };
  expect_noncanonical("alignment prior epsilon", "prior_epsilon", 0.2);
  expect_noncanonical("alignment initial attended", "initial_attended", 2);
  expect_noncanonical(
      "alignment ignored terminal tokens", "ignored_terminal_tokens", 4);
  expect_noncanonical(
      "alignment short-text prior boundary",
      "short_text_no_prior_max_tokens",
      6);
  expect_noncanonical("alignment lookahead", "lookahead", 7);
  expect_noncanonical("alignment sink threshold", "sink_threshold", 5);
}

void test_rejected_multiple_requests_per_session(const json& valid) {
  json candidate = valid;
  candidate.at("limits").at("maximum_sessions") = 2;
  candidate.at("limits").at("maximum_concurrent_requests") = 2;
  expect_manifest_error(
      "multiple active requests per session",
      ManifestStage::limits,
      ManifestErrorCode::invariant_violation,
      "/limits/maximum_concurrent_requests",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(
                candidate.dump()));
      });
}

void test_rejected_snapshot_size_contract_mutations(const json& valid) {
  {
    json candidate = valid;
    candidate.at("artifacts")
        .at("source_model")
        .at("acceptance_receipt")
        .at("size_bytes") = 4;
    expect_manifest_error(
        "snapshot budget does not match artifact sum",
        ManifestStage::limits,
        ManifestErrorCode::invariant_violation,
        "/limits/maximum_bundle_snapshot_bytes",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("limits").at("maximum_bundle_snapshot_bytes") =
        magpie_tts_rt::kMaximumBundleSnapshotBytes + 1;
    expect_manifest_error(
        "snapshot budget exceeds runtime hard limit",
        ManifestStage::limits,
        ManifestErrorCode::size_limit_exceeded,
        "/limits/maximum_bundle_snapshot_bytes",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("artifacts")
        .at("source_model")
        .at("acceptance_receipt")
        .at("size_bytes") =
        magpie_tts_rt::kMaximumBundleSnapshotBytes;
    candidate.at("limits").at("maximum_bundle_snapshot_bytes") =
        magpie_tts_rt::kMaximumBundleSnapshotBytes;
    expect_manifest_error(
        "artifact sum exceeds runtime hard limit",
        ManifestStage::limits,
        ManifestErrorCode::size_limit_exceeded,
        "/artifacts/export/export_receipt/size_bytes",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
}

void test_rejected_runtime_fingerprint(const std::string& valid_text) {
  const magpie_tts_rt::RuntimeBundleManifest manifest =
      magpie_tts_rt::parse_runtime_bundle_manifest(valid_text);
  magpie_tts_rt::RuntimeFingerprint actual = manifest.runtime;
  actual.driver_version = "595.79";
  expect_manifest_error(
      "runtime fingerprint mismatch",
      ManifestStage::runtime_compatibility,
      ManifestErrorCode::fingerprint_mismatch,
      "/runtime/driver_version",
      [&manifest, &actual] {
        magpie_tts_rt::require_exact_runtime_fingerprint(
            manifest.runtime, actual);
      });

  actual = manifest.runtime;
  actual.cublas.lt_library.sha256 = std::string(64U, 'f');
  expect_manifest_error(
      "cuBLASLt runtime fingerprint mismatch",
      ManifestStage::runtime_compatibility,
      ManifestErrorCode::fingerprint_mismatch,
      "/runtime/cublas/lt_library/sha256",
      [&manifest, &actual] {
        magpie_tts_rt::require_exact_runtime_fingerprint(
            manifest.runtime, actual);
      });
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: manifest_tests /path/to/minimal-valid.json\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string valid_text = read_file(argv[1]);
    const json valid = json::parse(valid_text);
    test_valid_manifest(valid_text);
    test_load_manifest(argv[1]);
    test_load_manifest_size_limit();
    test_load_manifest_rejects_fifo_without_blocking();
    test_missing_field(valid);
    test_unknown_field(valid);
    test_rejected_license_inventory(valid);
    test_malformed_sha256(valid);
    test_character_validation_boundaries(valid);
    test_rejected_tokenizer_row_contract(valid);
    test_rejected_plugin_library_name_as_creator(valid);
    test_rejected_control_byte_in_artifact_path(valid);
    test_rejected_impossible_timestamp(valid);
    test_rejected_non_sofia_export(valid);
    test_rejected_noncanonical_export_format(valid);
    test_rejected_unbaked_audio_bos(valid);
    test_rejected_golden_context_hash_mismatch(valid);
    test_rejected_golden_sample_count_mismatch(valid);
    test_rejected_golden_seed_and_frame_limit(valid);
    test_rejected_cfg_row_contract(valid);
    test_rejected_step_position_contract(valid);
    test_malformed_json();
    test_duplicate_json_key();
    test_rejected_four_twelve_schedule(valid);
    test_rejected_eos_frame_contract(valid);
    test_rejected_local_ar_position(valid);
    test_rejected_duplicate_kv_layer(valid);
    test_rejected_missing_engine_binding(valid);
    test_rejected_non_f32_emission(valid);
    test_rejected_missing_semantic_binding(valid);
    test_rejected_wrong_semantic_dtype(valid);
    test_rejected_empty_codec_state_contract(valid);
    test_rejected_initial_codec_state_input(valid);
    test_rejected_missing_initial_codec_state_output(valid);
    test_rejected_noncanonical_codec_state_registry(valid);
    test_rejected_fixed_cross_attention_shape(valid);
    test_rejected_canonical_profile_mutations(valid);
    test_rejected_sampling_contract_mutations(valid);
    test_rejected_local_ar_output_encoding_mutations(valid);
    test_rejected_alignment_controller_mutations(valid);
    test_rejected_multiple_requests_per_session(valid);
    test_rejected_snapshot_size_contract_mutations(valid);
    test_rejected_runtime_fingerprint(valid_text);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "manifest contract tests passed\n";
  return EXIT_SUCCESS;
}

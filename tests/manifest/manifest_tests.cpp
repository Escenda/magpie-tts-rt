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
#include <utility>

#include <nlohmann/json.hpp>
#include <sys/stat.h>

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
    const std::function<void()>& operation) {
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
    return;
  }
  test_failure(std::string(name) + ": expected ManifestError");
}

void test_valid_manifest(const std::string& valid_text) {
  const magpie_tts_rt::RuntimeBundleManifest manifest =
      magpie_tts_rt::parse_runtime_bundle_manifest(valid_text);
  require(manifest.schema_version == 1, "schema version");
  require(manifest.engines.size() == 7, "engine count");
  require(manifest.kv_cache.layer_bindings.size() == 12, "KV layer count");
  require(manifest.local_ar.positions.size() == 16, "Local AR positions");
  require(
      manifest.local_ar.invalid_rows_encoding == "cfg_row_bitmask_lsb",
      "Local AR invalid-row encoding");
  require(
      manifest.local_ar.no_eos_frame_index == -1,
      "Local AR no-EOS sentinel");
  require(manifest.codec.initial_frames == 4, "initial codec frames");
  require(manifest.codec.steady_frames == 8, "steady codec frames");
  require(manifest.codec.tail_min_frames == 1, "minimum tail codec frames");
  require(manifest.codec.tail_max_frames == 8, "maximum tail codec frames");
  require(
      manifest.artifacts.export_artifact.voice_id == "sofia",
      "Sofia voice-specific export");
  require(manifest.runtime.gpu_name == "NVIDIA Thor", "runtime GPU identity");
  require(
      manifest.artifacts.model.file.size_bytes == 3,
      "authenticated artifact size");
  require(
      manifest.golden_receipt.size_bytes == 3,
      "authenticated golden receipt size");
  require(
      manifest.limits.maximum_bundle_snapshot_bytes == 36,
      "exact aggregate snapshot budget");
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
      manifest.classifier_free_guidance.row_order ==
          "conditional_then_unconditional",
      "CFG row order");
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

void test_malformed_sha256(const json& valid) {
  json candidate = valid;
  candidate.at("artifacts").at("model").at("file").at("sha256") =
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
  expect_manifest_error(
      "malformed SHA-256",
      ManifestStage::artifacts,
      ManifestErrorCode::invalid_value,
      "/artifacts/model/file/sha256",
      [&candidate] {
        static_cast<void>(
            magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
      });
}

void test_rejected_control_byte_in_artifact_path(const json& valid) {
  json candidate = valid;
  std::string path_with_nul = "model/model.nemo";
  path_with_nul.push_back('\0');
  path_with_nul += "suffix";
  candidate.at("artifacts").at("model").at("file").at("path") =
      path_with_nul;
  expect_manifest_error(
      "NUL in artifact path",
      ManifestStage::artifacts,
      ManifestErrorCode::invalid_value,
      "/artifacts/model/file/path",
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
        "unconditional_mask_source") = "text_mask";
    expect_manifest_error(
        "non-zero unconditional CFG mask",
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

void test_rejected_local_ar_position(const json& valid) {
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
       {"shape", json::array({1, 1})}});
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
        return output.at("name") == "codec_state_initial_out";
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
        .at(1)
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
    candidate.at("sampling").at("forbidden_token_ids").push_back(2024);
    expect_manifest_error(
        "out-of-range sampling token",
        ManifestStage::sampling,
        ManifestErrorCode::invalid_value,
        "/sampling/forbidden_token_ids/2",
        [&candidate] {
          static_cast<void>(
              magpie_tts_rt::parse_runtime_bundle_manifest(candidate.dump()));
        });
  }
  {
    json candidate = valid;
    candidate.at("sampling").at("forbidden_token_ids").push_back(2);
    expect_manifest_error(
        "EOS in forbidden set",
        ManifestStage::sampling,
        ManifestErrorCode::invariant_violation,
        "/sampling/forbidden_token_ids/2",
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
      if (token != 2) {
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

void test_rejected_snapshot_size_contract_mutations(const json& valid) {
  {
    json candidate = valid;
    candidate.at("artifacts")
        .at("model")
        .at("file")
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
        .at("model")
        .at("file")
        .at("size_bytes") =
        magpie_tts_rt::kMaximumBundleSnapshotBytes;
    candidate.at("limits").at("maximum_bundle_snapshot_bytes") =
        magpie_tts_rt::kMaximumBundleSnapshotBytes;
    expect_manifest_error(
        "artifact sum exceeds runtime hard limit",
        ManifestStage::limits,
        ManifestErrorCode::size_limit_exceeded,
        "/artifacts/export/file/size_bytes",
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
    test_malformed_sha256(valid);
    test_rejected_control_byte_in_artifact_path(valid);
    test_rejected_impossible_timestamp(valid);
    test_rejected_non_sofia_export(valid);
    test_rejected_unbaked_audio_bos(valid);
    test_rejected_golden_context_hash_mismatch(valid);
    test_rejected_cfg_row_contract(valid);
    test_rejected_step_position_contract(valid);
    test_malformed_json();
    test_duplicate_json_key();
    test_rejected_four_twelve_schedule(valid);
    test_rejected_local_ar_position(valid);
    test_rejected_duplicate_kv_layer(valid);
    test_rejected_missing_engine_binding(valid);
    test_rejected_non_f32_emission(valid);
    test_rejected_missing_semantic_binding(valid);
    test_rejected_wrong_semantic_dtype(valid);
    test_rejected_empty_codec_state_contract(valid);
    test_rejected_initial_codec_state_input(valid);
    test_rejected_missing_initial_codec_state_output(valid);
    test_rejected_fixed_cross_attention_shape(valid);
    test_rejected_canonical_profile_mutations(valid);
    test_rejected_sampling_contract_mutations(valid);
    test_rejected_local_ar_output_encoding_mutations(valid);
    test_rejected_snapshot_size_contract_mutations(valid);
    test_rejected_runtime_fingerprint(valid_text);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "manifest contract tests passed\n";
  return EXIT_SUCCESS;
}

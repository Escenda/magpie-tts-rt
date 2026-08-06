#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "manifest/manifest.hpp"
#include "runtime/startup_golden.hpp"

namespace {

using magpie_tts_rt::RuntimeBundleManifest;
using magpie_tts_rt::StartupGoldenActual;
using magpie_tts_rt::StartupGoldenCapture;
using magpie_tts_rt::StartupGoldenError;
using magpie_tts_rt::StartupGoldenErrorCode;
using magpie_tts_rt::StartupGoldenFixture;
using Json = nlohmann::json;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] std::string repeated(
    const char value) {
  return std::string(64, value);
}

[[nodiscard]] Json valid_fixture_document(
    RuntimeBundleManifest& manifest) {
  constexpr std::string_view token_hash =
      "4bb807b0e896495b9aea56eb42e5fc192d48bb6d384bf512fa17e52a51ec9c0b";
  manifest.artifacts.tokenizer.identity_sha256 = repeated('1');
  manifest.golden_receipt.normalized_text_sha256 = repeated('2');
  manifest.golden_receipt.token_ids_sha256 = token_hash;
  manifest.golden_receipt.baked_context_sha256 = repeated('3');
  manifest.golden_receipt.seed = 7;
  manifest.golden_receipt.decoder_tokens_sha256 = repeated('4');
  manifest.golden_receipt.codec_codes_sha256 = repeated('5');
  manifest.golden_receipt.codec_frame_count = 3;
  manifest.golden_receipt.pcm_f32le_sha256 = repeated('6');
  manifest.golden_receipt.sample_count =
      3U * manifest.codec.hop_length_samples;

  return Json{
      {"schema_version", 1},
      {"fixture_id", "startup-ja-v1"},
      {"prepared_token_ids", Json::array({0, 3358})},
      {"seed", 7},
      {"tokenizer_identity_sha256", repeated('1')},
      {"oracle_lock_sha256", repeated('7')},
      {"normalized_text_sha256", repeated('2')},
      {"token_ids_sha256", token_hash},
      {"baked_context_sha256", repeated('3')},
      {"expected",
       {
           {"decoder_tokens_sha256", repeated('4')},
           {"codec_codes_sha256", repeated('5')},
           {"codec_frame_count", 3},
           {"pcm_f32le_sha256", repeated('6')},
           {"pcm_sample_count",
            3U * manifest.codec.hop_length_samples},
       }},
  };
}

void test_fixture_parser(const std::string& manifest_path) {
  RuntimeBundleManifest manifest =
      magpie_tts_rt::load_runtime_bundle_manifest(manifest_path);
  Json document = valid_fixture_document(manifest);
  const StartupGoldenFixture fixture =
      magpie_tts_rt::parse_startup_golden_fixture(
          document.dump(), manifest);
  require(fixture.fixture_id == "startup-ja-v1", "fixture id");
  require(
      fixture.prepared_token_ids ==
          std::vector<std::int32_t>({0, 3358}),
      "prepared token ids");
  require(fixture.seed == 7, "seed");

  document = valid_fixture_document(manifest);
  document["fixture_id"] = std::string(128U, 'A');
  require(
      magpie_tts_rt::parse_startup_golden_fixture(
          document.dump(), manifest)
              .fixture_id == std::string(128U, 'A'),
      "128-character fixture identifier boundary");

  document = valid_fixture_document(manifest);
  document["fixture_id"] = ".fixture";
  try {
    static_cast<void>(
        magpie_tts_rt::parse_startup_golden_fixture(
            document.dump(), manifest));
    throw std::runtime_error("invalid fixture identifier was accepted");
  } catch (const StartupGoldenError& caught) {
    require(
        caught.code() == StartupGoldenErrorCode::invalid_fixture,
        "invalid fixture identifier error code");
    require(
        caught.detail() ==
            "/fixture_id: identifier does not match schema version 1",
        "invalid fixture identifier error detail");
  }

  document = valid_fixture_document(manifest);
  document["oracle_lock_sha256"] = std::string(64U, 'A');
  try {
    static_cast<void>(
        magpie_tts_rt::parse_startup_golden_fixture(
            document.dump(), manifest));
    throw std::runtime_error("invalid fixture SHA-256 was accepted");
  } catch (const StartupGoldenError& caught) {
    require(
        caught.code() == StartupGoldenErrorCode::invalid_fixture,
        "invalid fixture SHA-256 error code");
    require(
        caught.detail() ==
            "/oracle_lock_sha256: expected 64 lowercase hexadecimal characters",
        "invalid fixture SHA-256 error detail");
  }

  document = valid_fixture_document(manifest);
  document["token_ids_sha256"] = repeated('8');
  try {
    static_cast<void>(
        magpie_tts_rt::parse_startup_golden_fixture(
            document.dump(), manifest));
    throw std::runtime_error("corrupt token hash was accepted");
  } catch (const StartupGoldenError& caught) {
    require(
        caught.code() == StartupGoldenErrorCode::invalid_fixture,
        "corrupt token hash error code");
  }

  for (const auto& invalid_tokens :
       std::vector<Json>{
           Json::array({0}),
           Json::array({0, 3358, 3358}),
           Json::array({0, 3357, 3358}),
           Json::array({0, 3359, 3358}),
       }) {
    document = valid_fixture_document(manifest);
    document["prepared_token_ids"] = invalid_tokens;
    try {
      static_cast<void>(
          magpie_tts_rt::parse_startup_golden_fixture(
              document.dump(), manifest));
      throw std::runtime_error(
          "invalid prepared frontend sequence was accepted");
    } catch (const StartupGoldenError& caught) {
      require(
          caught.code() == StartupGoldenErrorCode::invalid_fixture,
          "prepared sequence error code");
    }
  }
}

void test_capture_layout_and_match() {
  StartupGoldenCapture capture(2, 2, 8);
  const std::vector<std::int64_t> first_codes{
      1, 2, 10, 11};
  const std::vector<float> first_pcm{
      0.5F, -0.5F, 1.0F, -1.0F};
  capture.record_chunk(first_codes, 2, first_pcm);
  const std::vector<std::int64_t> tail_codes{3, 12};
  const std::vector<float> tail_pcm{0.25F, -0.25F};
  capture.record_chunk(tail_codes, 1, tail_pcm);

  const StartupGoldenActual actual = capture.finish();
  require(
      actual.decoder_tokens_sha256 ==
          "c86f3d68305efca2ed3faab4bdd36aa7509aeffae7c28e13d1ac7f536f425fa1",
      "decoder tensor byte order");
  require(
      actual.codec_codes_sha256 ==
          "53e404501b383401258ba2a495bb10bf4554f1a27e6869e190a8195887bfd1ae",
      "scheduled codec byte order");
  require(
      actual.pcm_f32le_sha256 ==
          "a95b55feecba5ac365de6f8370f60483b1754292622cb814a55f36fe38825ef0",
      "PCM byte order");
  require(actual.codec_frame_count == 3, "codec frame count");
  require(actual.pcm_sample_count == 6, "PCM sample count");

  StartupGoldenFixture fixture{
      .schema_version = 1,
      .fixture_id = "capture",
      .prepared_token_ids = {0},
      .seed = 0,
      .tokenizer_identity_sha256 = repeated('1'),
      .oracle_lock_sha256 = repeated('2'),
      .normalized_text_sha256 = repeated('3'),
      .token_ids_sha256 = repeated('4'),
      .baked_context_sha256 = repeated('5'),
      .expected =
          {
              .decoder_tokens_sha256 =
                  actual.decoder_tokens_sha256,
              .codec_codes_sha256 = actual.codec_codes_sha256,
              .codec_frame_count = actual.codec_frame_count,
              .pcm_f32le_sha256 = actual.pcm_f32le_sha256,
              .pcm_sample_count = actual.pcm_sample_count,
          },
  };
  magpie_tts_rt::require_startup_golden_match(fixture, actual);

  fixture.expected.pcm_f32le_sha256 = repeated('0');
  try {
    magpie_tts_rt::require_startup_golden_match(
        fixture, actual);
    throw std::runtime_error("corrupt PCM hash was accepted");
  } catch (const StartupGoldenError& caught) {
    require(
        caught.code() == StartupGoldenErrorCode::hash_mismatch,
        "PCM mismatch error code");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "expected one runtime manifest fixture path");
    }
    test_fixture_parser(argv[1]);
    test_capture_layout_and_match();
    std::cout << "startup golden tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& caught) {
    std::cerr << caught.what() << '\n';
    return EXIT_FAILURE;
  }
}

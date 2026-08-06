#define MTT_RUNTIME_SMOKE_TESTING
#include "../../tools/runtime_smoke.cpp"

#include <cstdlib>
#include <iostream>

namespace {

void require(const bool condition, const std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Function>
void require_failure(Function&& function, const std::string_view message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

[[nodiscard]] mtt_model_info_v1_t valid_model_info() {
  mtt_model_info_v1_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = MTT_ABI_VERSION_1;
  info.tokenizer_vocabulary_size = 4'096U;
  info.text_embedding_rows = 4'098U;
  info.bos_token_id = 4'096U;
  info.eos_token_id = 4'097U;
  info.japanese_global_pad_token_id = 1U;
  info.maximum_text_tokens = 512U;
  info.maximum_audio_frames = 1'024U;
  info.sample_rate_hz = 22'050U;
  info.channels = 1U;
  info.pcm_format = MTT_PCM_FORMAT_F32_MONO;
  info.codec_frame_samples = 1'024U;
  info.initial_frames = 4U;
  info.steady_frames = 8U;
  info.tail_min_frames = 1U;
  info.tail_max_frames = 8U;
  info.tokenizer_identity_sha256[0] = 1U;
  return info;
}

std::uint64_t release_count = 0U;

mtt_status_t release_audio(
    mtt_request_t*, const std::uint64_t, mtt_error_v1_t* error) {
  ++release_count;
  error->code = MTT_STATUS_OK;
  error->stage = MTT_ERROR_STAGE_NONE;
  error->message[0] = '\0';
  return MTT_STATUS_OK;
}

[[nodiscard]] NativeObjects fake_native() {
  static std::uint8_t request_storage = 0U;
  NativeObjects native;
  native.request =
      reinterpret_cast<mtt_request_t*>(&request_storage);
  native.api.audio_release = release_audio;
  return native;
}

[[nodiscard]] mtt_alignment_event_v1_t alignment_event(
    const std::uint64_t sample_index,
    const std::uint64_t committed_tokens) {
  mtt_alignment_event_v1_t event{};
  event.struct_size = sizeof(event);
  event.abi_version = MTT_ABI_VERSION_1;
  event.sample_index = sample_index;
  event.committed_text_tokens = committed_tokens;
  return event;
}

[[nodiscard]] mtt_audio_lease_v1_t audio_lease(
    const std::uint64_t lease_id, const std::uint64_t sequence,
    const std::uint64_t first_sample_index, float* samples,
    const std::uint64_t sample_count, const std::uint32_t flags,
    const std::uint64_t committed_tokens,
    mtt_alignment_event_v1_t* events,
    const std::uint64_t event_count) {
  mtt_audio_lease_v1_t lease{};
  lease.struct_size = sizeof(lease);
  lease.abi_version = MTT_ABI_VERSION_1;
  lease.lease_id = lease_id;
  lease.samples = samples;
  lease.sample_count = sample_count;
  lease.first_sample_index = first_sample_index;
  lease.sequence = sequence;
  lease.sample_rate_hz = 22'050U;
  lease.channels = 1U;
  lease.format = MTT_PCM_FORMAT_F32_MONO;
  lease.flags = flags;
  lease.committed_text_tokens = committed_tokens;
  lease.alignment_events = events;
  lease.alignment_event_count = event_count;
  return lease;
}

void test_help_is_successful() {
  char program[] = "mtt-runtime-smoke";
  char help[] = "--help";
  char* arguments[] = {program, help};
  require(run(2, arguments) == EXIT_SUCCESS, "--help must exit zero");
}

void test_error_and_model_headers_fail_closed() {
  auto error = fresh_error();
  require(
      validate_call_status("test", MTT_STATUS_OK, error) ==
          MTT_STATUS_OK,
      "canonical successful error buffer");
  error.struct_size = 0U;
  require_failure(
      [&]() {
        static_cast<void>(
            validate_call_status("test", MTT_STATUS_OK, error));
      },
      "bad error struct_size must fail");

  auto info = valid_model_info();
  validate_model_info(info);
  info.sample_rate_hz = 24'000U;
  require_failure(
      [&]() { validate_model_info(info); },
      "noncanonical sample rate must fail");
  info = valid_model_info();
  info.reserved[2] = 1U;
  require_failure(
      [&]() { validate_model_info(info); },
      "nonzero model reserved field must fail");
}

void test_chunk_schedule_and_alignment_contract() {
  auto native = fake_native();
  const auto info = valid_model_info();
  StreamMetrics metrics;
  std::vector<float> first_samples(4U * 1'024U, 0.25F);
  std::array<mtt_alignment_event_v1_t, 2> first_events{
      alignment_event(2'048U, 1U),
      alignment_event(4'096U, 2U),
  };
  const auto first = audio_lease(
      1U, 0U, 0U, first_samples.data(), first_samples.size(),
      MTT_AUDIO_FLAG_FIRST | MTT_AUDIO_FLAG_ALIGNMENT_VALID, 2U,
      first_events.data(), first_events.size());
  validate_and_release_lease(native, first, info, 4U, metrics);
  require(metrics.sample_count == 4'096U, "four-frame first chunk");

  std::vector<float> steady_samples(8U * 1'024U, 0.125F);
  auto steady_event = alignment_event(6'144U, 3U);
  const auto steady = audio_lease(
      2U, 1U, 4'096U, steady_samples.data(), steady_samples.size(),
      MTT_AUDIO_FLAG_ALIGNMENT_VALID, 3U, &steady_event, 1U);
  validate_and_release_lease(native, steady, info, 4U, metrics);
  require(metrics.sample_count == 12'288U, "eight-frame steady chunk");

  std::vector<float> tail_samples(1'024U, -0.25F);
  auto tail_event = alignment_event(13'312U, 4U);
  const auto tail = audio_lease(
      3U, 2U, 12'288U, tail_samples.data(), tail_samples.size(),
      MTT_AUDIO_FLAG_FINAL | MTT_AUDIO_FLAG_ALIGNMENT_VALID, 4U,
      &tail_event, 1U);
  validate_and_release_lease(native, tail, info, 4U, metrics);
  require(metrics.final_seen, "one-frame terminal tail");
  require(metrics.sample_count == 13'312U, "complete stream samples");

  StreamMetrics marker_metrics;
  const auto marker_first = audio_lease(
      4U, 0U, 0U, first_samples.data(), first_samples.size(),
      MTT_AUDIO_FLAG_FIRST | MTT_AUDIO_FLAG_ALIGNMENT_VALID, 2U,
      first_events.data(), first_events.size());
  validate_and_release_lease(
      native, marker_first, info, 4U, marker_metrics);
  const auto marker = audio_lease(
      5U, 1U, first_samples.size(), nullptr, 0U,
      MTT_AUDIO_FLAG_FINAL | MTT_AUDIO_FLAG_ALIGNMENT_VALID, 2U,
      nullptr, 0U);
  validate_and_release_lease(
      native, marker, info, 4U, marker_metrics);
  require(marker_metrics.final_seen, "zero-frame FINAL marker");
  require(
      marker_metrics.terminal_control_marker_seen,
      "zero-frame FINAL marker was not recorded");
  require(
      marker_metrics.sample_count == first_samples.size(),
      "zero-frame FINAL marker changed PCM accounting");

  StreamMetrics invalid_metrics;
  std::vector<float> one_frame(1'024U, 0.0F);
  const auto invalid_first = audio_lease(
      6U, 0U, 0U, one_frame.data(), one_frame.size(),
      MTT_AUDIO_FLAG_FIRST | MTT_AUDIO_FLAG_FINAL, 0U, nullptr, 0U);
  require_failure(
      [&]() {
        validate_and_release_lease(
            native, invalid_first, info, 3U, invalid_metrics);
      },
      "a one-frame first chunk must fail");
  const auto invalid_zero_first = audio_lease(
      7U, 0U, 0U, nullptr, 0U,
      MTT_AUDIO_FLAG_FIRST | MTT_AUDIO_FLAG_FINAL, 0U, nullptr, 0U);
  require_failure(
      [&]() {
        validate_and_release_lease(
            native, invalid_zero_first, info, 3U,
            invalid_metrics);
      },
      "a zero-frame FIRST marker must fail");

  StreamMetrics invalid_alignment_metrics;
  std::vector<float> valid_first_samples(4U * 1'024U, 0.0F);
  auto invalid_event = alignment_event(2'048U, 4U);
  const auto invalid_alignment = audio_lease(
      8U, 0U, 0U, valid_first_samples.data(),
      valid_first_samples.size(),
      MTT_AUDIO_FLAG_FIRST | MTT_AUDIO_FLAG_ALIGNMENT_VALID, 4U,
      &invalid_event, 1U);
  require_failure(
      [&]() {
        validate_and_release_lease(
            native, invalid_alignment, info, 3U,
            invalid_alignment_metrics);
      },
      "alignment beyond request token_count must fail");
  require(release_count == 8U, "every acquired lease must be released");
}

void test_snapshot_contract_and_receipt() {
  const auto info = valid_model_info();
  StreamMetrics metrics;
  mtt_request_snapshot_v1_t running{};
  running.struct_size = sizeof(running);
  running.abi_version = MTT_ABI_VERSION_1;
  running.revision = 1U;
  running.state = MTT_REQUEST_STATE_RUNNING;
  running.generated_codec_frames = 4U;
  running.published_samples = 4'096U;
  running.committed_text_tokens = 2U;
  validate_snapshot(
      "test", running, 3U, info, std::uint64_t{0U}, metrics);

  auto changed_without_revision = running;
  changed_without_revision.available_audio_leases = 1U;
  require_failure(
      [&]() {
        validate_snapshot(
            "test", changed_without_revision, 3U, info,
            std::nullopt, metrics);
      },
      "same revision with changed fields must fail");

  auto terminal = running;
  terminal.revision = 2U;
  terminal.state = MTT_REQUEST_STATE_COMPLETED;
  terminal.generated_codec_frames = 5U;
  terminal.published_samples = 5'120U;
  terminal.committed_text_tokens = 3U;
  validate_snapshot(
      "test", terminal, 3U, info, std::uint64_t{1U}, metrics);
  metrics.final_snapshot = terminal;
  metrics.chunk_count = 2U;
  metrics.sample_count = 5'120U;
  metrics.last_alignment_sample = 5'120U;
  metrics.terminal_control_marker_seen = true;

  std::filesystem::path directory;
  for (std::uint32_t attempt = 0U; attempt < 32U; ++attempt) {
    directory =
        std::filesystem::temp_directory_path() /
        ("mtt-runtime-smoke-tests-" +
         std::to_string(static_cast<std::uint64_t>(::getpid())) + "-" +
         std::to_string(attempt));
    if (std::filesystem::create_directory(directory)) {
      break;
    }
    directory.clear();
  }
  require(!directory.empty(), "create receipt test directory");
  const auto library = directory / "libmagpie_tts_rt.so.0.1.0";
  {
    std::ofstream output(library, std::ios::binary);
    output << "runtime-library";
  }
  const auto bundle = directory / "sofia-runtime-bundle";
  std::filesystem::create_directory(bundle);
  {
    std::ofstream output(
        bundle / "runtime-bundle-manifest.json",
        std::ios::binary);
    output << "{}";
  }

  Arguments arguments;
  arguments.library = library;
  arguments.bundle = bundle;
  arguments.manifest_sha256_hex = std::string(64U, '1');
  GoldenInput input{{1, 2, 3}, 7U};
  const auto receipt = build_receipt(
      arguments, input, info, metrics, std::chrono::milliseconds(10),
      std::chrono::milliseconds(20), std::chrono::milliseconds(30));
  require(
      receipt.at("schema_version") ==
          "magpie-tts-rt.runtime-smoke.v1",
      "receipt schema version");
  require(
      receipt.at("verification").at("native_cleanup") == "passed",
      "receipt cleanup gate");
  require(
      receipt.at("result").at("terminal_control_marker_seen") == true,
      "receipt terminal control marker evidence");
  require(
      receipt.at("inputs").at("library_name") ==
          "libmagpie_tts_rt.so.0.1.0" &&
          receipt.at("inputs").at("bundle_name") ==
              "sofia-runtime-bundle",
      "public receipt inputs contain basenames");
  require(
      receipt.at("inputs").at("library_size_bytes") ==
              std::filesystem::file_size(library) &&
          receipt.at("inputs").at("bundle_manifest_size_bytes") ==
              std::filesystem::file_size(
                  bundle / "runtime-bundle-manifest.json"),
      "public receipt inputs contain artifact sizes");
  require(
      receipt.dump().find(directory.string()) == std::string::npos,
      "public receipt must not contain the runner-local directory");
  require(
      receipt.dump().find("/home/") == std::string::npos &&
          receipt.dump().find("file://") == std::string::npos &&
          receipt.dump().find(R"(C:\\)") == std::string::npos,
      "public receipt must not contain path-shaped values");
  require(
      public_artifact_name("/home/x", "test") == "x",
      "Unix absolute paths are reduced to a basename");
  require(
      public_artifact_name("/home/x/", "test") == "x",
      "trailing directory separators do not change the public name");
  require_failure(
      [&]() {
        static_cast<void>(public_artifact_name(
            R"(C:\runner\file)", "test"));
      },
      "a Windows path-shaped POSIX basename must fail closed");
  const auto receipt_path = directory / "receipt.json";
  write_new_receipt(receipt_path, receipt);
  require(
      read_json_file(receipt_path, "test receipt") == receipt,
      "persisted receipt must round-trip");
  require_failure(
      [&]() { write_new_receipt(receipt_path, receipt); },
      "immutable receipt must not be overwritten");
  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  try {
    test_help_is_successful();
    test_error_and_model_headers_fail_closed();
    test_chunk_schedule_and_alignment_contract();
    test_snapshot_contract_and_receipt();
    std::cout << "runtime smoke tool tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << "runtime smoke tool test failed: "
              << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}

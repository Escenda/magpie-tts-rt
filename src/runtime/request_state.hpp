#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace magpie_tts_rt {

inline constexpr std::uint32_t kCodecSamplesPerFrame = 1024;
inline constexpr std::uint32_t kStreamingSampleRateHz = 22050;

enum class PreparedTokenErrorCode {
  none,
  empty,
  final_token_is_not_eos,
  non_final_token_is_not_a_normal_row,
};

struct PreparedTokenValidation {
  PreparedTokenErrorCode code;
  std::uint64_t index;
  std::int64_t token_id;

  [[nodiscard]] bool valid() const noexcept {
    return code == PreparedTokenErrorCode::none;
  }
};

// A prepared frontend sequence is not an arbitrary embedding-row sequence.
// Exactly one authenticated EOS terminates it; every preceding identifier is
// a normal tokenizer row. This rejects BOS/EOS in the body and every
// unauthenticated embedding row.
[[nodiscard]] PreparedTokenValidation validate_prepared_token_ids(
    std::span<const std::int32_t> token_ids,
    std::uint32_t tokenizer_vocabulary_size,
    std::uint32_t eos_token_id) noexcept;

[[nodiscard]] PreparedTokenValidation validate_prepared_token_ids(
    std::span<const std::int64_t> token_ids,
    std::uint32_t tokenizer_vocabulary_size,
    std::uint32_t eos_token_id) noexcept;

enum class RequestLifecycleState {
  running,
  completed,
  cancelled,
  failed,
};

enum class RequestStateErrorCode {
  invalid_transition,
  invalid_audio_chunk,
  backpressure,
  no_audio,
  unknown_lease,
  lease_sequence_exhausted,
};

[[nodiscard]] std::string_view to_string(
    RequestStateErrorCode code) noexcept;

class RequestStateError final : public std::runtime_error {
 public:
  RequestStateError(RequestStateErrorCode code, std::string detail);

  [[nodiscard]] RequestStateErrorCode code() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  RequestStateErrorCode code_;
  std::string detail_;
};

struct AlignmentProgress {
  std::uint64_t sample_index;
  std::uint64_t committed_text_tokens;
};

// Immutable PCM storage shared by the request queue and a live lease. The
// runtime constructs this with a cudaHostAlloc-backed shared array, while the
// contract tests use the copying vector constructor. Keeping the owner here
// makes the C ABI sample pointer stable without copying when a lease moves
// from the available queue to the live-lease table.
class AudioBuffer final {
 public:
  AudioBuffer() = default;
  AudioBuffer(std::vector<float> samples);
  AudioBuffer(
      std::shared_ptr<const float[]> samples,
      std::uint64_t sample_count);

  [[nodiscard]] const float* data() const noexcept;
  [[nodiscard]] std::uint64_t size() const noexcept;

 private:
  std::shared_ptr<const float[]> samples_;
  std::uint64_t sample_count_{0};
};

struct AudioChunk {
  AudioBuffer samples;
  std::uint64_t first_sample_index;
  std::uint64_t sequence;
  std::uint32_t codec_frame_count;
  bool first;
  bool final;
  bool alignment_valid;
  std::uint64_t committed_text_tokens;
  std::vector<AlignmentProgress> alignment_events;
};

struct RequestStateSnapshot {
  std::uint64_t revision;
  RequestLifecycleState state;
  std::uint32_t available_audio_leases;
  std::uint64_t generated_codec_frames;
  std::uint64_t published_samples;
  std::uint64_t committed_text_tokens;
  std::int32_t terminal_status;
  std::int32_t terminal_error_stage;
  std::string terminal_error_message;
};

struct AudioLeaseView {
  std::uint64_t lease_id;
  const float* samples;
  std::uint64_t sample_count;
  std::uint64_t first_sample_index;
  std::uint64_t sequence;
  bool first;
  bool final;
  bool alignment_valid;
  std::uint64_t committed_text_tokens;
  const AlignmentProgress* alignment_events;
  std::uint64_t alignment_event_count;
};

class LeaseIdSequence final {
 public:
  explicit LeaseIdSequence(std::uint64_t initial_value = 1);

  LeaseIdSequence(const LeaseIdSequence&) = delete;
  LeaseIdSequence& operator=(const LeaseIdSequence&) = delete;

  [[nodiscard]] std::uint64_t acquire();

 private:
  std::atomic<std::uint64_t> next_;
};

class StreamingRequestState final {
 public:
  StreamingRequestState(
      std::uint64_t text_token_count,
      std::uint32_t pcm_ring_capacity_frames,
      LeaseIdSequence& lease_ids);

  StreamingRequestState(const StreamingRequestState&) = delete;
  StreamingRequestState& operator=(const StreamingRequestState&) = delete;

  [[nodiscard]] bool can_publish(std::uint32_t codec_frames) const;
  // Returns false only when an accepted cancellation closed the publication
  // gate. The cancellation and publication decisions share mutex_, so a
  // successful request_cancellation() cannot be followed by a new lease.
  [[nodiscard]] bool publish(AudioChunk chunk);
  // Returns true when cancellation was accepted (including an idempotent
  // repeat), and false when completion/failure won the terminal race.
  [[nodiscard]] bool request_cancellation();
  // Reports whether cancellation has been accepted without making the request
  // terminal. The synthesis worker uses this to close publication immediately
  // while retaining RUNNING until every armed CUDA boundary has drained.
  [[nodiscard]] bool cancellation_requested() const;
  // Publishes an accepted cancellation only after the synthesis pipeline has
  // drained every armed generation/codec event. This is not an ordinary engine
  // boundary hook. Returns false unless the request is still RUNNING with an
  // accepted cancellation.
  [[nodiscard]] bool complete_cancellation_after_drain();
  void fail(
      std::int32_t terminal_status,
      std::int32_t terminal_error_stage,
      std::string terminal_error_message);

  [[nodiscard]] RequestStateSnapshot snapshot() const;
  [[nodiscard]] bool wait_for_revision(
      std::uint64_t after_revision,
      std::chrono::nanoseconds timeout,
      RequestStateSnapshot& snapshot) const;

  [[nodiscard]] AudioLeaseView acquire_audio();
  void release_audio(std::uint64_t lease_id);

  [[nodiscard]] bool is_terminal() const;
  [[nodiscard]] bool has_live_leases() const;
  [[nodiscard]] std::uint32_t occupied_codec_frames() const;

 private:
  struct StoredLease {
    std::uint64_t lease_id;
    AudioChunk chunk;
  };

  [[nodiscard]] RequestStateSnapshot snapshot_locked() const;
  void validate_chunk_locked(const AudioChunk& chunk) const;
  void increment_revision_locked();
  void discard_unleased_audio_locked();
  void complete_cancellation_locked();

  std::uint64_t text_token_count_;
  std::uint32_t pcm_ring_capacity_frames_;
  LeaseIdSequence& lease_ids_;

  mutable std::mutex mutex_;
  mutable std::condition_variable revision_changed_;
  std::uint64_t revision_{1};
  RequestLifecycleState state_{RequestLifecycleState::running};
  bool cancellation_requested_{false};
  std::uint64_t generated_codec_frames_{0};
  std::uint64_t published_samples_{0};
  std::uint64_t committed_text_tokens_{0};
  std::uint64_t next_sample_index_{0};
  std::uint64_t next_sequence_{0};
  std::uint32_t occupied_codec_frames_{0};
  std::int32_t terminal_status_{0};
  std::int32_t terminal_error_stage_{0};
  std::string terminal_error_message_;
  std::deque<AudioChunk> available_audio_;
  std::map<std::uint64_t, StoredLease> live_leases_;
};

}  // namespace magpie_tts_rt

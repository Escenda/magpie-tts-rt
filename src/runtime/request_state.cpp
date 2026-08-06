#include "runtime/request_state.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string build_error_message(
    const RequestStateErrorCode code,
    const std::string_view detail) {
  return "streaming request state error [code=" +
         std::string(to_string(code)) + "]: " + std::string(detail);
}

[[noreturn]] void fail_state(
    const RequestStateErrorCode code,
    const std::string& detail) {
  throw RequestStateError(code, detail);
}

[[nodiscard]] bool is_terminal_state(
    const RequestLifecycleState state) noexcept {
  return state != RequestLifecycleState::running;
}

template <typename Token>
[[nodiscard]] PreparedTokenValidation validate_prepared_token_ids_impl(
    const std::span<const Token> token_ids,
    const std::uint32_t tokenizer_vocabulary_size,
    const std::uint32_t eos_token_id) noexcept {
  if (token_ids.empty()) {
    return PreparedTokenValidation{
        .code = PreparedTokenErrorCode::empty,
        .index = 0,
        .token_id = -1,
    };
  }
  const std::size_t final_index = token_ids.size() - 1;
  const std::int64_t final_token =
      static_cast<std::int64_t>(token_ids[final_index]);
  if (final_token != static_cast<std::int64_t>(eos_token_id)) {
    return PreparedTokenValidation{
        .code = PreparedTokenErrorCode::final_token_is_not_eos,
        .index = static_cast<std::uint64_t>(final_index),
        .token_id = final_token,
    };
  }
  for (std::size_t index = 0; index < final_index; ++index) {
    const std::int64_t token =
        static_cast<std::int64_t>(token_ids[index]);
    if (token < 0 ||
        token >=
            static_cast<std::int64_t>(tokenizer_vocabulary_size)) {
      return PreparedTokenValidation{
          .code =
              PreparedTokenErrorCode::
                  non_final_token_is_not_a_normal_row,
          .index = static_cast<std::uint64_t>(index),
          .token_id = token,
      };
    }
  }
  return PreparedTokenValidation{
      .code = PreparedTokenErrorCode::none,
      .index = 0,
      .token_id = 0,
  };
}

}  // namespace

PreparedTokenValidation validate_prepared_token_ids(
    const std::span<const std::int32_t> token_ids,
    const std::uint32_t tokenizer_vocabulary_size,
    const std::uint32_t eos_token_id) noexcept {
  return validate_prepared_token_ids_impl(
      token_ids, tokenizer_vocabulary_size, eos_token_id);
}

PreparedTokenValidation validate_prepared_token_ids(
    const std::span<const std::int64_t> token_ids,
    const std::uint32_t tokenizer_vocabulary_size,
    const std::uint32_t eos_token_id) noexcept {
  return validate_prepared_token_ids_impl(
      token_ids, tokenizer_vocabulary_size, eos_token_id);
}

std::string_view to_string(const RequestStateErrorCode code) noexcept {
  switch (code) {
    case RequestStateErrorCode::invalid_transition:
      return "invalid_transition";
    case RequestStateErrorCode::invalid_audio_chunk:
      return "invalid_audio_chunk";
    case RequestStateErrorCode::backpressure:
      return "backpressure";
    case RequestStateErrorCode::no_audio:
      return "no_audio";
    case RequestStateErrorCode::unknown_lease:
      return "unknown_lease";
    case RequestStateErrorCode::lease_sequence_exhausted:
      return "lease_sequence_exhausted";
  }
  return "unknown";
}

RequestStateError::RequestStateError(
    const RequestStateErrorCode code,
    std::string detail)
    : std::runtime_error(build_error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

RequestStateErrorCode RequestStateError::code() const noexcept {
  return code_;
}

const std::string& RequestStateError::detail() const noexcept {
  return detail_;
}

AudioBuffer::AudioBuffer(std::vector<float> samples)
    : sample_count_(samples.size()) {
  if (samples.empty()) {
    return;
  }
  std::shared_ptr<float[]> owned(
      new float[samples.size()],
      std::default_delete<float[]>());
  std::copy(samples.begin(), samples.end(), owned.get());
  samples_ = std::move(owned);
}

AudioBuffer::AudioBuffer(
    std::shared_ptr<const float[]> samples,
    const std::uint64_t sample_count)
    : samples_(std::move(samples)), sample_count_(sample_count) {
  if ((samples_ == nullptr) != (sample_count_ == 0)) {
    throw std::invalid_argument(
        "audio buffer pointer and sample count disagree");
  }
}

const float* AudioBuffer::data() const noexcept {
  return samples_.get();
}

std::uint64_t AudioBuffer::size() const noexcept {
  return sample_count_;
}

LeaseIdSequence::LeaseIdSequence(const std::uint64_t initial_value)
    : next_(initial_value) {
  if (initial_value == 0) {
    throw std::invalid_argument("lease identifier sequence must start above zero");
  }
}

std::uint64_t LeaseIdSequence::acquire() {
  std::uint64_t candidate = next_.load(std::memory_order_relaxed);
  while (true) {
    if (candidate == 0 ||
        candidate == std::numeric_limits<std::uint64_t>::max()) {
      fail_state(
          RequestStateErrorCode::lease_sequence_exhausted,
          "process-wide lease identifier sequence is exhausted");
    }
    if (next_.compare_exchange_weak(
            candidate,
            candidate + 1,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return candidate;
    }
  }
}

StreamingRequestState::StreamingRequestState(
    const std::uint64_t text_token_count,
    const std::uint32_t pcm_ring_capacity_frames,
    LeaseIdSequence& lease_ids)
    : text_token_count_(text_token_count),
      pcm_ring_capacity_frames_(pcm_ring_capacity_frames),
      lease_ids_(lease_ids) {
  if (text_token_count == 0) {
    throw std::invalid_argument("request text token count must be positive");
  }
  if (pcm_ring_capacity_frames < 8) {
    throw std::invalid_argument(
        "PCM ring capacity must hold one steady eight-frame chunk");
  }
}

bool StreamingRequestState::can_publish(
    const std::uint32_t codec_frames) const {
  std::scoped_lock lock(mutex_);
  return state_ == RequestLifecycleState::running &&
         !cancellation_requested_ &&
         codec_frames > 0 &&
         codec_frames <=
             pcm_ring_capacity_frames_ - occupied_codec_frames_;
}

void StreamingRequestState::validate_chunk_locked(
    const AudioChunk& chunk) const {
  if (state_ != RequestLifecycleState::running) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "audio cannot be published after the request becomes terminal");
  }
  if (chunk.codec_frame_count > 8) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "codec frame count must be in [0, 8]");
  }
  const bool terminal_control_marker =
      chunk.codec_frame_count == 0;
  if (terminal_control_marker &&
      (!chunk.final || chunk.first || chunk.samples.size() != 0)) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "a zero-frame chunk must be a non-FIRST FINAL control marker "
        "with no PCM");
  }
  const std::uint64_t expected_sample_count =
      static_cast<std::uint64_t>(chunk.codec_frame_count) *
      kCodecSamplesPerFrame;
  if (chunk.samples.size() != expected_sample_count) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "PCM sample count does not equal codec_frame_count * 1024");
  }
  if (chunk.first_sample_index != next_sample_index_) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "PCM sample ranges are not contiguous");
  }
  if (chunk.sequence != next_sequence_) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "audio sequence is not contiguous");
  }
  if (chunk.first != (next_sequence_ == 0)) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "FIRST flag does not match sequence zero");
  }
  if (chunk.first && chunk.codec_frame_count != 4) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "the first audio chunk must contain exactly four codec frames");
  }
  if (!chunk.first && !chunk.final &&
      chunk.codec_frame_count != 8) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "a non-terminal steady chunk must contain exactly eight codec frames");
  }
  if (chunk.codec_frame_count >
      pcm_ring_capacity_frames_ - occupied_codec_frames_) {
    fail_state(
        RequestStateErrorCode::backpressure,
        "PCM ring has insufficient free codec frames");
  }

  std::uint64_t previous_sample = chunk.first_sample_index;
  std::uint64_t previous_tokens = committed_text_tokens_;
  const std::uint64_t chunk_end =
      chunk.first_sample_index + expected_sample_count;
  if (!chunk.alignment_valid &&
      (!chunk.alignment_events.empty() ||
       chunk.committed_text_tokens != 0)) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "a chunk without valid alignment must not carry progress");
  }
  if (chunk.alignment_valid &&
      (chunk.committed_text_tokens < committed_text_tokens_ ||
       chunk.committed_text_tokens > text_token_count_)) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "lease-level alignment progress is not monotonic and in range");
  }
  if (chunk.alignment_valid && chunk.alignment_events.empty() &&
      chunk.committed_text_tokens != committed_text_tokens_) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "lease-level alignment progress cannot advance without an event");
  }
  for (const AlignmentProgress& event : chunk.alignment_events) {
    if (event.sample_index <= previous_sample ||
        event.sample_index > chunk_end ||
        event.sample_index % kCodecSamplesPerFrame != 0) {
      fail_state(
          RequestStateErrorCode::invalid_audio_chunk,
          "alignment event is outside its lease or not frame-aligned");
    }
    if (event.committed_text_tokens <= previous_tokens ||
        event.committed_text_tokens > text_token_count_) {
      fail_state(
          RequestStateErrorCode::invalid_audio_chunk,
          "alignment token progress is not a strict in-range advance");
    }
    previous_sample = event.sample_index;
    previous_tokens = event.committed_text_tokens;
  }
  if (chunk.alignment_valid && !chunk.alignment_events.empty() &&
      chunk.committed_text_tokens !=
          chunk.alignment_events.back().committed_text_tokens) {
    fail_state(
        RequestStateErrorCode::invalid_audio_chunk,
        "lease-level alignment progress does not equal the last event");
  }
}

bool StreamingRequestState::publish(AudioChunk chunk) {
  std::unique_lock lock(mutex_);
  if (cancellation_requested_) {
    // Cancellation acceptance closes publication atomically, but only the
    // synthesis worker knows when every already-armed CUDA event has drained.
    // Keep RUNNING here so a drain failure can still become FAILED instead of
    // exposing a premature CANCELLED terminal state.
    return false;
  }
  validate_chunk_locked(chunk);
  const std::uint64_t sample_count = chunk.samples.size();
  const std::uint32_t codec_frames = chunk.codec_frame_count;
  const bool final = chunk.final;
  if (chunk.alignment_valid) {
    committed_text_tokens_ = chunk.committed_text_tokens;
  }
  next_sample_index_ += sample_count;
  ++next_sequence_;
  generated_codec_frames_ += codec_frames;
  published_samples_ += sample_count;
  occupied_codec_frames_ += codec_frames;
  available_audio_.push_back(std::move(chunk));
  if (final) {
    state_ = RequestLifecycleState::completed;
  }
  increment_revision_locked();
  lock.unlock();
  revision_changed_.notify_all();
  return true;
}

void StreamingRequestState::discard_unleased_audio_locked() {
  for (const AudioChunk& chunk : available_audio_) {
    if (chunk.codec_frame_count > occupied_codec_frames_) {
      fail_state(
          RequestStateErrorCode::invalid_transition,
          "PCM ring accounting underflow");
    }
    occupied_codec_frames_ -= chunk.codec_frame_count;
  }
  available_audio_.clear();
}

void StreamingRequestState::complete_cancellation_locked() {
  if (state_ != RequestLifecycleState::running ||
      !cancellation_requested_) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "cancellation completion requires an accepted running request");
  }
  discard_unleased_audio_locked();
  state_ = RequestLifecycleState::cancelled;
  terminal_status_ = 10;
  terminal_error_stage_ = 6;
  terminal_error_message_.clear();
}

bool StreamingRequestState::request_cancellation() {
  std::unique_lock lock(mutex_);
  if (state_ == RequestLifecycleState::cancelled) {
    return true;
  }
  if (state_ != RequestLifecycleState::running) {
    return false;
  }
  if (cancellation_requested_) {
    return true;
  }
  cancellation_requested_ = true;
  discard_unleased_audio_locked();
  increment_revision_locked();
  lock.unlock();
  revision_changed_.notify_all();
  return true;
}

bool StreamingRequestState::cancellation_requested() const {
  std::scoped_lock lock(mutex_);
  return state_ == RequestLifecycleState::running &&
         cancellation_requested_;
}

bool StreamingRequestState::complete_cancellation_after_drain() {
  std::unique_lock lock(mutex_);
  if (state_ != RequestLifecycleState::running ||
      !cancellation_requested_) {
    return false;
  }
  complete_cancellation_locked();
  increment_revision_locked();
  lock.unlock();
  revision_changed_.notify_all();
  return true;
}

void StreamingRequestState::fail(
    const std::int32_t terminal_status,
    const std::int32_t terminal_error_stage,
    std::string terminal_error_message) {
  if (terminal_status <= 0 || terminal_status == 3 ||
      terminal_status == 10 || terminal_status == 11 ||
      terminal_status == 12 || terminal_error_stage <= 0 ||
      terminal_error_message.empty()) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "failed request requires a declared failure status, stage, and message");
  }
  std::unique_lock lock(mutex_);
  if (state_ != RequestLifecycleState::running) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "only a running request can fail");
  }
  discard_unleased_audio_locked();
  state_ = RequestLifecycleState::failed;
  terminal_status_ = terminal_status;
  terminal_error_stage_ = terminal_error_stage;
  terminal_error_message_ = std::move(terminal_error_message);
  increment_revision_locked();
  lock.unlock();
  revision_changed_.notify_all();
}

RequestStateSnapshot StreamingRequestState::snapshot_locked() const {
  if (available_audio_.size() >
      std::numeric_limits<std::uint32_t>::max()) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "available audio lease count exceeds UINT32");
  }
  return RequestStateSnapshot{
      .revision = revision_,
      .state = state_,
      .available_audio_leases =
          static_cast<std::uint32_t>(available_audio_.size()),
      .generated_codec_frames = generated_codec_frames_,
      .published_samples = published_samples_,
      .committed_text_tokens = committed_text_tokens_,
      .terminal_status = terminal_status_,
      .terminal_error_stage = terminal_error_stage_,
      .terminal_error_message = terminal_error_message_,
  };
}

RequestStateSnapshot StreamingRequestState::snapshot() const {
  std::scoped_lock lock(mutex_);
  return snapshot_locked();
}

bool StreamingRequestState::wait_for_revision(
    const std::uint64_t after_revision,
    const std::chrono::nanoseconds timeout,
    RequestStateSnapshot& output) const {
  std::unique_lock lock(mutex_);
  const bool changed = revision_changed_.wait_for(
      lock,
      timeout,
      [&]() { return revision_ > after_revision; });
  if (!changed) {
    return false;
  }
  output = snapshot_locked();
  return true;
}

AudioLeaseView StreamingRequestState::acquire_audio() {
  std::unique_lock lock(mutex_);
  if (available_audio_.empty()) {
    fail_state(
        RequestStateErrorCode::no_audio,
        "no audio chunk is currently available");
  }
  const std::uint64_t lease_id = lease_ids_.acquire();
  AudioChunk chunk = std::move(available_audio_.front());
  available_audio_.pop_front();
  const auto [iterator, inserted] = live_leases_.emplace(
      lease_id,
      StoredLease{
          .lease_id = lease_id,
          .chunk = std::move(chunk),
      });
  if (!inserted) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "process-wide lease identifier was reused");
  }
  increment_revision_locked();
  const StoredLease& stored = iterator->second;
  const std::vector<AlignmentProgress>& events =
      stored.chunk.alignment_events;
  AudioLeaseView view{
      .lease_id = stored.lease_id,
      .samples = stored.chunk.samples.data(),
      .sample_count = stored.chunk.samples.size(),
      .first_sample_index = stored.chunk.first_sample_index,
      .sequence = stored.chunk.sequence,
      .first = stored.chunk.first,
      .final = stored.chunk.final,
      .alignment_valid = stored.chunk.alignment_valid,
      .committed_text_tokens = stored.chunk.committed_text_tokens,
      .alignment_events = events.empty() ? nullptr : events.data(),
      .alignment_event_count = events.size(),
  };
  lock.unlock();
  revision_changed_.notify_all();
  return view;
}

void StreamingRequestState::release_audio(
    const std::uint64_t lease_id) {
  std::unique_lock lock(mutex_);
  const auto found = live_leases_.find(lease_id);
  if (found == live_leases_.end()) {
    fail_state(
        RequestStateErrorCode::unknown_lease,
        "lease identifier is not live on this request");
  }
  if (found->second.chunk.codec_frame_count >
      occupied_codec_frames_) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "PCM ring accounting underflow");
  }
  occupied_codec_frames_ -= found->second.chunk.codec_frame_count;
  live_leases_.erase(found);
  increment_revision_locked();
  lock.unlock();
  revision_changed_.notify_all();
}

bool StreamingRequestState::is_terminal() const {
  std::scoped_lock lock(mutex_);
  return is_terminal_state(state_);
}

bool StreamingRequestState::has_live_leases() const {
  std::scoped_lock lock(mutex_);
  return !live_leases_.empty();
}

std::uint32_t StreamingRequestState::occupied_codec_frames() const {
  std::scoped_lock lock(mutex_);
  return occupied_codec_frames_;
}

void StreamingRequestState::increment_revision_locked() {
  if (revision_ == std::numeric_limits<std::uint64_t>::max()) {
    fail_state(
        RequestStateErrorCode::invalid_transition,
        "request revision is exhausted");
  }
  ++revision_;
}

}  // namespace magpie_tts_rt

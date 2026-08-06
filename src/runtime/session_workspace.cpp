#include "runtime/session_workspace.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "runtime/engine_execution.hpp"

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const SessionWorkspaceErrorCode code,
    const std::string_view logical_name,
    const std::string_view detail) {
  return "session workspace creation failed [code=" +
         std::string(to_string(code)) + ", logical_name=" +
         std::string(logical_name) + "]: " + std::string(detail);
}

[[noreturn]] void fail(
    const SessionWorkspaceErrorCode code,
    const std::string_view logical_name,
    const std::string& detail) {
  throw SessionWorkspaceError(
      code, std::string(logical_name), detail);
}

[[nodiscard]] const TensorSpec& require_tensor(
    const EngineManifest& engine,
    const std::string_view name,
    const bool input) {
  const std::vector<TensorSpec>& tensors =
      input ? engine.inputs : engine.outputs;
  const auto found = std::find_if(
      tensors.begin(),
      tensors.end(),
      [&](const TensorSpec& tensor) {
        return tensor.name == name;
      });
  if (found == tensors.end()) {
    fail(
        SessionWorkspaceErrorCode::missing_tensor,
        name,
        "engine " + engine.name +
            (input ? " omits input " : " omits output ") +
            std::string(name));
  }
  return *found;
}

[[nodiscard]] EngineShapeParameters maximum_parameters(
    const RuntimeBundleManifest& manifest) {
  return EngineShapeParameters{
      .text_token_count = manifest.limits.maximum_text_tokens,
      .codec_frame_count = manifest.codec.tail_max_frames,
      .codec_hop_length_samples = manifest.codec.hop_length_samples,
  };
}

[[nodiscard]] std::uint64_t storage_bytes(
    const RuntimeBundleManifest& manifest,
    const EngineManifest& engine,
    const TensorSpec& tensor) {
  return tensor_storage_bytes(
      tensor.dtype,
      resolve_tensor_shape(
          engine.role, tensor, maximum_parameters(manifest)));
}

void require_storage_compatible(
    const RuntimeBundleManifest& manifest,
    const EngineManifest& left_engine,
    const TensorSpec& left,
    const EngineManifest& right_engine,
    const TensorSpec& right,
    const std::string_view logical_name) {
  if (left.dtype != right.dtype ||
      storage_bytes(manifest, left_engine, left) !=
          storage_bytes(manifest, right_engine, right)) {
    fail(
        SessionWorkspaceErrorCode::incompatible_tensor_contract,
        logical_name,
        left_engine.name + "/" + left.name +
            " is not storage-compatible with " +
            right_engine.name + "/" + right.name);
  }
}

[[nodiscard]] void* allocate_tensor(
    DeviceMemoryRegistry& memory,
    const RuntimeBundleManifest& manifest,
    const std::string& logical_name,
    const EngineManifest& engine,
    const TensorSpec& tensor) {
  return memory.allocate(
      logical_name, storage_bytes(manifest, engine, tensor));
}

[[nodiscard]] void* allocate_bytes(
    DeviceMemoryRegistry& memory,
    const std::string& logical_name,
    const std::uint64_t bytes) {
  return memory.allocate(logical_name, bytes);
}

}  // namespace

std::string_view to_string(
    const SessionWorkspaceErrorCode code) noexcept {
  switch (code) {
    case SessionWorkspaceErrorCode::invalid_memory_limit:
      return "invalid_memory_limit";
    case SessionWorkspaceErrorCode::missing_engine:
      return "missing_engine";
    case SessionWorkspaceErrorCode::missing_tensor:
      return "missing_tensor";
    case SessionWorkspaceErrorCode::incompatible_tensor_contract:
      return "incompatible_tensor_contract";
    case SessionWorkspaceErrorCode::cuda_event_failure:
      return "cuda_event_failure";
  }
  return "unknown";
}

SessionWorkspaceError::SessionWorkspaceError(
    const SessionWorkspaceErrorCode code,
    std::string logical_name,
    std::string detail)
    : std::runtime_error(
          error_message(code, logical_name, detail)),
      code_(code),
      logical_name_(std::move(logical_name)),
      detail_(std::move(detail)) {}

SessionWorkspaceErrorCode
SessionWorkspaceError::code() const noexcept {
  return code_;
}

const std::string&
SessionWorkspaceError::logical_name() const noexcept {
  return logical_name_;
}

const std::string& SessionWorkspaceError::detail() const noexcept {
  return detail_;
}

SessionWorkspace::SessionWorkspace(
    const RuntimeBundleManifest& manifest,
    const std::uint64_t context_device_memory_bytes)
    : memory_(
          context_device_memory_bytes <
                  manifest.limits.maximum_device_memory_bytes
              ? manifest.limits.maximum_device_memory_bytes -
                    context_device_memory_bytes
              : throw SessionWorkspaceError(
                    SessionWorkspaceErrorCode::invalid_memory_limit,
                    "maximum_device_memory_bytes",
                    "context memory leaves no authenticated space for state")),
      context_device_memory_bytes_(context_device_memory_bytes) {
  const EngineManifest& text =
      require_engine(manifest, EngineRole::text_encoder);
  const EngineManifest& prefill =
      require_engine(manifest, EngineRole::main_decoder_prefill);
  const EngineManifest& step =
      require_engine(manifest, EngineRole::main_decoder_step);
  const EngineManifest& local =
      require_engine(manifest, EngineRole::local_ar_16);
  const EngineManifest& codec_initial =
      require_engine(manifest, EngineRole::nanocodec_initial_4);
  const EngineManifest& codec_steady =
      require_engine(manifest, EngineRole::nanocodec_steady_8);
  const EngineManifest& codec_tail =
      require_engine(manifest, EngineRole::nanocodec_tail_1_8);

  const TensorSpec& text_ids =
      require_tensor(text, "text_token_ids", true);
  const TensorSpec& source_mask =
      require_tensor(text, "text_mask", true);
  const TensorSpec& encoded =
      require_tensor(text, "text_condition", false);
  const TensorSpec& cfg_condition =
      require_tensor(prefill, "condition", true);
  const TensorSpec& cfg_mask =
      require_tensor(prefill, "condition_mask", true);
  const TensorSpec& prefill_hidden =
      require_tensor(prefill, "last_hidden", false);
  const TensorSpec& local_hidden =
      require_tensor(local, "decoder_hidden", true);
  const TensorSpec& prefill_alignment =
      require_tensor(
          prefill, manifest.alignment.prefill_output_binding, false);
  const TensorSpec& step_alignment =
      require_tensor(
          step, manifest.alignment.step_alignment_output_binding, false);
  const TensorSpec& step_prior =
      require_tensor(
          step, manifest.alignment.step_prior_input_binding, true);
  require_storage_compatible(
      manifest,
      prefill,
      prefill_hidden,
      local,
      local_hidden,
      "decoder_hidden");
  require_storage_compatible(
      manifest,
      prefill,
      prefill_alignment,
      step,
      step_alignment,
      "alignment_scores");

  text_token_ids_ = allocate_tensor(
      memory_, manifest, "text_token_ids", text, text_ids);
  text_mask_ = allocate_tensor(
      memory_, manifest, "text_mask", text, source_mask);
  text_condition_ = allocate_tensor(
      memory_, manifest, "text_condition", text, encoded);
  cfg_condition_ = allocate_tensor(
      memory_, manifest, "cfg_condition", prefill, cfg_condition);
  condition_mask_ = allocate_tensor(
      memory_, manifest, "condition_mask", prefill, cfg_mask);
  decoder_hidden_ = allocate_tensor(
      memory_, manifest, "decoder_hidden", local, local_hidden);
  alignment_scores_ = allocate_tensor(
      memory_,
      manifest,
      "alignment_scores",
      prefill,
      prefill_alignment);
  alignment_prior_ = allocate_tensor(
      memory_, manifest, "alignment_prior", step, step_prior);

  const std::uint64_t text_slots =
      manifest.limits.maximum_text_tokens;
  alignment_counters_ = static_cast<std::uint32_t*>(
      allocate_bytes(
          memory_,
          "alignment_counters",
          text_slots * sizeof(std::uint32_t)));
  last_attended_ = static_cast<std::uint32_t*>(
      allocate_bytes(
          memory_, "last_attended", sizeof(std::uint32_t)));
  for (std::size_t step = 0;
       step < kMaximumDecoderStepsPerEmission;
       ++step) {
    const std::string suffix = std::to_string(step);
    attended_token_indices_[step] =
        static_cast<std::int64_t*>(
            allocate_bytes(
                memory_,
                "attended_token_index_" + suffix,
                sizeof(std::int64_t)));
    unfinished_text_steps_[step] = static_cast<bool*>(
        allocate_bytes(
            memory_,
            "unfinished_text_" + suffix,
            sizeof(bool)));
    alignment_invalid_steps_[step] =
        static_cast<std::int32_t*>(
            allocate_bytes(
                memory_,
                "alignment_invalid_" + suffix,
                sizeof(std::int32_t)));
  }
  generation_finished_ = static_cast<bool*>(
      allocate_bytes(
          memory_, "generation_finished", sizeof(bool)));
  forbid_eos_ = static_cast<bool*>(
      allocate_bytes(memory_, "forbid_eos", sizeof(bool)));
  rng_seed_ = static_cast<std::int64_t*>(
      allocate_bytes(memory_, "rng_seed", sizeof(std::int64_t)));
  rng_counter_[0] = static_cast<std::int64_t*>(
      allocate_bytes(memory_, "rng_counter_a", sizeof(std::int64_t)));
  rng_counter_[1] = static_cast<std::int64_t*>(
      allocate_bytes(memory_, "rng_counter_b", sizeof(std::int64_t)));

  const TensorSpec& local_codes =
      require_tensor(local, "codec_tokens", false);
  const TensorSpec& local_invalid =
      require_tensor(local, "invalid_rows", false);
  const TensorSpec& local_eos =
      require_tensor(local, "end_frame_index", false);
  local_codec_tokens_ = allocate_tensor(
      memory_, manifest, "local_codec_tokens", local, local_codes);
  const std::uint64_t local_invalid_bytes =
      storage_bytes(manifest, local, local_invalid);
  const std::uint64_t local_eos_bytes =
      storage_bytes(manifest, local, local_eos);
  for (std::size_t step = 0;
       step < kMaximumDecoderStepsPerEmission;
       ++step) {
    const std::string suffix = std::to_string(step);
    local_invalid_rows_steps_[step] =
        static_cast<std::int32_t*>(
            allocate_bytes(
                memory_,
                "local_invalid_rows_" + suffix,
                local_invalid_bytes));
    local_end_frame_indices_[step] =
        static_cast<std::int32_t*>(
            allocate_bytes(
                memory_,
                "local_end_frame_index_" + suffix,
                local_eos_bytes));
  }
  for (std::size_t slot = 0;
       slot < kGenerationBatchSlotCount;
       ++slot) {
    aggregate_codec_tokens_[slot] = allocate_bytes(
        memory_,
        "aggregate_codec_tokens_" + std::to_string(slot),
        static_cast<std::uint64_t>(
            manifest.local_ar.codebooks_per_frame) *
            manifest.codec.steady_frames * sizeof(std::int64_t));
  }
  codec_input_tokens_ = allocate_bytes(
      memory_,
      "codec_input_tokens",
      static_cast<std::uint64_t>(manifest.local_ar.codebooks_per_frame) *
          manifest.codec.steady_frames * sizeof(std::int64_t));

  const TensorSpec& initial_pcm =
      require_tensor(codec_initial, "pcm", false);
  const TensorSpec& steady_pcm =
      require_tensor(codec_steady, "pcm", false);
  const TensorSpec& tail_pcm =
      require_tensor(codec_tail, "pcm", false);
  const std::uint64_t initial_pcm_bytes =
      storage_bytes(manifest, codec_initial, initial_pcm);
  const std::uint64_t steady_pcm_bytes =
      storage_bytes(manifest, codec_steady, steady_pcm);
  const std::uint64_t tail_pcm_bytes =
      storage_bytes(manifest, codec_tail, tail_pcm);
  const std::uint64_t maximum_pcm_bytes =
      std::max({initial_pcm_bytes, steady_pcm_bytes, tail_pcm_bytes});
  codec_pcm_ = allocate_bytes(
      memory_, "codec_pcm", maximum_pcm_bytes);
  const TensorSpec& initial_valid =
      require_tensor(
          codec_initial, "valid_sample_length", false);
  const TensorSpec& steady_valid =
      require_tensor(
          codec_steady, "valid_sample_length", false);
  const TensorSpec& tail_valid =
      require_tensor(codec_tail, "valid_sample_length", false);
  require_storage_compatible(
      manifest,
      codec_initial,
      initial_valid,
      codec_steady,
      steady_valid,
      "codec_valid_sample_length");
  require_storage_compatible(
      manifest,
      codec_initial,
      initial_valid,
      codec_tail,
      tail_valid,
      "codec_valid_sample_length");
  codec_valid_sample_length_ = static_cast<std::int64_t*>(
      allocate_tensor(
          memory_,
          manifest,
          "codec_valid_sample_length",
          codec_initial,
          initial_valid));
  pinned_pcm_capacity_samples_ =
      static_cast<std::uint64_t>(manifest.codec.steady_frames) *
      manifest.codec.hop_length_samples;
  pinned_pcm_ = std::make_unique<PinnedAllocation>(
      pinned_pcm_capacity_samples_ * sizeof(float));
  for (std::size_t slot = 0;
       slot < kDecoderPositionHostSlotCount;
       ++slot) {
    pinned_decoder_positions_[slot] =
        std::make_unique<PinnedAllocation>(sizeof(std::int64_t));
    if (reinterpret_cast<std::uintptr_t>(
            pinned_decoder_positions_[slot]->data()) %
            256U !=
        0U) {
      fail(
          SessionWorkspaceErrorCode::incompatible_tensor_contract,
          "decoder_position",
          "TensorRT HOST shape input slot " +
              std::to_string(slot) +
              " is not 256-byte aligned");
    }
  }
  pinned_valid_sample_length_ =
      std::make_unique<PinnedAllocation>(sizeof(std::int64_t));
  for (std::size_t slot = 0;
       slot < kGenerationBatchSlotCount;
       ++slot) {
    pinned_attended_token_indices_[slot] =
        std::make_unique<PinnedAllocation>(
            kMaximumDecoderStepsPerEmission *
            sizeof(std::int64_t));
    pinned_alignment_invalid_steps_[slot] =
        std::make_unique<PinnedAllocation>(
            kMaximumDecoderStepsPerEmission *
            sizeof(std::int32_t));
    pinned_local_invalid_rows_[slot] =
        std::make_unique<PinnedAllocation>(
            kMaximumDecoderStepsPerEmission *
            sizeof(std::int32_t));
    pinned_end_frame_indices_[slot] =
        std::make_unique<PinnedAllocation>(
            kMaximumDecoderStepsPerEmission *
            sizeof(std::int32_t));
  }
  pinned_startup_codec_code_capacity_ =
      static_cast<std::uint64_t>(
          manifest.local_ar.codebooks_per_frame) *
      manifest.codec.steady_frames;
  pinned_startup_codec_codes_ =
      std::make_unique<PinnedAllocation>(
          pinned_startup_codec_code_capacity_ *
          sizeof(std::int64_t));

  decoder_layers_.reserve(manifest.kv_cache.layer_bindings.size());
  for (const KvLayerBindings& binding :
       manifest.kv_cache.layer_bindings) {
    const TensorSpec& prefill_key =
        require_tensor(
            prefill, binding.prefill_self_key_output, false);
    const TensorSpec& prefill_value =
        require_tensor(
            prefill, binding.prefill_self_value_output, false);
    const TensorSpec& prefill_mask =
        require_tensor(
            prefill, binding.prefill_self_mask_output, false);
    const TensorSpec& step_key_input =
        require_tensor(step, binding.step_self_key_input, true);
    const TensorSpec& step_value_input =
        require_tensor(step, binding.step_self_value_input, true);
    const TensorSpec& step_mask_input =
        require_tensor(step, binding.step_self_mask_input, true);
    const TensorSpec& step_key_output =
        require_tensor(step, binding.step_self_key_output, false);
    const TensorSpec& step_value_output =
        require_tensor(step, binding.step_self_value_output, false);
    const TensorSpec& step_mask_output =
        require_tensor(step, binding.step_self_mask_output, false);
    require_storage_compatible(
        manifest,
        prefill,
        prefill_key,
        step,
        step_key_input,
        binding.step_self_key_input);
    require_storage_compatible(
        manifest,
        step,
        step_key_input,
        step,
        step_key_output,
        binding.step_self_key_output);
    require_storage_compatible(
        manifest,
        prefill,
        prefill_value,
        step,
        step_value_input,
        binding.step_self_value_input);
    require_storage_compatible(
        manifest,
        step,
        step_value_input,
        step,
        step_value_output,
        binding.step_self_value_output);
    require_storage_compatible(
        manifest,
        prefill,
        prefill_mask,
        step,
        step_mask_input,
        binding.step_self_mask_input);
    require_storage_compatible(
        manifest,
        step,
        step_mask_input,
        step,
        step_mask_output,
        binding.step_self_mask_output);

    const std::string layer =
        std::to_string(binding.layer_index);
    DecoderLayerWorkspace workspace{
        .bindings = binding,
        .self_key = {
            allocate_tensor(
                memory_,
                manifest,
                "self_key_" + layer + "_a",
                prefill,
                prefill_key),
            allocate_tensor(
                memory_,
                manifest,
                "self_key_" + layer + "_b",
                step,
                step_key_output),
        },
        .self_value = {
            allocate_tensor(
                memory_,
                manifest,
                "self_value_" + layer + "_a",
                prefill,
                prefill_value),
            allocate_tensor(
                memory_,
                manifest,
                "self_value_" + layer + "_b",
                step,
                step_value_output),
        },
        .self_mask = {
            allocate_tensor(
                memory_,
                manifest,
                "self_mask_" + layer + "_a",
                prefill,
                prefill_mask),
            allocate_tensor(
                memory_,
                manifest,
                "self_mask_" + layer + "_b",
                step,
                step_mask_output),
        },
        .cross_key = nullptr,
        .cross_value = nullptr,
    };
    const TensorSpec& prefill_cross_key =
        require_tensor(
            prefill, binding.prefill_cross_key_output, false);
    const TensorSpec& prefill_cross_value =
        require_tensor(
            prefill, binding.prefill_cross_value_output, false);
    const TensorSpec& step_cross_key =
        require_tensor(step, binding.step_cross_key_input, true);
    const TensorSpec& step_cross_value =
        require_tensor(step, binding.step_cross_value_input, true);
    require_storage_compatible(
        manifest,
        prefill,
        prefill_cross_key,
        step,
        step_cross_key,
        binding.step_cross_key_input);
    require_storage_compatible(
        manifest,
        prefill,
        prefill_cross_value,
        step,
        step_cross_value,
        binding.step_cross_value_input);
    workspace.cross_key = allocate_tensor(
        memory_,
        manifest,
        "cross_key_" + layer,
        prefill,
        prefill_cross_key);
    workspace.cross_value = allocate_tensor(
        memory_,
        manifest,
        "cross_value_" + layer,
        prefill,
        prefill_cross_value);
    decoder_layers_.push_back(std::move(workspace));
  }

  codec_states_.reserve(manifest.codec.state_bindings.size());
  for (const CodecStateBinding& binding :
       manifest.codec.state_bindings) {
    const TensorSpec& initial_output =
        require_tensor(
            codec_initial, binding.initial_output_binding, false);
    const TensorSpec& steady_input =
        require_tensor(
            codec_steady, binding.steady_input_binding, true);
    const TensorSpec& steady_output =
        require_tensor(
            codec_steady, binding.steady_output_binding, false);
    const TensorSpec& tail_input =
        require_tensor(codec_tail, binding.tail_input_binding, true);
    const TensorSpec& tail_output =
        require_tensor(codec_tail, binding.tail_output_binding, false);
    if (initial_output.dtype != binding.dtype ||
        initial_output.shape != binding.shape) {
      fail(
          SessionWorkspaceErrorCode::incompatible_tensor_contract,
          binding.logical_name,
          "codec state binding differs from initial engine output");
    }
    require_storage_compatible(
        manifest,
        codec_initial,
        initial_output,
        codec_steady,
        steady_input,
        binding.logical_name);
    require_storage_compatible(
        manifest,
        codec_steady,
        steady_input,
        codec_steady,
        steady_output,
        binding.logical_name);
    require_storage_compatible(
        manifest,
        codec_steady,
        steady_input,
        codec_tail,
        tail_input,
        binding.logical_name);
    require_storage_compatible(
        manifest,
        codec_tail,
        tail_input,
        codec_tail,
        tail_output,
        binding.logical_name);
    codec_states_.push_back(CodecStateWorkspace{
        .binding = binding,
        .state = {
            allocate_tensor(
                memory_,
                manifest,
                "codec_state." + binding.logical_name + ".a",
                codec_initial,
                initial_output),
            allocate_tensor(
                memory_,
                manifest,
                "codec_state." + binding.logical_name + ".b",
                codec_steady,
                steady_output),
        },
    });
  }
  // Create the raw CUDA events only after all other potentially-throwing
  // workspace construction. A partial event loop is explicitly unwound below.
  for (std::size_t slot = 0;
       slot < kDecoderPositionHostSlotCount;
       ++slot) {
    const cudaError_t event_status = cudaEventCreateWithFlags(
        &decoder_position_consumed_events_[slot],
        cudaEventDisableTiming);
    if (event_status != cudaSuccess) {
      for (std::size_t created = 0; created < slot; ++created) {
        static_cast<void>(
            cudaEventDestroy(
                decoder_position_consumed_events_[created]));
        decoder_position_consumed_events_[created] = nullptr;
      }
      fail(
          SessionWorkspaceErrorCode::cuda_event_failure,
          "decoder_position",
          "failed to create input-consumed event for slot " +
              std::to_string(slot) + ": " +
              cudaGetErrorString(event_status));
    }
  }
}

SessionWorkspace::~SessionWorkspace() {
  // SessionResources synchronizes both session streams before workspace
  // teardown. The events therefore no longer guard live TensorRT inputs here.
  for (cudaEvent_t& event : decoder_position_consumed_events_) {
    if (event != nullptr) {
      static_cast<void>(cudaEventDestroy(event));
      event = nullptr;
    }
  }
}

void* SessionWorkspace::text_token_ids() const noexcept {
  return text_token_ids_;
}
void* SessionWorkspace::text_mask() const noexcept {
  return text_mask_;
}
void* SessionWorkspace::text_condition() const noexcept {
  return text_condition_;
}
void* SessionWorkspace::cfg_condition() const noexcept {
  return cfg_condition_;
}
void* SessionWorkspace::condition_mask() const noexcept {
  return condition_mask_;
}
void* SessionWorkspace::decoder_hidden() const noexcept {
  return decoder_hidden_;
}
DecoderPositionInput SessionWorkspace::acquire_decoder_position(
    const std::int64_t position) {
  const std::size_t slot = next_decoder_position_slot_;
  cudaEvent_t event = decoder_position_consumed_events_[slot];
  const cudaError_t query_status = cudaEventQuery(event);
  if (query_status == cudaErrorNotReady) {
    fail(
        SessionWorkspaceErrorCode::cuda_event_failure,
        "decoder_position",
        "attempted to reuse live HOST input slot " +
            std::to_string(slot) +
            "; decoder batches must reach their codes-ready boundary "
            "before a position slot wraps");
  } else if (query_status != cudaSuccess) {
    fail(
        SessionWorkspaceErrorCode::cuda_event_failure,
        "decoder_position",
        "failed to query input-consumed event for slot " +
            std::to_string(slot) + ": " +
            cudaGetErrorString(query_status));
  }
  auto* address = static_cast<std::int64_t*>(
      pinned_decoder_positions_[slot]->data());
  *address = position;
  next_decoder_position_slot_ =
      (slot + 1U) % kDecoderPositionHostSlotCount;
  return DecoderPositionInput{
      .address = address,
      .input_consumed_event = event,
      .slot = slot,
  };
}
void* SessionWorkspace::alignment_scores() const noexcept {
  return alignment_scores_;
}
void* SessionWorkspace::alignment_prior() const noexcept {
  return alignment_prior_;
}
std::uint32_t* SessionWorkspace::alignment_counters() const noexcept {
  return alignment_counters_;
}
std::uint32_t* SessionWorkspace::last_attended() const noexcept {
  return last_attended_;
}
std::int64_t* SessionWorkspace::attended_token_index(
    const std::size_t step_slot) const {
  if (step_slot >= kMaximumDecoderStepsPerEmission) {
    throw std::out_of_range(
        "alignment step slot must be in [0,4)");
  }
  return attended_token_indices_.at(step_slot);
}
bool* SessionWorkspace::unfinished_text(
    const std::size_t step_slot) const {
  if (step_slot >= kMaximumDecoderStepsPerEmission) {
    throw std::out_of_range(
        "unfinished-text step slot must be in [0,4)");
  }
  return unfinished_text_steps_.at(step_slot);
}
std::int32_t* SessionWorkspace::alignment_invalid(
    const std::size_t step_slot) const {
  if (step_slot >= kMaximumDecoderStepsPerEmission) {
    throw std::out_of_range(
        "alignment-invalid step slot must be in [0,4)");
  }
  return alignment_invalid_steps_.at(step_slot);
}
bool* SessionWorkspace::generation_finished() const noexcept {
  return generation_finished_;
}
bool* SessionWorkspace::forbid_eos() const noexcept {
  return forbid_eos_;
}
std::int64_t* SessionWorkspace::rng_seed() const noexcept {
  return rng_seed_;
}
std::int64_t* SessionWorkspace::rng_counter(
    const std::size_t index) const {
  if (index >= rng_counter_.size()) {
    throw std::out_of_range("RNG counter buffer index must be 0 or 1");
  }
  return rng_counter_[index];
}
void* SessionWorkspace::local_codec_tokens() const noexcept {
  return local_codec_tokens_;
}
std::int32_t* SessionWorkspace::local_invalid_rows(
    const std::size_t step_slot) const {
  if (step_slot >= kMaximumDecoderStepsPerEmission) {
    throw std::out_of_range(
        "Local AR invalid-row step slot must be in [0,4)");
  }
  return local_invalid_rows_steps_.at(step_slot);
}
std::int32_t* SessionWorkspace::local_end_frame_index(
    const std::size_t step_slot) const {
  if (step_slot >= kMaximumDecoderStepsPerEmission) {
    throw std::out_of_range(
        "Local AR EOS step slot must be in [0,4)");
  }
  return local_end_frame_indices_.at(step_slot);
}
void* SessionWorkspace::aggregate_codec_tokens(
    const std::size_t batch_slot) const {
  if (batch_slot >= aggregate_codec_tokens_.size()) {
    throw std::out_of_range(
        "generation batch slot must be 0 or 1");
  }
  return aggregate_codec_tokens_.at(batch_slot);
}
void* SessionWorkspace::codec_input_tokens() const noexcept {
  return codec_input_tokens_;
}
void* SessionWorkspace::codec_pcm() const noexcept {
  return codec_pcm_;
}
std::int64_t*
SessionWorkspace::codec_valid_sample_length() const noexcept {
  return codec_valid_sample_length_;
}
float* SessionWorkspace::pinned_pcm() const noexcept {
  return static_cast<float*>(pinned_pcm_->data());
}
std::uint64_t
SessionWorkspace::pinned_pcm_capacity_samples() const noexcept {
  return pinned_pcm_capacity_samples_;
}
std::int64_t*
SessionWorkspace::pinned_valid_sample_length() const noexcept {
  return static_cast<std::int64_t*>(
      pinned_valid_sample_length_->data());
}
std::int64_t*
SessionWorkspace::pinned_attended_token_indices(
    const std::size_t batch_slot) const {
  if (batch_slot >= pinned_attended_token_indices_.size()) {
    throw std::out_of_range(
        "generation diagnostic batch slot must be 0 or 1");
  }
  return static_cast<std::int64_t*>(
      pinned_attended_token_indices_.at(batch_slot)->data());
}
std::int32_t*
SessionWorkspace::pinned_alignment_invalid_steps(
    const std::size_t batch_slot) const {
  if (batch_slot >= pinned_alignment_invalid_steps_.size()) {
    throw std::out_of_range(
        "generation diagnostic batch slot must be 0 or 1");
  }
  return static_cast<std::int32_t*>(
      pinned_alignment_invalid_steps_.at(batch_slot)->data());
}
std::int32_t*
SessionWorkspace::pinned_local_invalid_rows(
    const std::size_t batch_slot) const {
  if (batch_slot >= pinned_local_invalid_rows_.size()) {
    throw std::out_of_range(
        "generation diagnostic batch slot must be 0 or 1");
  }
  return static_cast<std::int32_t*>(
      pinned_local_invalid_rows_.at(batch_slot)->data());
}
std::int32_t*
SessionWorkspace::pinned_end_frame_indices(
    const std::size_t batch_slot) const {
  if (batch_slot >= pinned_end_frame_indices_.size()) {
    throw std::out_of_range(
        "generation diagnostic batch slot must be 0 or 1");
  }
  return static_cast<std::int32_t*>(
      pinned_end_frame_indices_.at(batch_slot)->data());
}
std::int64_t*
SessionWorkspace::pinned_startup_codec_codes() const noexcept {
  return static_cast<std::int64_t*>(
      pinned_startup_codec_codes_->data());
}
std::uint64_t
SessionWorkspace::pinned_startup_codec_code_capacity() const noexcept {
  return pinned_startup_codec_code_capacity_;
}

std::vector<DecoderLayerWorkspace>&
SessionWorkspace::decoder_layers() noexcept {
  return decoder_layers_;
}
const std::vector<DecoderLayerWorkspace>&
SessionWorkspace::decoder_layers() const noexcept {
  return decoder_layers_;
}
std::vector<CodecStateWorkspace>&
SessionWorkspace::codec_states() noexcept {
  return codec_states_;
}
const std::vector<CodecStateWorkspace>&
SessionWorkspace::codec_states() const noexcept {
  return codec_states_;
}

std::uint64_t
SessionWorkspace::allocated_device_memory_bytes() const noexcept {
  return memory_.allocated_bytes();
}

std::uint64_t SessionWorkspace::total_device_memory_bytes() const noexcept {
  if (memory_.allocated_bytes() >
      std::numeric_limits<std::uint64_t>::max() -
          context_device_memory_bytes_) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return context_device_memory_bytes_ + memory_.allocated_bytes();
}

}  // namespace magpie_tts_rt

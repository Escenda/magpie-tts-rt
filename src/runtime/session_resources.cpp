#include "runtime/session_resources.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const SessionResourceErrorCode code,
    const std::string_view detail) {
  return "session resource creation failed [code=" +
         std::string(to_string(code)) + "]: " + std::string(detail);
}

[[noreturn]] void fail(
    const SessionResourceErrorCode code,
    const std::string& detail) {
  throw SessionResourceError(code, detail);
}

[[nodiscard]] std::mutex& graph_memory_accounting_mutex() {
  static std::mutex mutex;
  return mutex;
}

void require_cuda(
    const cudaError_t status,
    const SessionResourceErrorCode code,
    const std::string_view operation) {
  if (status != cudaSuccess) {
    fail(
        code,
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

[[nodiscard]] CudaGraphMemorySnapshot device_memory_snapshot() {
  int device = 0;
  require_cuda(
      cudaGetDevice(&device),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "query CUDA device for graph accounting");
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  require_cuda(
      cudaMemGetInfo(&free_bytes, &total_bytes),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "query free device memory for CUDA graph accounting");
  std::size_t graph_used_bytes = 0;
  std::size_t graph_reserved_bytes = 0;
  std::size_t graph_used_high_bytes = 0;
  std::size_t graph_reserved_high_bytes = 0;
  require_cuda(
      cudaDeviceGetGraphMemAttribute(
          device,
          cudaGraphMemAttrUsedMemCurrent,
          &graph_used_bytes),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "query used CUDA graph memory");
  require_cuda(
      cudaDeviceGetGraphMemAttribute(
          device,
          cudaGraphMemAttrReservedMemCurrent,
          &graph_reserved_bytes),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "query reserved CUDA graph memory");
  require_cuda(
      cudaDeviceGetGraphMemAttribute(
          device,
          cudaGraphMemAttrUsedMemHigh,
          &graph_used_high_bytes),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "query CUDA graph used-memory high watermark");
  require_cuda(
      cudaDeviceGetGraphMemAttribute(
          device,
          cudaGraphMemAttrReservedMemHigh,
          &graph_reserved_high_bytes),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "query CUDA graph reserved-memory high watermark");
  static_cast<void>(total_bytes);
  return CudaGraphMemorySnapshot{
      .free_bytes = static_cast<std::uint64_t>(free_bytes),
      .graph_used_bytes =
          static_cast<std::uint64_t>(graph_used_bytes),
      .graph_reserved_bytes =
          static_cast<std::uint64_t>(graph_reserved_bytes),
      .graph_used_high_bytes =
          static_cast<std::uint64_t>(graph_used_high_bytes),
      .graph_reserved_high_bytes =
          static_cast<std::uint64_t>(graph_reserved_high_bytes),
  };
}

void reset_graph_memory_high_watermarks() {
  int device = 0;
  require_cuda(
      cudaGetDevice(&device),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "query CUDA device for graph high-water reset");
  std::size_t zero = 0;
  require_cuda(
      cudaDeviceSetGraphMemAttribute(
          device, cudaGraphMemAttrUsedMemHigh, &zero),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "reset CUDA graph used-memory high watermark");
  require_cuda(
      cudaDeviceSetGraphMemAttribute(
          device, cudaGraphMemAttrReservedMemHigh, &zero),
      SessionResourceErrorCode::cuda_graph_memory_query_failed,
      "reset CUDA graph reserved-memory high watermark");
}

}  // namespace

std::string_view to_string(
    const SessionResourceErrorCode code) noexcept {
  switch (code) {
    case SessionResourceErrorCode::invalid_manifest_limit:
      return "invalid_manifest_limit";
    case SessionResourceErrorCode::device_memory_limit_exceeded:
      return "device_memory_limit_exceeded";
    case SessionResourceErrorCode::execution_context_creation_failed:
      return "execution_context_creation_failed";
    case SessionResourceErrorCode::cuda_stream_creation_failed:
      return "cuda_stream_creation_failed";
    case SessionResourceErrorCode::cuda_event_creation_failed:
      return "cuda_event_creation_failed";
    case SessionResourceErrorCode::cuda_stream_synchronization_failed:
      return "cuda_stream_synchronization_failed";
    case SessionResourceErrorCode::cuda_graph_memory_query_failed:
      return "cuda_graph_memory_query_failed";
    case SessionResourceErrorCode::missing_execution_context:
      return "missing_execution_context";
  }
  return "unknown";
}

SessionResourceError::SessionResourceError(
    const SessionResourceErrorCode code,
    std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

SessionResourceErrorCode SessionResourceError::code() const noexcept {
  return code_;
}

const std::string& SessionResourceError::detail() const noexcept {
  return detail_;
}

SessionResources::SessionResources(
    const std::vector<LoadedEngine>& engines,
    const RuntimeBundleManifest& manifest) {
  const LimitsManifest& limits = manifest.limits;
  maximum_device_memory_bytes_ =
      limits.maximum_device_memory_bytes;
  if (engines.empty() || limits.maximum_device_memory_bytes == 0) {
    fail(
        SessionResourceErrorCode::invalid_manifest_limit,
        "engines and a positive device-memory limit are required");
  }

  const auto account_context_memory = [&](const LoadedEngine& loaded) {
    const std::int64_t context_bytes =
        loaded.engine->getDeviceMemorySizeV2();
    if (context_bytes < 0) {
      fail(
          SessionResourceErrorCode::execution_context_creation_failed,
          loaded.name +
              " returned a negative context device-memory size");
    }
    const std::uint64_t unsigned_bytes =
        static_cast<std::uint64_t>(context_bytes);
    if (unsigned_bytes >
        std::numeric_limits<std::uint64_t>::max() -
            context_device_memory_bytes_) {
      fail(
          SessionResourceErrorCode::device_memory_limit_exceeded,
          "execution-context memory sum overflowed UINT64");
    }
    context_device_memory_bytes_ += unsigned_bytes;
  };
  for (const LoadedEngine& loaded : engines) {
    account_context_memory(loaded);
    if (loaded.role == EngineRole::main_decoder_step ||
        loaded.role == EngineRole::nanocodec_steady_8) {
      // A-to-B and B-to-A are distinct captured executions. TensorRT context
      // state is part of a CUDA Graph, so the reverse direction owns another
      // context and its full activation-memory charge.
      account_context_memory(loaded);
    }
  }
  if (context_device_memory_bytes_ >
      limits.maximum_device_memory_bytes) {
    fail(
        SessionResourceErrorCode::device_memory_limit_exceeded,
        "TensorRT contexts require " +
            std::to_string(context_device_memory_bytes_) +
            " bytes, exceeding manifest limit " +
            std::to_string(limits.maximum_device_memory_bytes));
  }

  require_cuda(
      cudaStreamCreateWithFlags(
          &generation_stream_, cudaStreamNonBlocking),
      SessionResourceErrorCode::cuda_stream_creation_failed,
      "create generation stream");
  try {
    require_cuda(
        cudaStreamCreateWithFlags(
            &codec_stream_, cudaStreamNonBlocking),
        SessionResourceErrorCode::cuda_stream_creation_failed,
        "create codec stream");
    for (std::size_t slot = 0;
         slot < kGenerationBatchSlotCount;
         ++slot) {
      require_cuda(
          cudaEventCreateWithFlags(
              &codes_ready_events_[slot],
              cudaEventDisableTiming | cudaEventBlockingSync),
          SessionResourceErrorCode::cuda_event_creation_failed,
          "create codes-ready event");
      require_cuda(
          cudaEventCreateWithFlags(
              &codes_consumed_events_[slot],
              cudaEventDisableTiming),
          SessionResourceErrorCode::cuda_event_creation_failed,
          "create codes-consumed event");
    }
    require_cuda(
        cudaEventCreateWithFlags(
            &audio_ready_event_,
            cudaEventDisableTiming | cudaEventBlockingSync),
        SessionResourceErrorCode::cuda_event_creation_failed,
        "create audio-ready event");

    contexts_.reserve(engines.size());
    for (const LoadedEngine& loaded : engines) {
      std::unique_ptr<nvinfer1::IExecutionContext> context(
          loaded.engine->createExecutionContext(
              nvinfer1::ExecutionContextAllocationStrategy::kSTATIC));
      if (context == nullptr) {
        fail(
            SessionResourceErrorCode::execution_context_creation_failed,
            "TensorRT returned a null context for " + loaded.name);
      }
      contexts_.push_back(EngineExecutionContext{
          .role = loaded.role,
          .name = loaded.name,
          .context = std::move(context),
      });
      if (loaded.role == EngineRole::main_decoder_step) {
        if (main_decoder_reverse_context_ != nullptr) {
          fail(
              SessionResourceErrorCode::execution_context_creation_failed,
              "more than one Main Decoder step engine was loaded");
        }
        main_decoder_reverse_context_.reset(
            loaded.engine->createExecutionContext(
                nvinfer1::ExecutionContextAllocationStrategy::kSTATIC));
        if (main_decoder_reverse_context_ == nullptr) {
          fail(
              SessionResourceErrorCode::execution_context_creation_failed,
              "TensorRT returned a null reverse-cache context for " +
                  loaded.name);
        }
      }
      if (loaded.role == EngineRole::nanocodec_steady_8) {
        if (codec_steady_reverse_context_ != nullptr) {
          fail(
              SessionResourceErrorCode::execution_context_creation_failed,
              "more than one steady NanoCodec engine was loaded");
        }
        codec_steady_reverse_context_.reset(
            loaded.engine->createExecutionContext(
                nvinfer1::ExecutionContextAllocationStrategy::kSTATIC));
        if (codec_steady_reverse_context_ == nullptr) {
          fail(
              SessionResourceErrorCode::execution_context_creation_failed,
              "TensorRT returned a null reverse-state context for " +
                  loaded.name);
        }
      }
    }
    if (main_decoder_reverse_context_ == nullptr) {
      fail(
          SessionResourceErrorCode::execution_context_creation_failed,
          "the Main Decoder step engine did not create its reverse-cache "
          "execution context");
    }
    if (codec_steady_reverse_context_ == nullptr) {
      fail(
          SessionResourceErrorCode::execution_context_creation_failed,
          "the steady NanoCodec engine did not create its reverse-state "
          "execution context");
    }
    workspace_ = std::make_unique<SessionWorkspace>(
        manifest, context_device_memory_bytes_);
  } catch (...) {
    contexts_.clear();
    main_decoder_reverse_context_.reset();
    codec_steady_reverse_context_.reset();
    if (audio_ready_event_ != nullptr) {
      static_cast<void>(cudaEventDestroy(audio_ready_event_));
      audio_ready_event_ = nullptr;
    }
    for (cudaEvent_t& event : codes_consumed_events_) {
      if (event != nullptr) {
        static_cast<void>(cudaEventDestroy(event));
        event = nullptr;
      }
    }
    for (cudaEvent_t& event : codes_ready_events_) {
      if (event != nullptr) {
        static_cast<void>(cudaEventDestroy(event));
        event = nullptr;
      }
    }
    if (codec_stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(codec_stream_));
      codec_stream_ = nullptr;
    }
    if (generation_stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(generation_stream_));
      generation_stream_ = nullptr;
    }
    throw;
  }
}

SessionResources::~SessionResources() {
  // A startup or enqueue failure can unwind before the request pipeline has
  // armed a completion event. Keep TensorRT contexts and workspace storage
  // alive until both session streams have stopped using them. This is a
  // teardown-only wait; healthy per-request execution uses event dependencies.
  if (!teardown_synchronized_) {
    if (generation_stream_ != nullptr) {
      static_cast<void>(
          cudaStreamSynchronize(generation_stream_));
    }
    if (codec_stream_ != nullptr) {
      static_cast<void>(
          cudaStreamSynchronize(codec_stream_));
    }
  }
  // Captured TensorRT work retains the context state and bound addresses.
  // Destroy it only after both streams are idle, and before either owner.
  nanocodec_graphs_.reset();
  local_ar_graph_.reset();
  main_decoder_graphs_.reset();
  workspace_.reset();
  codec_steady_reverse_context_.reset();
  main_decoder_reverse_context_.reset();
  contexts_.clear();
  if (audio_ready_event_ != nullptr) {
    static_cast<void>(cudaEventDestroy(audio_ready_event_));
  }
  for (cudaEvent_t event : codes_consumed_events_) {
    if (event != nullptr) {
      static_cast<void>(cudaEventDestroy(event));
    }
  }
  for (cudaEvent_t event : codes_ready_events_) {
    if (event != nullptr) {
      static_cast<void>(cudaEventDestroy(event));
    }
  }
  if (codec_stream_ != nullptr) {
    static_cast<void>(cudaStreamDestroy(codec_stream_));
  }
  if (generation_stream_ != nullptr) {
    static_cast<void>(cudaStreamDestroy(generation_stream_));
  }
}

void SessionResources::synchronize_for_teardown() {
  require_cuda(
      cudaStreamSynchronize(generation_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "synchronize generation stream for teardown");
  require_cuda(
      cudaStreamSynchronize(codec_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "synchronize codec stream for teardown");
  teardown_synchronized_ = true;
}

nvinfer1::IExecutionContext& SessionResources::context(
    const EngineRole role) {
  const auto found = std::find_if(
      contexts_.begin(),
      contexts_.end(),
      [role](const EngineExecutionContext& execution) {
        return execution.role == role;
      });
  if (found == contexts_.end()) {
    fail(
        SessionResourceErrorCode::missing_execution_context,
        "required engine role is absent: " +
            std::string(to_string(role)));
  }
  return *found->context;
}

nvinfer1::IExecutionContext& SessionResources::main_decoder_context(
    const std::size_t cache_input) {
  if (cache_input == 0U) {
    return context(EngineRole::main_decoder_step);
  }
  if (cache_input == 1U && main_decoder_reverse_context_ != nullptr) {
    return *main_decoder_reverse_context_;
  }
  fail(
      SessionResourceErrorCode::missing_execution_context,
      "Main Decoder cache input must be 0 or 1 and own its context");
}

nvinfer1::IExecutionContext& SessionResources::codec_steady_context(
    const std::size_t state_input) {
  if (state_input == 0U) {
    return context(EngineRole::nanocodec_steady_8);
  }
  if (state_input == 1U && codec_steady_reverse_context_ != nullptr) {
    return *codec_steady_reverse_context_;
  }
  fail(
      SessionResourceErrorCode::missing_execution_context,
      "steady NanoCodec state input must be 0 or 1 and own its context");
}

cudaStream_t SessionResources::generation_stream() const noexcept {
  return generation_stream_;
}

cudaStream_t SessionResources::codec_stream() const noexcept {
  return codec_stream_;
}

cudaEvent_t SessionResources::codes_ready_event(
    const std::size_t batch_slot) const {
  if (batch_slot >= codes_ready_events_.size()) {
    throw std::out_of_range(
        "generation batch slot must be 0 or 1");
  }
  return codes_ready_events_.at(batch_slot);
}

cudaEvent_t SessionResources::codes_consumed_event(
    const std::size_t batch_slot) const {
  if (batch_slot >= codes_consumed_events_.size()) {
    throw std::out_of_range(
        "generation batch slot must be 0 or 1");
  }
  return codes_consumed_events_.at(batch_slot);
}

cudaEvent_t SessionResources::audio_ready_event() const noexcept {
  return audio_ready_event_;
}

std::uint64_t
SessionResources::context_device_memory_bytes() const noexcept {
  return context_device_memory_bytes_;
}

std::uint64_t
SessionResources::cuda_graph_device_memory_bytes() const noexcept {
  return cuda_graph_device_memory_bytes_;
}

SessionWorkspace& SessionResources::workspace() noexcept {
  return *workspace_;
}

const SessionWorkspace& SessionResources::workspace() const noexcept {
  return *workspace_;
}

bool SessionResources::local_ar_graph_ready() const noexcept {
  return local_ar_graph_.ready();
}

bool SessionResources::main_decoder_warmed(
    const std::size_t cache_input) const {
  return main_decoder_graphs_.warmed(cache_input);
}

bool SessionResources::main_decoder_graph_ready(
    const std::size_t cache_input) const {
  return main_decoder_graphs_.ready(cache_input);
}

bool SessionResources::main_decoder_graphs_ready() const noexcept {
  return main_decoder_graphs_.ready();
}

bool SessionResources::nanocodec_initial_graph_ready() const noexcept {
  return nanocodec_graphs_.initial_ready();
}

bool SessionResources::nanocodec_steady_graph_ready(
    const std::size_t state_input) const {
  return nanocodec_graphs_.steady_ready(state_input);
}

bool SessionResources::nanocodec_graphs_ready() const noexcept {
  return nanocodec_graphs_.ready();
}

bool SessionResources::cuda_graph_memory_accounted() const noexcept {
  return graph_memory_accounted_;
}

void SessionResources::begin_main_decoder_request() {
  if (main_graph_memory_accounting_pending_) {
    fail(
        SessionResourceErrorCode::cuda_graph_memory_query_failed,
        "a preceding Main Decoder request retained graph-accounting "
        "ownership");
  }
  // run_synthesis_pipeline settles every generation boundary before it
  // returns. Destroying the preceding request's text-shape-specific graphs
  // here is therefore ordered after their final launch without adding a
  // healthy-path stream synchronization.
  main_decoder_graphs_.reset_checked();
}

void SessionResources::record_main_decoder_eager_warmup(
    const std::size_t cache_input) {
  nvinfer1::IExecutionContext& decoder_context =
      main_decoder_context(cache_input);
  main_decoder_graphs_.record_eager_warmup(
      cache_input, &decoder_context);
}

void SessionResources::capture_and_upload_main_decoder_graph(
    const std::size_t cache_input,
    const CudaGraphExecutable::EnqueueOperation& enqueue) {
  if (!main_graph_memory_accounting_pending_) {
    if (main_decoder_graphs_.ready()) {
      fail(
          SessionResourceErrorCode::cuda_graph_memory_query_failed,
          "Main Decoder graph accounting began after both graphs were "
          "already ready");
    }
    main_graph_uses_startup_accounting_ =
        graph_memory_accounting_pending_ && !graph_memory_accounted_;
    if (!main_graph_uses_startup_accounting_) {
      if (!graph_memory_accounted_) {
        fail(
            SessionResourceErrorCode::cuda_graph_memory_query_failed,
            "normal Main Decoder capture began before startup graph-memory "
            "accounting completed");
      }
      // Arm failure cleanup before either mutex acquisition or CUDA memory
      // queries can throw. Pipeline failure handling will then release any
      // acquired lock and destroy a partial graph.
      main_graph_memory_accounting_pending_ = true;
      main_graph_memory_accounting_lock_ =
          std::unique_lock<std::mutex>(
              graph_memory_accounting_mutex());
      reset_graph_memory_high_watermarks();
    } else {
      main_graph_memory_accounting_pending_ = true;
    }
  }

  nvinfer1::IExecutionContext& decoder_context =
      main_decoder_context(cache_input);
  main_decoder_graphs_.capture(
      generation_stream_, cache_input, &decoder_context, enqueue);
}

void SessionResources::launch_main_decoder_graph(
    const std::size_t cache_input) {
  main_decoder_graphs_.launch(generation_stream_, cache_input);
  if (!main_decoder_graphs_.ready() ||
      !main_graph_memory_accounting_pending_) {
    return;
  }

  if (main_graph_uses_startup_accounting_) {
    // Startup's aggregate scope began before Local AR capture and remains
    // active through both NanoCodec directions. The startup finalizer performs
    // the one aggregate measurement after every graph has executed.
    main_graph_memory_accounting_pending_ = false;
    main_graph_uses_startup_accounting_ = false;
    return;
  }

  // Dynamic T changes TensorRT's captured launch parameters. Measure every
  // request after both request-specific graphs have launched; the startup
  // fixture's graph size is never assumed to bound another text length.
  require_cuda(
      cudaStreamSynchronize(generation_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete request-specific Main Decoder graph capture");
  const CudaGraphMemorySnapshot after = device_memory_snapshot();
  const std::uint64_t observed_bytes =
      observed_cuda_graph_memory_growth(
          aggregate_graph_memory_baseline_, after);
  const std::uint64_t explicit_session_bytes =
      workspace_->total_device_memory_bytes();
  if (!cuda_graph_memory_fits_budget(
          maximum_device_memory_bytes_,
          explicit_session_bytes,
          observed_bytes)) {
    fail(
        SessionResourceErrorCode::device_memory_limit_exceeded,
        "session contexts and workspace require " +
            std::to_string(explicit_session_bytes) +
            " bytes and request-specific CUDA graphs raised observed graph "
            "memory to " + std::to_string(observed_bytes) +
            " bytes, exceeding manifest limit " +
            std::to_string(maximum_device_memory_bytes_));
  }
  cuda_graph_device_memory_bytes_ =
      std::max(cuda_graph_device_memory_bytes_, observed_bytes);
  main_graph_memory_accounting_pending_ = false;
  main_graph_memory_accounting_lock_.unlock();
}

void SessionResources::abort_main_decoder_request() noexcept {
  if (!main_graph_memory_accounting_pending_) {
    return;
  }
  // A failed capture/enqueue may leave request work in flight without a
  // codes-ready event. This failure-only wait makes graph destruction and a
  // later request boundary safe; it is never used to continue generation.
  if (generation_stream_ != nullptr) {
    static_cast<void>(cudaStreamSynchronize(generation_stream_));
  }
  main_decoder_graphs_.reset();
  main_graph_memory_accounting_pending_ = false;
  main_graph_uses_startup_accounting_ = false;
  if (main_graph_memory_accounting_lock_.owns_lock()) {
    main_graph_memory_accounting_lock_.unlock();
  }
}

void SessionResources::begin_cuda_graph_memory_accounting() {
  if (graph_memory_accounting_pending_ || graph_memory_accounted_ ||
      graph_memory_accounting_lock_.owns_lock()) {
    fail(
        SessionResourceErrorCode::cuda_graph_memory_query_failed,
        "startup graph-memory accounting may begin exactly once");
  }
  graph_memory_accounting_lock_ =
      std::unique_lock<std::mutex>(graph_memory_accounting_mutex());
  reset_graph_memory_high_watermarks();
  aggregate_graph_memory_baseline_ = device_memory_snapshot();
  graph_memory_accounting_pending_ = true;
}

void SessionResources::capture_and_upload_local_ar_graph() {
  if (!graph_memory_accounting_pending_ || graph_memory_accounted_) {
    fail(
        SessionResourceErrorCode::cuda_graph_memory_query_failed,
        "Local AR capture occurred outside startup graph-memory accounting");
  }
  // TensorRT performs deferred setup during the first enqueue after binding
  // context state. The aggregate accounting scope already includes that
  // warmup and every lazily-created Main Decoder cuBLAS bank.
  require_cuda(
      cudaStreamSynchronize(generation_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete Local AR graph warmup");
  nvinfer1::IExecutionContext& local_context =
      context(EngineRole::local_ar_16);
  local_ar_graph_.capture_and_upload(
      generation_stream_,
      [&local_context](const cudaStream_t stream) noexcept {
        return local_context.enqueueV3(stream);
      });
  require_cuda(
      cudaStreamSynchronize(generation_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete Local AR graph upload");
}

void SessionResources::capture_and_upload_nanocodec_initial_graph() {
  if (!graph_memory_accounting_pending_ || graph_memory_accounted_) {
    fail(
        SessionResourceErrorCode::cuda_graph_memory_query_failed,
        "NanoCodec initial graph capture was outside startup graph-memory "
        "accounting");
  }
  require_cuda(
      cudaStreamSynchronize(codec_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete NanoCodec initial graph warmup");
  nvinfer1::IExecutionContext& initial_context =
      context(EngineRole::nanocodec_initial_4);
  nanocodec_graphs_.capture_initial(
      codec_stream_,
      &initial_context,
      [&initial_context](const cudaStream_t stream) noexcept {
        return initial_context.enqueueV3(stream);
      });
  require_cuda(
      cudaStreamSynchronize(codec_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete NanoCodec initial graph upload");
}

void SessionResources::capture_and_upload_nanocodec_steady_graph(
    const std::size_t state_input) {
  if (!graph_memory_accounting_pending_ || graph_memory_accounted_) {
    fail(
        SessionResourceErrorCode::cuda_graph_memory_query_failed,
        "NanoCodec steady graph capture was outside startup graph-memory "
        "accounting");
  }
  require_cuda(
      cudaStreamSynchronize(codec_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete NanoCodec steady graph warmup");
  nvinfer1::IExecutionContext& steady_context =
      codec_steady_context(state_input);
  nanocodec_graphs_.capture_steady(
      codec_stream_,
      state_input,
      &steady_context,
      [&steady_context](const cudaStream_t stream) noexcept {
        return steady_context.enqueueV3(stream);
      });
  require_cuda(
      cudaStreamSynchronize(codec_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete NanoCodec steady graph upload");
}

void SessionResources::finalize_cuda_graph_memory_accounting() {
  if (!main_decoder_graphs_.ready() ||
      !local_ar_graph_.ready() || !nanocodec_graphs_.ready() ||
      main_graph_memory_accounting_pending_ ||
      !graph_memory_accounting_pending_ ||
      graph_memory_accounted_) {
    fail(
        SessionResourceErrorCode::cuda_graph_memory_query_failed,
        "required CUDA graphs were not all ready for exactly one startup "
        "memory-accounting finalization");
  }
  // The startup golden has replayed the graph through the production path.
  // Complete it before observing current and high-water device usage.
  require_cuda(
      cudaStreamSynchronize(generation_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete graph-backed startup golden");
  require_cuda(
      cudaStreamSynchronize(codec_stream_),
      SessionResourceErrorCode::cuda_stream_synchronization_failed,
      "complete graph-backed NanoCodec startup golden");
  const CudaGraphMemorySnapshot after = device_memory_snapshot();
  cuda_graph_device_memory_bytes_ =
      observed_cuda_graph_memory_growth(
          aggregate_graph_memory_baseline_, after);

  const std::uint64_t explicit_session_bytes =
      workspace_->total_device_memory_bytes();
  if (!cuda_graph_memory_fits_budget(
          maximum_device_memory_bytes_,
          explicit_session_bytes,
          cuda_graph_device_memory_bytes_)) {
    fail(
        SessionResourceErrorCode::device_memory_limit_exceeded,
        "session contexts and workspace require " +
            std::to_string(explicit_session_bytes) +
            " bytes and the required CUDA graphs added " +
            std::to_string(cuda_graph_device_memory_bytes_) +
            " bytes, exceeding manifest limit " +
            std::to_string(maximum_device_memory_bytes_));
  }
  graph_memory_accounting_pending_ = false;
  graph_memory_accounted_ = true;
  graph_memory_accounting_lock_.unlock();
}

void SessionResources::launch_local_ar_graph() {
  local_ar_graph_.launch(generation_stream_);
}

void SessionResources::launch_nanocodec_initial_graph() {
  nanocodec_graphs_.launch_initial(codec_stream_);
}

void SessionResources::launch_nanocodec_steady_graph(
    const std::size_t state_input) {
  nanocodec_graphs_.launch_steady(codec_stream_, state_input);
}

}  // namespace magpie_tts_rt

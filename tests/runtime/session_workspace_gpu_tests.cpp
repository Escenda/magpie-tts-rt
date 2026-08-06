#include "manifest/manifest.hpp"
#include "runtime/session_workspace.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_cuda(
    const cudaError_t status,
    const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        operation + ": " + cudaGetErrorString(status));
  }
}

void test_workspace(const std::filesystem::path& manifest_path) {
  const magpie_tts_rt::RuntimeBundleManifest manifest =
      magpie_tts_rt::load_runtime_bundle_manifest(manifest_path);
  std::size_t free_before = 0;
  std::size_t total_before = 0;
  require_cuda(
      cudaMemGetInfo(&free_before, &total_before),
      "query CUDA memory before workspace");
  {
    magpie_tts_rt::SessionWorkspace workspace(manifest, 0);
    require(
        workspace.allocated_device_memory_bytes() > 0,
        "workspace did not account for device allocations");
    require(
        workspace.total_device_memory_bytes() ==
            workspace.allocated_device_memory_bytes(),
        "workspace total does not include the expected context bytes");
    require(
        workspace.decoder_layers().size() ==
            manifest.kv_cache.layers,
        "decoder layer workspace count mismatch");
    require(
        workspace.codec_states().size() ==
            manifest.codec.state_bindings.size(),
        "codec state workspace count mismatch");
    require(
        workspace.text_token_ids() != nullptr &&
            workspace.decoder_hidden() != nullptr &&
            workspace.aggregate_codec_tokens(0) != nullptr &&
            workspace.aggregate_codec_tokens(1) != nullptr &&
            workspace.aggregate_codec_tokens(0) !=
                workspace.aggregate_codec_tokens(1) &&
            workspace.codec_pcm() != nullptr,
        "required device buffer is null or generation slots alias");
    std::array<magpie_tts_rt::DecoderPositionInput, 4>
        position_inputs{};
    std::array<std::int64_t, 4> expected_positions{
        218, 219, 220, 221};
    std::int64_t* copied_positions = nullptr;
    cudaStream_t position_stream = nullptr;
    require_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&copied_positions),
            expected_positions.size() * sizeof(std::int64_t)),
        "allocate copied decoder positions");
    require_cuda(
        cudaStreamCreateWithFlags(
            &position_stream, cudaStreamNonBlocking),
        "create decoder-position test stream");
    for (std::size_t index = 0;
         index < position_inputs.size();
         ++index) {
      position_inputs[index] =
          workspace.acquire_decoder_position(
              expected_positions[index]);
      require(
          position_inputs[index].slot == index,
          "decoder position slot did not advance");
      require(
          reinterpret_cast<std::uintptr_t>(
              position_inputs[index].address) %
                  256U ==
              0U,
          "decoder position HOST binding is not 256-byte aligned");
      cudaPointerAttributes position_attributes{};
      require_cuda(
          cudaPointerGetAttributes(
              &position_attributes,
              position_inputs[index].address),
          "inspect decoder position HOST binding");
      require(
          position_attributes.type == cudaMemoryTypeHost,
          "decoder position was allocated in the device arena");
      for (std::size_t earlier = 0; earlier < index; ++earlier) {
        require(
            position_inputs[index].address !=
                position_inputs[earlier].address,
            "decoder position HOST slots alias");
      }
      // This H2D read and event record model TensorRT's asynchronous input
      // consumption. Four distinct positions are submitted without a stream
      // synchronization between them.
      require_cuda(
          cudaMemcpyAsync(
              copied_positions + index,
              position_inputs[index].address,
              sizeof(std::int64_t),
              cudaMemcpyHostToDevice,
              position_stream),
          "copy decoder position from HOST input");
      require_cuda(
          cudaEventRecord(
              position_inputs[index].input_consumed_event,
              position_stream),
          "record decoder-position input consumption");
    }
    // The pipeline reaches a codes-ready boundary between decoder batches.
    // Model that boundary here before reusing the four HOST input slots.
    require_cuda(
        cudaStreamSynchronize(position_stream),
        "complete decoder-position batch");
    const magpie_tts_rt::DecoderPositionInput wrapped =
        workspace.acquire_decoder_position(222);
    require(
        wrapped.slot == 0 &&
            wrapped.address == position_inputs[0].address &&
            *wrapped.address == 222,
        "decoder position slot ring did not wait and wrap at four");
    std::array<std::int64_t, 4> observed_positions{};
    require_cuda(
        cudaMemcpy(
            observed_positions.data(),
            copied_positions,
            observed_positions.size() * sizeof(std::int64_t),
            cudaMemcpyDeviceToHost),
        "copy consumed decoder positions");
    require(
        observed_positions == expected_positions,
        "an in-flight decoder position HOST slot was overwritten");
    require_cuda(
        cudaStreamDestroy(position_stream),
        "destroy decoder-position test stream");
    require_cuda(
        cudaFree(copied_positions),
        "free copied decoder positions");
    std::size_t free_during = 0;
    std::size_t total_during = 0;
    require_cuda(
        cudaMemGetInfo(&free_during, &total_during),
        "query CUDA memory during workspace");
    require(
        total_during == total_before && free_during < free_before,
        "workspace allocation was not visible to CUDA");
  }
  require_cuda(cudaDeviceSynchronize(), "synchronize workspace teardown");
}

}  // namespace

int main(const int argc, const char* const* argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "expected the runtime manifest fixture path");
    }
    test_workspace(argv[1]);
  } catch (const std::exception& error) {
    std::cerr << "session workspace GPU test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}

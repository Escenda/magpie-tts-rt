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
            workspace.canonical_local_invalid_rows() != nullptr &&
            workspace.canonical_local_end_frame_index() != nullptr &&
            workspace.canonical_local_invalid_rows() !=
                workspace.local_invalid_rows(0) &&
            workspace.canonical_local_end_frame_index() !=
                workspace.local_end_frame_index(0) &&
            workspace.decoder_position() != nullptr &&
            workspace.codec_pcm() != nullptr,
        "required device buffer is null or canonical buffers alias diagnostics");
    cudaPointerAttributes position_attributes{};
    require_cuda(
        cudaPointerGetAttributes(
            &position_attributes,
            workspace.decoder_position()),
        "inspect Main Decoder position binding");
    require(
        position_attributes.type == cudaMemoryTypeDevice,
        "Main Decoder position is not a DEVICE tensor");
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

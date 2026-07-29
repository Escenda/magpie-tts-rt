#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "manifest/manifest.hpp"

namespace {

constexpr int kUsageError = 2;
constexpr int kManifestError = 3;
constexpr int kUnexpectedError = 4;

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: mtt-validate-manifest <runtime-bundle-manifest.json>\n";
    return kUsageError;
  }

  try {
    const auto manifest =
        magpie_tts_rt::load_runtime_bundle_manifest(std::filesystem::path(argv[1]));
    std::cout << "valid runtime bundle manifest"
              << " schema=" << manifest.schema_version
              << " bundle_id=" << manifest.bundle_id
              << " engines=" << manifest.engines.size() << '\n';
    return EXIT_SUCCESS;
  } catch (const magpie_tts_rt::ManifestError& error) {
    std::cerr << error.what() << '\n';
    return kManifestError;
  } catch (const std::exception& error) {
    std::cerr << "unexpected manifest validator failure: " << error.what() << '\n';
    return kUnexpectedError;
  }
}

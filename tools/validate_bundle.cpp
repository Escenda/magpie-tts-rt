#include "bundle/bundle.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr
        << "usage: mtt-validate-bundle /path/to/runtime-bundle "
           "<trusted-manifest-sha256>\n";
    return EXIT_FAILURE;
  }

  try {
    const magpie_tts_rt::VerifiedRuntimeBundle bundle =
        magpie_tts_rt::load_and_verify_runtime_bundle(argv[1], argv[2]);
    std::cout << "bundle_id=" << bundle.manifest.bundle_id << '\n'
              << "manifest_sha256=" << bundle.manifest_snapshot.sha256 << '\n'
              << "verified_artifacts=" << bundle.artifacts.size() << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "bundle/bundle.hpp"
#include "runtime/model_loader.hpp"

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: mtt_plugin_owner_gpu_tests "
          "/absolute/runtime-bundle <manifest-sha256> <cuda-device>");
    }
    const std::int32_t cuda_device = std::stoi(argv[3]);

    auto first_bundle =
        magpie_tts_rt::load_and_verify_runtime_bundle(
            argv[1], argv[2]);
    auto second_bundle =
        magpie_tts_rt::load_and_verify_runtime_bundle(
            argv[1], argv[2]);
    magpie_tts_rt::RuntimePluginState first;
    magpie_tts_rt::RuntimePluginState second;
    magpie_tts_rt::set_plugin_owner_test_fault(
        magpie_tts_rt::PluginOwnerTestFault::
            after_preparation_before_registration);
    bool allocation_fault_observed = false;
    try {
      first.authenticate_and_register(
          first_bundle, cuda_device);
    } catch (const std::bad_alloc&) {
      allocation_fault_observed = true;
    }
    require(
        allocation_fault_observed,
        "injected pre-registration allocation fault was not observed");

    std::barrier start(3);
    std::exception_ptr first_error;
    std::exception_ptr second_error;

    std::thread first_thread([&]() {
      start.arrive_and_wait();
      try {
        first.authenticate_and_register(
            first_bundle, cuda_device);
      } catch (...) {
        first_error = std::current_exception();
      }
    });
    std::thread second_thread([&]() {
      start.arrive_and_wait();
      try {
        second.authenticate_and_register(
            second_bundle, cuda_device);
      } catch (...) {
        second_error = std::current_exception();
      }
    });
    start.arrive_and_wait();
    first_thread.join();
    second_thread.join();
    if (first_error != nullptr) {
      std::rethrow_exception(first_error);
    }
    if (second_error != nullptr) {
      std::rethrow_exception(second_error);
    }

    require(
        first.sha256() == second.sha256(),
        "simultaneous runtimes did not share one plugin digest");
    require(
        first.abi_version() == second.abi_version(),
        "simultaneous runtimes did not share one plugin ABI");

    auto sequential_bundle =
        magpie_tts_rt::load_and_verify_runtime_bundle(
            argv[1], argv[2]);
    magpie_tts_rt::RuntimePluginState sequential;
    sequential.authenticate_and_register(
        sequential_bundle, cuda_device);
    require(
        sequential.sha256() == first.sha256() &&
            sequential.abi_version() == first.abi_version(),
        "sequential runtime did not reuse the process-global plugin owner");
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "process-global plugin owner test failed: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

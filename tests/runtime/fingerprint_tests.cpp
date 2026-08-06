#include "runtime/fingerprint.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "magpie-tts-rt-fingerprint-XXXXXX")
            .string();
    pattern.push_back('\0');
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void write_text(
    const std::filesystem::path& path,
    const std::string& contents) {
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("unable to create test os-release");
  }
  stream << contents;
  if (!stream) {
    throw std::runtime_error("unable to write test os-release");
  }
}

template <typename Function>
void require_failure(
    Function&& function,
    const std::string& expected_fragment) {
  try {
    function();
  } catch (const magpie_tts_rt::RuntimeFingerprintError& error) {
    require(
        std::string(error.what()).find(expected_fragment) !=
            std::string::npos,
        "failure did not contain expected diagnostic");
    return;
  }
  throw std::runtime_error("expected RuntimeFingerprintError");
}

}  // namespace

int main() {
  try {
    TemporaryDirectory temporary;
    const std::filesystem::path os_release =
        temporary.path() / "os-release";

    write_text(
        os_release,
        "NAME=\"Ubuntu\"\n"
        "ID=ubuntu\n"
        "VERSION_ID=\"24.04\"\n");
    const auto [os_name, os_version] =
        magpie_tts_rt::collect_linux_distribution_fingerprint(
            os_release);
    require(os_name == "linux", "unexpected operating-system name");
    require(
        os_version == "ubuntu-24.04",
        "unexpected operating-system version");

    write_text(
        os_release,
        "ID=ubuntu\n"
        "ID=ubuntu\n"
        "VERSION_ID=24.04\n");
    require_failure(
        [&]() {
          static_cast<void>(
              magpie_tts_rt::collect_linux_distribution_fingerprint(
                  os_release));
        },
        "repeats field ID");

    write_text(os_release, "ID=ubuntu\n");
    require_failure(
        [&]() {
          static_cast<void>(
              magpie_tts_rt::collect_linux_distribution_fingerprint(
                  os_release));
        },
        "missing required field VERSION_ID");

    write_text(
        os_release,
        "ID=debian\n"
        "VERSION_ID=12\n");
    require_failure(
        [&]() {
          static_cast<void>(
              magpie_tts_rt::collect_linux_distribution_fingerprint(
                  os_release));
        },
        "only Ubuntu is accepted");

    write_text(
        os_release,
        "ID=ubuntu\n"
        "VERSION_ID=\"24.04\n");
    require_failure(
        [&]() {
          static_cast<void>(
              magpie_tts_rt::collect_linux_distribution_fingerprint(
                  os_release));
        },
        "unterminated quoted value");
  } catch (const std::exception& error) {
    std::cerr << "fingerprint test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

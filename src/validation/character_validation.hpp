#pragma once

#include <cstddef>
#include <string_view>

namespace magpie_tts_rt::character_validation {

[[nodiscard]] constexpr bool is_ascii_digit(const char value) noexcept {
  return value >= '0' && value <= '9';
}

[[nodiscard]] constexpr bool is_ascii_alpha(const char value) noexcept {
  return (value >= 'A' && value <= 'Z') ||
         (value >= 'a' && value <= 'z');
}

[[nodiscard]] constexpr bool is_ascii_alphanumeric(
    const char value) noexcept {
  return is_ascii_alpha(value) || is_ascii_digit(value);
}

[[nodiscard]] constexpr bool is_identifier_tail(
    const char value) noexcept {
  return is_ascii_alphanumeric(value) || value == '.' || value == '_' ||
         value == '-';
}

[[nodiscard]] constexpr bool is_identifier(
    const std::string_view value) noexcept {
  if (value.empty() || value.size() > 128U ||
      !is_ascii_alphanumeric(value.front())) {
    return false;
  }
  for (const char character : value.substr(1)) {
    if (!is_identifier_tail(character)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool is_lowercase_sha256(
    const std::string_view value) noexcept {
  if (value.size() != 64U) {
    return false;
  }
  for (const char character : value) {
    if (!(is_ascii_digit(character) ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

constexpr void consume_ascii_digits(
    const std::string_view value,
    std::size_t& offset) noexcept {
  while (offset < value.size() && is_ascii_digit(value[offset])) {
    ++offset;
  }
}

[[nodiscard]] constexpr bool is_dotted_numeric_version(
    const std::string_view value) noexcept {
  std::size_t offset = 0;
  consume_ascii_digits(value, offset);
  if (offset == 0) {
    return false;
  }

  std::size_t component_count = 0;
  while (offset < value.size() && value[offset] == '.') {
    ++offset;
    const std::size_t component_start = offset;
    consume_ascii_digits(value, offset);
    if (offset == component_start) {
      return false;
    }
    ++component_count;
  }
  if (component_count == 0) {
    return false;
  }
  if (offset == value.size()) {
    return true;
  }
  if (value[offset] != '-' && value[offset] != '+') {
    return false;
  }
  ++offset;
  if (offset == value.size()) {
    return false;
  }
  for (; offset < value.size(); ++offset) {
    if (!is_identifier_tail(value[offset])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool is_major_minor_version(
    const std::string_view value) noexcept {
  std::size_t offset = 0;
  consume_ascii_digits(value, offset);
  if (offset == 0 || offset == value.size() || value[offset] != '.') {
    return false;
  }
  ++offset;
  const std::size_t minor_start = offset;
  consume_ascii_digits(value, offset);
  return offset > minor_start && offset == value.size();
}

[[nodiscard]] constexpr unsigned int decimal_pair(
    const std::string_view value,
    const std::size_t offset) noexcept {
  return static_cast<unsigned int>(value[offset] - '0') * 10U +
         static_cast<unsigned int>(value[offset + 1U] - '0');
}

[[nodiscard]] constexpr bool is_rfc3339_utc_lexeme(
    const std::string_view value) noexcept {
  if (value.size() < 20U || value[4] != '-' || value[7] != '-' ||
      value[10] != 'T' || value[13] != ':' || value[16] != ':') {
    return false;
  }
  constexpr std::size_t digit_offsets[] = {
      0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U,
      11U, 12U, 14U, 15U, 17U, 18U,
  };
  for (const std::size_t offset : digit_offsets) {
    if (!is_ascii_digit(value[offset])) {
      return false;
    }
  }
  if (decimal_pair(value, 5U) < 1U || decimal_pair(value, 5U) > 12U ||
      decimal_pair(value, 8U) < 1U || decimal_pair(value, 8U) > 31U ||
      decimal_pair(value, 11U) > 23U ||
      decimal_pair(value, 14U) > 59U ||
      decimal_pair(value, 17U) > 59U) {
    return false;
  }
  if (value.size() == 20U) {
    return value[19] == 'Z';
  }
  if (value.size() < 22U || value[19] != '.' || value.back() != 'Z') {
    return false;
  }
  for (std::size_t offset = 20U; offset + 1U < value.size(); ++offset) {
    if (!is_ascii_digit(value[offset])) {
      return false;
    }
  }
  return true;
}

}  // namespace magpie_tts_rt::character_validation

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gram {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

struct Status {
  bool ok = true;
  std::string message;

  static Status Ok() { return {}; }
  static Status Error(std::string msg) { return Status{.ok = false, .message = std::move(msg)}; }
};

}  // namespace gram

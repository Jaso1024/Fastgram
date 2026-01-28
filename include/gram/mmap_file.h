#pragma once

#include <cstddef>
#include <string>

#include "gram/common.h"

namespace gram {

class MmapFile {
 public:
  MmapFile() = default;
  explicit MmapFile(std::string path);

  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;

  MmapFile(MmapFile&& other) noexcept;
  MmapFile& operator=(MmapFile&& other) noexcept;

  ~MmapFile();

  [[nodiscard]] const std::string& path() const { return path_; }
  [[nodiscard]] const u8* data() const { return data_; }
  [[nodiscard]] u8* data_mut() { return data_; }
  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] bool valid() const { return data_ != nullptr; }

  [[nodiscard]] static Status AdviseRandom(const u8* addr, std::size_t length);

 private:
  void Reset();

  std::string path_;
  u8* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace gram

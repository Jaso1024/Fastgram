#include "gram/mmap_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>
#include <utility>

namespace gram {

MmapFile::MmapFile(std::string path) : path_(std::move(path)) {
  int fd = open(path_.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("open failed: " + path_);
  }
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("fstat failed: " + path_);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ == 0) {
    close(fd);
    throw std::runtime_error("empty file: " + path_);
  }
  void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    throw std::runtime_error("mmap failed: " + path_);
  }
  data_ = static_cast<u8*>(p);
  (void)AdviseRandom(data_, size_);
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : path_(std::move(other.path_)), data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  Reset();
  path_ = std::move(other.path_);
  data_ = other.data_;
  size_ = other.size_;
  other.data_ = nullptr;
  other.size_ = 0;
  return *this;
}

MmapFile::~MmapFile() { Reset(); }

void MmapFile::Reset() {
  if (data_ && size_ != 0) {
    munmap(data_, size_);
  }
  data_ = nullptr;
  size_ = 0;
  path_.clear();
}

Status MmapFile::AdviseRandom(const u8* addr, std::size_t length) {
  if (!addr || length == 0) {
    return Status::Ok();
  }
  int rc = madvise(const_cast<u8*>(addr), length, MADV_RANDOM);
  if (rc != 0) {
    return Status::Error("madvise failed");
  }
  return Status::Ok();
}

}  // namespace gram

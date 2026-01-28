#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gram/common.h"
#include "gram/mmap_file.h"

namespace gram {

struct Shard {
  MmapFile ds;
  MmapFile sa;
  MmapFile od;
  std::optional<MmapFile> mt;
  std::optional<MmapFile> om;
  std::optional<MmapFile> unigram;

  u64 tok_cnt = 0;
  u64 ds_size = 0;
  u8 ptr_size = 0;
  u64 doc_cnt = 0;
};

struct IndexConfig {
  std::vector<std::string> index_dirs;
  u64 version = 4;
  std::size_t token_width = 2;
  u64 eos_token_id = 0;
  u64 vocab_size = 65535;
  bool load_to_ram = false;
  std::size_t num_threads = 0;
};

class Index {
 public:
  explicit Index(IndexConfig cfg);

  [[nodiscard]] const IndexConfig& cfg() const { return cfg_; }
  [[nodiscard]] const std::vector<Shard>& shards() const { return shards_; }
  [[nodiscard]] std::size_t num_shards() const { return shards_.size(); }
  [[nodiscard]] const std::optional<MmapFile>& unigram_ranges() const { return unigram_ranges_; }

  [[nodiscard]] Status Load();

 private:
  Status LoadIndexDir(const std::filesystem::path& dir);

  IndexConfig cfg_;
  std::vector<Shard> shards_;
  std::optional<MmapFile> unigram_ranges_;
};

}  // namespace gram

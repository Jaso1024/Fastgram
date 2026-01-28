#include "gram/index.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Token>
std::uint64_t ReadPtr(const gram::Shard& shard, std::uint64_t rank) {
  const auto* src = shard.sa.data() + rank * shard.ptr_size;
  switch (shard.ptr_size) {
    case 1:
      return src[0];
    case 2: {
      std::uint16_t v;
      std::memcpy(&v, src, sizeof(v));
      return v;
    }
    case 3:
      return static_cast<std::uint64_t>(src[0]) | (static_cast<std::uint64_t>(src[1]) << 8) |
             (static_cast<std::uint64_t>(src[2]) << 16);
    case 4: {
      std::uint32_t v;
      std::memcpy(&v, src, sizeof(v));
      return v;
    }
    case 5: {
      std::uint32_t lo;
      std::memcpy(&lo, src, sizeof(lo));
      return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(src[4]) << 32);
    }
    case 6: {
      std::uint32_t lo;
      std::memcpy(&lo, src, sizeof(lo));
      std::uint16_t hi;
      std::memcpy(&hi, src + 4, sizeof(hi));
      return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
    }
    case 7: {
      std::uint32_t lo;
      std::memcpy(&lo, src, sizeof(lo));
      std::uint16_t hi;
      std::memcpy(&hi, src + 4, sizeof(hi));
      return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32) |
             (static_cast<std::uint64_t>(src[6]) << 48);
    }
    case 8: {
      std::uint64_t v;
      std::memcpy(&v, src, sizeof(v));
      return v;
    }
    default:
      throw std::runtime_error("unsupported ptr_size");
  }
}

template <typename Token>
int Run(const std::string& index_dir, const std::string& out_path) {
  gram::IndexConfig cfg;
  cfg.index_dirs = {index_dir};
  cfg.version = 4;
  cfg.token_width = sizeof(Token);
  cfg.eos_token_id = 0;
  cfg.vocab_size = (sizeof(Token) == 1) ? 255 : (sizeof(Token) == 2) ? 65535 : 0;

  gram::Index index(cfg);
  auto st = index.Load();
  if (!st.ok) {
    std::cerr << st.message << "\n";
    return 2;
  }

  const std::uint64_t token_space = static_cast<std::uint64_t>(1) << (8 * sizeof(Token));
  const std::size_t shard_count = index.num_shards();
  const std::size_t cells = static_cast<std::size_t>(token_space) * shard_count;

  std::vector<std::uint64_t> starts(cells, std::numeric_limits<std::uint64_t>::max());
  std::vector<std::uint64_t> ends(cells, 0);

  for (std::size_t s = 0; s < shard_count; s++) {
    const auto& shard = index.shards()[s];
    const auto* ds = reinterpret_cast<const Token*>(shard.ds.data());
    for (std::uint64_t rank = 0; rank < shard.tok_cnt; rank++) {
      const std::uint64_t ptr = ReadPtr<Token>(shard, rank);
      const Token tok = ds[ptr / sizeof(Token)];
      const std::size_t ix = static_cast<std::size_t>(tok) * shard_count + s;
      if (starts[ix] == std::numeric_limits<std::uint64_t>::max()) {
        starts[ix] = rank;
      }
      ends[ix] = rank + 1;
    }
  }

  const std::filesystem::path out = out_path.empty()
                                       ? (std::filesystem::path(index_dir) / "unigram_ranges.bin")
                                       : std::filesystem::path(out_path);
  const std::filesystem::path tmp = out.string() + ".tmp";
  std::filesystem::create_directories(out.parent_path());

  FILE* f = std::fopen(tmp.string().c_str(), "wb");
  if (!f) {
    std::cerr << "failed to open output\n";
    return 2;
  }
  for (std::uint64_t tok = 0; tok < token_space; tok++) {
    for (std::size_t s = 0; s < shard_count; s++) {
      const std::size_t ix = static_cast<std::size_t>(tok) * shard_count + s;
      std::uint64_t start = starts[ix];
      std::uint64_t end = ends[ix];
      if (start == std::numeric_limits<std::uint64_t>::max()) {
        start = 0;
        end = 0;
      }
      std::fwrite(&start, sizeof(start), 1, f);
      std::fwrite(&end, sizeof(end), 1, f);
    }
  }
  std::fclose(f);

  std::filesystem::rename(tmp, out);
  std::cout << "wrote\t" << out.string() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: tg_build_unigram_ranges <index_dir> <dtype:u8|u16> [out]\n";
    return 2;
  }
  const std::string index_dir = argv[1];
  const std::string dtype = argv[2];
  const std::string out = (argc >= 4) ? argv[3] : "";

  if (dtype == "u8") return Run<gram::u8>(index_dir, out);
  if (dtype == "u16") return Run<gram::u16>(index_dir, out);
  std::cerr << "unsupported dtype\n";
  return 2;
}

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint64_t> ReadU64s(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  in.seekg(0, std::ios::end);
  const std::size_t sz = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  if (sz % sizeof(std::uint64_t) != 0) return {};
  std::vector<std::uint64_t> out(sz / sizeof(std::uint64_t));
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
  return out;
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& p, std::size_t max_bytes) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  in.seekg(0, std::ios::end);
  std::size_t sz = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  if (max_bytes < sz) sz = max_bytes;
  std::vector<std::uint8_t> out(sz);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
  return out;
}

bool WriteBytes(const std::filesystem::path& p, const std::vector<std::uint8_t>& data) {
  std::ofstream out(p, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return true;
}

bool WriteU64s(const std::filesystem::path& p, const std::vector<std::uint64_t>& data) {
  std::ofstream out(p, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(std::uint64_t)));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: tg_slice_index <in_dir> <out_dir> <max_docs> <token_width> [max_tokens]\n";
    return 2;
  }
  const std::filesystem::path in_dir = argv[1];
  const std::filesystem::path out_dir = argv[2];
  const std::uint64_t max_docs = std::strtoull(argv[3], nullptr, 10);
  const std::uint64_t token_width = std::strtoull(argv[4], nullptr, 10);
  const std::uint64_t max_tokens = (argc >= 6) ? std::strtoull(argv[5], nullptr, 10) : 0;

  if (token_width != 1 && token_width != 2 && token_width != 4) {
    std::cerr << "token_width must be 1, 2, or 4\n";
    return 2;
  }

  const auto ds_path = in_dir / "tokenized.0";
  const auto od_path = in_dir / "offset.0";
  const auto mt_path = in_dir / "metadata.0";
  const auto om_path = in_dir / "metaoff.0";

  auto offsets = ReadU64s(od_path);
  if (offsets.empty()) {
    std::cerr << "offset.0 missing or invalid\n";
    return 2;
  }

  std::uint64_t doc_cnt = offsets.size();
  std::uint64_t end_doc = std::min(max_docs, doc_cnt);
  if (max_tokens > 0) {
    const std::uint64_t max_bytes = max_tokens * token_width;
    std::uint64_t lo = 0;
    std::uint64_t hi = end_doc;
    while (lo < hi) {
      std::uint64_t mid = (lo + hi + 1) >> 1;
      std::uint64_t ptr = (mid < doc_cnt) ? offsets[mid] : offsets.back();
      if (ptr <= max_bytes) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    end_doc = lo;
  }

  std::uint64_t end_ptr = (end_doc < doc_cnt) ? offsets[end_doc] : offsets.back();
  if (end_ptr > SIZE_MAX) {
    std::cerr << "file too large for this system\n";
    return 2;
  }
  auto ds_bytes = ReadBytes(ds_path, static_cast<std::size_t>(end_ptr));
  if (ds_bytes.empty()) {
    std::cerr << "tokenized.0 missing or empty\n";
    return 2;
  }

  offsets.resize(end_doc);

  std::vector<std::uint8_t> mt_bytes;
  std::vector<std::uint64_t> metaoffs;
  if (std::filesystem::exists(mt_path) && std::filesystem::exists(om_path)) {
    metaoffs = ReadU64s(om_path);
    if (metaoffs.size() < end_doc) {
      std::cerr << "metaoff.0 too small\n";
      return 2;
    }
    std::uint64_t meta_end = (end_doc < metaoffs.size()) ? metaoffs[end_doc] : metaoffs.back();
    mt_bytes = ReadBytes(mt_path, static_cast<std::size_t>(meta_end));
    metaoffs.resize(end_doc);
  }

  std::filesystem::create_directories(out_dir);
  if (!WriteBytes(out_dir / "tokenized.0", ds_bytes)) return 2;
  if (!WriteU64s(out_dir / "offset.0", offsets)) return 2;
  if (!mt_bytes.empty()) {
    if (!WriteBytes(out_dir / "metadata.0", mt_bytes)) return 2;
    if (!WriteU64s(out_dir / "metaoff.0", metaoffs)) return 2;
  }

  return 0;
}

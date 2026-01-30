#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <sys/resource.h>

namespace {

// Safe copy that skips when source == destination
void SafeCopyFile(const std::filesystem::path& src, const std::filesystem::path& dst) {
  if (std::filesystem::equivalent(src, dst)) return;
  std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
}

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

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  in.seekg(0, std::ios::end);
  const std::size_t sz = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
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

std::vector<std::uint32_t> ReadTokens(const std::vector<std::uint8_t>& bytes, std::size_t token_width) {
  const std::size_t n = bytes.size() / token_width;
  std::vector<std::uint32_t> out(n);
  const std::uint8_t* p = bytes.data();
  for (std::size_t i = 0; i < n; i++) {
    if (token_width == 1) {
      out[i] = p[i];
    } else if (token_width == 2) {
      std::uint16_t v;
      std::memcpy(&v, p + i * 2, 2);
      out[i] = v;
    } else {
      std::uint32_t v;
      std::memcpy(&v, p + i * 4, 4);
      out[i] = v;
    }
  }
  return out;
}

std::uint32_t ByteSwapToken(std::uint32_t t, std::size_t token_width) {
  if (token_width == 1) return t;
  if (token_width == 2) {
    return ((t & 0xFF) << 8) | ((t >> 8) & 0xFF);
  }
#ifdef _MSC_VER
  return _byteswap_ulong(t);
#else
  return __builtin_bswap32(t);
#endif
}

std::vector<std::uint32_t> BuildSuffixArray(const std::vector<std::uint32_t>& s) {
  const std::size_t n = s.size();
  std::vector<std::uint32_t> sa(n), tmp(n), rank(n), nrank(n);
  std::uint32_t maxv = 0;
  for (std::size_t i = 0; i < n; i++) {
    sa[i] = static_cast<std::uint32_t>(i);
    rank[i] = s[i] + 1;
    if (rank[i] > maxv) maxv = rank[i];
  }
  for (std::size_t k = 1; k < n; k <<= 1) {
    const std::uint32_t lim = std::max<std::uint32_t>(maxv + 1, static_cast<std::uint32_t>(n + 1));
    std::vector<std::uint32_t> cnt(lim + 1, 0);
    for (std::size_t i = 0; i < n; i++) {
      const std::uint32_t key = (i + k < n) ? rank[i + k] : 0;
      cnt[key]++;
    }
    for (std::size_t i = 1; i < cnt.size(); i++) cnt[i] += cnt[i - 1];
    for (std::size_t i = n; i-- > 0;) {
      const std::uint32_t idx = sa[i];
      const std::uint32_t key = (idx + k < n) ? rank[idx + k] : 0;
      tmp[--cnt[key]] = idx;
    }
    std::fill(cnt.begin(), cnt.end(), 0);
    for (std::size_t i = 0; i < n; i++) cnt[rank[i]]++;
    for (std::size_t i = 1; i < cnt.size(); i++) cnt[i] += cnt[i - 1];
    for (std::size_t i = n; i-- > 0;) {
      const std::uint32_t idx = tmp[i];
      const std::uint32_t key = rank[idx];
      sa[--cnt[key]] = idx;
    }
    nrank[sa[0]] = 1;
    std::uint32_t r = 1;
    for (std::size_t i = 1; i < n; i++) {
      const std::uint32_t a = sa[i - 1];
      const std::uint32_t b = sa[i];
      const std::uint32_t a1 = rank[a];
      const std::uint32_t b1 = rank[b];
      const std::uint32_t a2 = (a + k < n) ? rank[a + k] : 0;
      const std::uint32_t b2 = (b + k < n) ? rank[b + k] : 0;
      if (a1 != b1 || a2 != b2) r++;
      nrank[b] = r;
    }
    rank.swap(nrank);
    maxv = r;
    if (r == n) break;
  }
  return sa;
}

std::vector<std::uint8_t> EncodeTable(const std::vector<std::uint32_t>& sa, std::size_t token_width, std::uint8_t* ptr_size_out) {
  const std::uint64_t max_ptr = (sa.empty() ? 0 : static_cast<std::uint64_t>(sa.size() - 1) * token_width);
  std::uint8_t ptr_size = 1;
  while ((max_ptr >> (ptr_size * 8)) != 0) ptr_size++;
  const std::size_t out_sz = sa.size() * ptr_size;
  std::vector<std::uint8_t> out(out_sz);
  for (std::size_t i = 0; i < sa.size(); i++) {
    std::uint64_t ptr = static_cast<std::uint64_t>(sa[i]) * token_width;
    for (std::uint8_t b = 0; b < ptr_size; b++) {
      out[i * ptr_size + b] = static_cast<std::uint8_t>((ptr >> (8 * b)) & 0xFF);
    }
  }
  *ptr_size_out = ptr_size;
  return out;
}

void SetRamCap(std::uint64_t bytes) {
  if (bytes == 0) return;
#ifndef _WIN32
  rlimit lim;
  lim.rlim_cur = static_cast<rlim_t>(bytes);
  lim.rlim_max = static_cast<rlim_t>(bytes);
  setrlimit(RLIMIT_AS, &lim);
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: tg_build_index <in_dir> <out_dir> <token_width> <version> <mode:table_only|full> [ram_cap_bytes]\n";
    return 2;
  }
  const std::filesystem::path in_dir = argv[1];
  const std::filesystem::path out_dir = argv[2];
  const std::size_t token_width = static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10));
  const std::uint64_t version = std::strtoull(argv[4], nullptr, 10);
  const std::string mode = argv[5];
  const std::uint64_t ram_cap = (argc >= 7) ? std::strtoull(argv[6], nullptr, 10) : 0;

  if (token_width != 1 && token_width != 2 && token_width != 4) {
    std::cerr << "token_width must be 1, 2, or 4\n";
    return 2;
  }
  if (mode != "table_only" && mode != "full") {
    std::cerr << "mode must be 'table_only' or 'full'\n";
    return 2;
  }

  SetRamCap(ram_cap);

  const auto t0 = std::chrono::steady_clock::now();
  const auto ds_path = in_dir / "tokenized.0";
  auto ds_bytes = ReadBytes(ds_path);
  if (ds_bytes.empty()) {
    std::cerr << "tokenized.0 missing or empty\n";
    return 2;
  }
  if (token_width == 0 || ds_bytes.size() % token_width != 0) {
    std::cerr << "bad token_width\n";
    return 2;
  }
  const auto t1 = std::chrono::steady_clock::now();

  auto tokens = ReadTokens(ds_bytes, token_width);
  const auto t2 = std::chrono::steady_clock::now();

  std::vector<std::uint32_t> sym(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); i++) {
    sym[i] = ByteSwapToken(tokens[i], token_width);
  }
  auto sa = BuildSuffixArray(sym);
  const auto t3 = std::chrono::steady_clock::now();

  std::uint8_t ptr_size = 0;
  auto table = EncodeTable(sa, token_width, &ptr_size);
  const auto t4 = std::chrono::steady_clock::now();

  std::filesystem::create_directories(out_dir);
  if (!WriteBytes(out_dir / "table.0", table)) {
    std::cerr << "failed to write table\n";
    return 2;
  }
  const auto t5 = std::chrono::steady_clock::now();

  if (mode == "full") {
    const auto od_path = in_dir / "offset.0";
    const auto mt_path = in_dir / "metadata.0";
    const auto om_path = in_dir / "metaoff.0";
    const auto ug_path = in_dir / "unigram.0";
    SafeCopyFile(ds_path, out_dir / "tokenized.0");
    if (std::filesystem::exists(od_path)) {
      SafeCopyFile(od_path, out_dir / "offset.0");
    }
    if (std::filesystem::exists(mt_path)) {
      SafeCopyFile(mt_path, out_dir / "metadata.0");
    }
    if (std::filesystem::exists(om_path)) {
      SafeCopyFile(om_path, out_dir / "metaoff.0");
    }
    if (std::filesystem::exists(ug_path)) {
      SafeCopyFile(ug_path, out_dir / "unigram.0");
    }
  }
  const auto t6 = std::chrono::steady_clock::now();

  const auto ns_read = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  const auto ns_tokens = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  const auto ns_sa = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
  const auto ns_encode = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
  const auto ns_write = std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count();
  const auto ns_copy = std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count();
  const auto ns_total = std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t0).count();

  std::cout << "version\t" << version << "\n";
  std::cout << "token_width\t" << token_width << "\n";
  std::cout << "ptr_size\t" << static_cast<int>(ptr_size) << "\n";
  std::cout << "tokens\t" << tokens.size() << "\n";
  std::cout << "read_ns\t" << ns_read << "\n";
  std::cout << "tokenize_ns\t" << ns_tokens << "\n";
  std::cout << "sa_ns\t" << ns_sa << "\n";
  std::cout << "encode_ns\t" << ns_encode << "\n";
  std::cout << "write_ns\t" << ns_write << "\n";
  std::cout << "copy_ns\t" << ns_copy << "\n";
  std::cout << "total_ns\t" << ns_total << "\n";
  return 0;
}

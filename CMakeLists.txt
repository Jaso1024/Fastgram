cmake_minimum_required(VERSION 3.20)

project(gram LANGUAGES CXX)

option(GRAM_BUILD_TESTS "Build gram tests" ON)
option(GRAM_BUILD_TOOLS "Build gram CLI tools" ON)
option(GRAM_BUILD_BENCH "Build gram benchmarks" ON)

add_library(gram
  src/engine.cc
  src/mmap_file.cc
  src/thread_pool.cc
)

target_include_directories(gram PUBLIC include)
target_compile_features(gram PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(gram PRIVATE /W4)
else()
  target_compile_options(gram PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GRAM_BUILD_TOOLS)
  add_executable(tg_query tools/tg_query.cc)
  target_link_libraries(tg_query PRIVATE gram)

  add_executable(tg_build_unigram_ranges tools/tg_build_unigram_ranges.cc)
  target_link_libraries(tg_build_unigram_ranges PRIVATE gram)

  add_executable(tg_rpc tools/tg_rpc.cc)
  target_link_libraries(tg_rpc PRIVATE gram)

  add_executable(tg_slice_index tools/tg_slice_index.cc)
  target_compile_features(tg_slice_index PRIVATE cxx_std_20)

  add_executable(tg_build_index tools/tg_build_index.cc)
  target_compile_features(tg_build_index PRIVATE cxx_std_20)
endif()

if(GRAM_BUILD_TESTS)
  enable_testing()
  add_executable(tg_tests tests/tg_tests.cc)
  target_link_libraries(tg_tests PRIVATE gram)
  add_test(NAME tg_tests COMMAND tg_tests)
endif()

if(GRAM_BUILD_BENCH)
  add_executable(tg_bench_find bench/tg_bench_find.cc)
  target_link_libraries(tg_bench_find PRIVATE gram)
  add_executable(tg_bench_ntd bench/tg_bench_ntd.cc)
  target_link_libraries(tg_bench_ntd PRIVATE gram)
  add_executable(tg_bench_suite bench/tg_bench_suite.cc)
  target_link_libraries(tg_bench_suite PRIVATE gram)
  add_executable(tg_bench_draft bench/tg_bench_draft.cc)
  target_link_libraries(tg_bench_draft PRIVATE gram)
endif()

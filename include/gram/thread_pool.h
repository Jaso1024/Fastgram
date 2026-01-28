#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace gram {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t thread_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void Enqueue(std::function<void()> fn);
  void WaitIdle();

 private:
  void WorkerLoop();

  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable idle_cv_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> threads_;
  bool stop_ = false;
  std::size_t in_flight_ = 0;
};

}  // namespace gram

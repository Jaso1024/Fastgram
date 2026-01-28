#include "gram/thread_pool.h"

namespace gram {

ThreadPool::ThreadPool(std::size_t thread_count) {
  if (thread_count == 0) {
    thread_count = 1;
  }
  threads_.reserve(thread_count);
  for (std::size_t i = 0; i < thread_count; i++) {
    threads_.emplace_back([this]() { WorkerLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void ThreadPool::Enqueue(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    tasks_.push(std::move(fn));
  }
  cv_.notify_one();
}

void ThreadPool::WaitIdle() {
  std::unique_lock<std::mutex> lock(mu_);
  idle_cv_.wait(lock, [this]() { return tasks_.empty() && in_flight_ == 0; });
}

void ThreadPool::WorkerLoop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
      in_flight_++;
    }

    task();

    {
      std::lock_guard<std::mutex> lock(mu_);
      in_flight_--;
      if (tasks_.empty() && in_flight_ == 0) {
        idle_cv_.notify_all();
      }
    }
  }
}

}  // namespace gram

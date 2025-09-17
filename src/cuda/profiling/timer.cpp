#include "timer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

Timer::Timer() : cpu_time_ms_(0.0), gpu_time_ms_(0.0f) {
    // Create CUDA events for GPU timing
    cudaEventCreate(&gpu_start_);
    cudaEventCreate(&gpu_end_);
}

Timer::~Timer() {
    // Destroy CUDA events
    cudaEventDestroy(gpu_start_);
    cudaEventDestroy(gpu_end_);
}

void Timer::start_cpu() {
    cpu_start_ = std::chrono::high_resolution_clock::now();
}

void Timer::stop_cpu() {
    cpu_end_ = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end_ - cpu_start_);
    cpu_time_ms_ = duration.count() / 1000.0;
}

double Timer::get_cpu_time_ms() const {
    return cpu_time_ms_;
}

void Timer::start_gpu() {
    cudaEventRecord(gpu_start_);
}

void Timer::stop_gpu() {
    cudaEventRecord(gpu_end_);
    cudaEventSynchronize(gpu_end_);
    cudaEventElapsedTime(&gpu_time_ms_, gpu_start_, gpu_end_);
}

double Timer::get_gpu_time_ms() const {
    return static_cast<double>(gpu_time_ms_);
}

void Timer::reset() {
    cpu_time_ms_ = 0.0;
    gpu_time_ms_ = 0.0f;
    sessions_.clear();
}

double Timer::get_average_cpu_time(int iterations) const {
    return cpu_time_ms_ / iterations;
}

double Timer::get_average_gpu_time(int iterations) const {
    return static_cast<double>(gpu_time_ms_) / iterations;
}

void Timer::add_timing_session(const std::string& name) {
    TimingSession session;
    session.name = name;
    session.elapsed_time_ms = 0.0;
    session.is_running = false;
    sessions_.push_back(session);
}

void Timer::start_session(const std::string& name) {
    TimingSession* session = find_session(name);
    if (session) {
        session->start_time = std::chrono::high_resolution_clock::now();
        session->is_running = true;
    } else {
        add_timing_session(name);
        start_session(name);
    }
}

void Timer::stop_session(const std::string& name) {
    TimingSession* session = find_session(name);
    if (session && session->is_running) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - session->start_time);
        session->elapsed_time_ms = duration.count() / 1000.0;
        session->is_running = false;
    }
}

double Timer::get_session_time(const std::string& name) const {
    auto it = std::find_if(sessions_.begin(), sessions_.end(),
        [&name](const TimingSession& session) { return session.name == name; });
    
    if (it != sessions_.end()) {
        return it->elapsed_time_ms;
    }
    return 0.0;
}

void Timer::print_all_sessions() const {
    std::cout << "\n=== Timing Sessions ===\n";
    std::cout << std::setw(20) << "Session Name" << std::setw(15) << "Time (ms)" << "\n";
    std::cout << std::string(35, '-') << "\n";
    
    for (const auto& session : sessions_) {
        std::cout << std::setw(20) << session.name 
                  << std::setw(15) << std::fixed << std::setprecision(3) 
                  << session.elapsed_time_ms << "\n";
    }
    std::cout << std::string(35, '-') << "\n";
}

Timer::TimingSession* Timer::find_session(const std::string& name) {
    auto it = std::find_if(sessions_.begin(), sessions_.end(),
        [&name](const TimingSession& session) { return session.name == name; });
    
    return (it != sessions_.end()) ? &(*it) : nullptr;
}






#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <cuda_runtime.h>

class Timer {
public:
    Timer();
    ~Timer();

    // CPU timing
    void start_cpu();
    void stop_cpu();
    double get_cpu_time_ms() const;

    // GPU timing using CUDA events
    void start_gpu();
    void stop_gpu();
    double get_gpu_time_ms() const;

    // Utility functions
    void reset();
    double get_average_cpu_time(int iterations) const;
    double get_average_gpu_time(int iterations) const;

    // Multiple timing sessions
    void add_timing_session(const std::string& name);
    void start_session(const std::string& name);
    void stop_session(const std::string& name);
    double get_session_time(const std::string& name) const;
    void print_all_sessions() const;

private:
    // CPU timing
    std::chrono::high_resolution_clock::time_point cpu_start_;
    std::chrono::high_resolution_clock::time_point cpu_end_;
    double cpu_time_ms_;

    // GPU timing
    cudaEvent_t gpu_start_;
    cudaEvent_t gpu_end_;
    float gpu_time_ms_;

    // Multiple sessions
    struct TimingSession {
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        double elapsed_time_ms;
        bool is_running;
    };
    
    std::vector<TimingSession> sessions_;
    
    // Helper functions
    TimingSession* find_session(const std::string& name);
};

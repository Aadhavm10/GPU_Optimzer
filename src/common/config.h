#pragma once

#include <string>
#include <cstdint>

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

struct Config {
    // Logging configuration
    LogLevel log_level = LogLevel::INFO;
    std::string log_file = "";
    
    // NVML monitoring configuration
    int nvml_sample_rate = 10;  // Hz
    bool enable_power_monitoring = true;
    bool enable_thermal_monitoring = true;
    bool enable_memory_monitoring = true;
    
    // CUDA profiling configuration
    bool enable_cuda_events = true;
    bool enable_occupancy_analysis = true;
    int max_kernel_variants = 10;
    
    // Dashboard configuration
    int dashboard_port = 8080;
    std::string dashboard_host = "localhost";
    bool enable_rest_api = true;
    
    // Data storage configuration
    std::string data_directory = "./data";
    int max_history_hours = 24;
    bool enable_data_export = true;
    
    // GPU configuration
    int target_gpu_id = 0;
    bool multi_gpu_mode = false;
    
    // Performance configuration
    int warmup_iterations = 5;
    int benchmark_iterations = 100;
    bool enable_correctness_checks = true;
    
    // File paths
    std::string config_file = "";
    
    // Load configuration from file
    bool load_from_file(const std::string& filename);
    
    // Save configuration to file
    bool save_to_file(const std::string& filename) const;
    
    // Validate configuration
    bool validate() const;
};

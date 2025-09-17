#include "config.h"
#include <fstream>
#include <iostream>
#include <sstream>

bool Config::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Find key-value separator
        size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Parse configuration values
        if (key == "log_level") {
            if (value == "DEBUG") log_level = LogLevel::DEBUG;
            else if (value == "INFO") log_level = LogLevel::INFO;
            else if (value == "WARN") log_level = LogLevel::WARN;
            else if (value == "ERROR") log_level = LogLevel::ERROR;
        } else if (key == "log_file") {
            log_file = value;
        } else if (key == "nvml_sample_rate") {
            nvml_sample_rate = std::stoi(value);
        } else if (key == "enable_power_monitoring") {
            enable_power_monitoring = (value == "true" || value == "1");
        } else if (key == "enable_thermal_monitoring") {
            enable_thermal_monitoring = (value == "true" || value == "1");
        } else if (key == "enable_memory_monitoring") {
            enable_memory_monitoring = (value == "true" || value == "1");
        } else if (key == "enable_cuda_events") {
            enable_cuda_events = (value == "true" || value == "1");
        } else if (key == "enable_occupancy_analysis") {
            enable_occupancy_analysis = (value == "true" || value == "1");
        } else if (key == "max_kernel_variants") {
            max_kernel_variants = std::stoi(value);
        } else if (key == "dashboard_port") {
            dashboard_port = std::stoi(value);
        } else if (key == "dashboard_host") {
            dashboard_host = value;
        } else if (key == "enable_rest_api") {
            enable_rest_api = (value == "true" || value == "1");
        } else if (key == "data_directory") {
            data_directory = value;
        } else if (key == "max_history_hours") {
            max_history_hours = std::stoi(value);
        } else if (key == "enable_data_export") {
            enable_data_export = (value == "true" || value == "1");
        } else if (key == "target_gpu_id") {
            target_gpu_id = std::stoi(value);
        } else if (key == "multi_gpu_mode") {
            multi_gpu_mode = (value == "true" || value == "1");
        } else if (key == "warmup_iterations") {
            warmup_iterations = std::stoi(value);
        } else if (key == "benchmark_iterations") {
            benchmark_iterations = std::stoi(value);
        } else if (key == "enable_correctness_checks") {
            enable_correctness_checks = (value == "true" || value == "1");
        }
    }
    
    return true;
}

bool Config::save_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# GPU Utilization Optimizer Configuration\n\n";
    
    // Logging configuration
    file << "# Logging\n";
    file << "log_level=";
    switch (log_level) {
        case LogLevel::DEBUG: file << "DEBUG"; break;
        case LogLevel::INFO: file << "INFO"; break;
        case LogLevel::WARN: file << "WARN"; break;
        case LogLevel::ERROR: file << "ERROR"; break;
    }
    file << "\n";
    file << "log_file=" << log_file << "\n\n";
    
    // NVML monitoring configuration
    file << "# NVML Monitoring\n";
    file << "nvml_sample_rate=" << nvml_sample_rate << "\n";
    file << "enable_power_monitoring=" << (enable_power_monitoring ? "true" : "false") << "\n";
    file << "enable_thermal_monitoring=" << (enable_thermal_monitoring ? "true" : "false") << "\n";
    file << "enable_memory_monitoring=" << (enable_memory_monitoring ? "true" : "false") << "\n\n";
    
    // CUDA profiling configuration
    file << "# CUDA Profiling\n";
    file << "enable_cuda_events=" << (enable_cuda_events ? "true" : "false") << "\n";
    file << "enable_occupancy_analysis=" << (enable_occupancy_analysis ? "true" : "false") << "\n";
    file << "max_kernel_variants=" << max_kernel_variants << "\n\n";
    
    // Dashboard configuration
    file << "# Dashboard\n";
    file << "dashboard_port=" << dashboard_port << "\n";
    file << "dashboard_host=" << dashboard_host << "\n";
    file << "enable_rest_api=" << (enable_rest_api ? "true" : "false") << "\n\n";
    
    // Data storage configuration
    file << "# Data Storage\n";
    file << "data_directory=" << data_directory << "\n";
    file << "max_history_hours=" << max_history_hours << "\n";
    file << "enable_data_export=" << (enable_data_export ? "true" : "false") << "\n\n";
    
    // GPU configuration
    file << "# GPU Configuration\n";
    file << "target_gpu_id=" << target_gpu_id << "\n";
    file << "multi_gpu_mode=" << (multi_gpu_mode ? "true" : "false") << "\n\n";
    
    // Performance configuration
    file << "# Performance\n";
    file << "warmup_iterations=" << warmup_iterations << "\n";
    file << "benchmark_iterations=" << benchmark_iterations << "\n";
    file << "enable_correctness_checks=" << (enable_correctness_checks ? "true" : "false") << "\n";
    
    return true;
}

bool Config::validate() const {
    // Validate configuration values
    if (nvml_sample_rate <= 0 || nvml_sample_rate > 1000) {
        std::cerr << "Invalid nvml_sample_rate: " << nvml_sample_rate << " (must be 1-1000 Hz)" << std::endl;
        return false;
    }
    
    if (dashboard_port <= 0 || dashboard_port > 65535) {
        std::cerr << "Invalid dashboard_port: " << dashboard_port << " (must be 1-65535)" << std::endl;
        return false;
    }
    
    if (max_kernel_variants <= 0 || max_kernel_variants > 50) {
        std::cerr << "Invalid max_kernel_variants: " << max_kernel_variants << " (must be 1-50)" << std::endl;
        return false;
    }
    
    if (target_gpu_id < 0) {
        std::cerr << "Invalid target_gpu_id: " << target_gpu_id << " (must be >= 0)" << std::endl;
        return false;
    }
    
    if (warmup_iterations < 0 || warmup_iterations > 100) {
        std::cerr << "Invalid warmup_iterations: " << warmup_iterations << " (must be 0-100)" << std::endl;
        return false;
    }
    
    if (benchmark_iterations <= 0 || benchmark_iterations > 10000) {
        std::cerr << "Invalid benchmark_iterations: " << benchmark_iterations << " (must be 1-10000)" << std::endl;
        return false;
    }
    
    if (max_history_hours <= 0 || max_history_hours > 168) { // Max 1 week
        std::cerr << "Invalid max_history_hours: " << max_history_hours << " (must be 1-168)" << std::endl;
        return false;
    }
    
    return true;
}

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

struct GPUInfo {
    int device_id;
    std::string name;
    std::string uuid;
    int major_version;
    int minor_version;
    size_t total_memory_mb;
    int max_clock_mhz;
    int max_mem_clock_mhz;
    int multiprocessor_count;
    int max_threads_per_multiprocessor;
    int max_threads_per_block;
    int warp_size;
    bool ecc_enabled;
    std::string driver_version;
    std::string cuda_version;
};

struct PowerMetrics {
    double power_draw_watts;
    double power_limit_watts;
    double power_usage_percentage;
    bool power_management_enabled;
};

struct ThermalMetrics {
    int gpu_temperature_c;
    int memory_temperature_c;
    int max_operating_temperature_c;
    int shutdown_temperature_c;
    int slowdown_temperature_c;
    bool thermal_throttling_active;
};

struct MemoryMetrics {
    size_t total_memory_mb;
    size_t used_memory_mb;
    size_t free_memory_mb;
    double usage_percentage;
    int memory_clock_mhz;
    int memory_bus_width;
    double memory_bandwidth_gb_s;
};

struct UtilizationMetrics {
    int gpu_utilization_percentage;
    int memory_utilization_percentage;
    int encoder_utilization_percentage;
    int decoder_utilization_percentage;
};

struct ClockMetrics {
    int graphics_clock_mhz;
    int memory_clock_mhz;
    int sm_clock_mhz;
    int video_clock_mhz;
    int max_graphics_clock_mhz;
    int max_memory_clock_mhz;
};

struct ProcessMetrics {
    uint32_t pid;
    std::string process_name;
    size_t used_memory_mb;
    int gpu_utilization_percentage;
    int memory_utilization_percentage;
};

struct GPUMetrics {
    // Metadata
    std::chrono::system_clock::time_point timestamp;
    int device_id;
    std::string device_name;
    
    // Core metrics
    PowerMetrics power;
    ThermalMetrics thermal;
    MemoryMetrics memory;
    UtilizationMetrics utilization;
    ClockMetrics clocks;
    
    // Process information
    std::vector<ProcessMetrics> processes;
    
    // Derived metrics
    double compute_efficiency;      // Utilization vs power ratio
    double memory_efficiency;      // Memory bandwidth utilization
    double thermal_efficiency;     // Performance vs temperature
    
    // Performance indicators
    bool is_throttled;
    bool is_overheating;
    bool is_power_limited;
    bool is_memory_bottlenecked;
    
    // Constructor
    GPUMetrics() : timestamp(std::chrono::system_clock::now()), device_id(-1) {}
    
    // Utility functions
    double get_age_seconds() const;
    bool is_valid() const;
    std::string to_json() const;
    std::string to_csv_line() const;
    static std::string get_csv_header();
};






#pragma once

#include "gpu_metrics.h"
#include "common/config.h"
#include <vector>
#include <memory>

class DataProcessor {
public:
    explicit DataProcessor(const Config& config);
    ~DataProcessor();

    // Process metrics and add derived values
    void process_metrics(GPUMetrics& metrics);
    
    // Analysis functions
    bool detect_memory_bottleneck(const GPUMetrics& metrics);
    bool detect_thermal_throttling(const GPUMetrics& metrics);
    bool detect_power_limiting(const GPUMetrics& metrics);
    bool detect_compute_bottleneck(const GPUMetrics& metrics);
    
    // Efficiency calculations
    double calculate_power_efficiency(const GPUMetrics& metrics);
    double calculate_thermal_efficiency(const GPUMetrics& metrics);
    double calculate_memory_efficiency(const GPUMetrics& metrics);
    
    // Trend analysis (requires historical data)
    struct TrendAnalysis {
        double temperature_trend;    // °C per minute
        double power_trend;         // W per minute
        double utilization_trend;   // % per minute
        bool stable;
    };
    
    TrendAnalysis analyze_trends(const std::vector<GPUMetrics>& historical_data);

private:
    Config config_;
    
    // Historical data for trend analysis
    std::vector<GPUMetrics> recent_metrics_;
    static constexpr size_t MAX_RECENT_METRICS = 300; // 5 minutes at 1Hz
    
    // Thresholds
    double memory_bottleneck_threshold_;
    double thermal_throttle_threshold_;
    double power_limit_threshold_;
    
    // Helper functions
    void update_recent_metrics(const GPUMetrics& metrics);
    double calculate_moving_average(const std::vector<double>& values, size_t window_size);
    double calculate_slope(const std::vector<double>& values, const std::vector<double>& timestamps);
};






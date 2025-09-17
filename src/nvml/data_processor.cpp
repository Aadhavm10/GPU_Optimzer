#include "data_processor.h"
#include "common/logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>

DataProcessor::DataProcessor(const Config& config) 
    : config_(config),
      memory_bottleneck_threshold_(0.9),
      thermal_throttle_threshold_(85.0),
      power_limit_threshold_(0.95) {
    
    recent_metrics_.reserve(MAX_RECENT_METRICS);
}

DataProcessor::~DataProcessor() = default;

void DataProcessor::process_metrics(GPUMetrics& metrics) {
    // Calculate efficiency metrics
    metrics.compute_efficiency = calculate_power_efficiency(metrics);
    metrics.thermal_efficiency = calculate_thermal_efficiency(metrics);
    metrics.memory_efficiency = calculate_memory_efficiency(metrics);
    
    // Detect bottlenecks and issues
    metrics.is_memory_bottlenecked = detect_memory_bottleneck(metrics);
    metrics.is_throttled = detect_thermal_throttling(metrics);
    metrics.is_power_limited = detect_power_limiting(metrics);
    metrics.is_overheating = metrics.thermal.gpu_temperature_c > thermal_throttle_threshold_;
    
    // Update recent metrics for trend analysis
    update_recent_metrics(metrics);
}

bool DataProcessor::detect_memory_bottleneck(const GPUMetrics& metrics) {
    // Memory bottleneck indicators:
    // 1. High memory utilization but low GPU utilization
    // 2. Memory bandwidth saturation
    
    double memory_util = metrics.utilization.memory_utilization_percentage / 100.0;
    double gpu_util = metrics.utilization.gpu_utilization_percentage / 100.0;
    
    // High memory utilization with low GPU utilization suggests memory bottleneck
    bool high_mem_low_gpu = (memory_util > memory_bottleneck_threshold_) && (gpu_util < 0.5);
    
    // Memory usage close to capacity
    bool memory_pressure = metrics.memory.usage_percentage > (memory_bottleneck_threshold_ * 100);
    
    return high_mem_low_gpu || memory_pressure;
}

bool DataProcessor::detect_thermal_throttling(const GPUMetrics& metrics) {
    // Check temperature thresholds
    bool high_temp = metrics.thermal.gpu_temperature_c > thermal_throttle_threshold_;
    
    // Check if explicitly reported by driver
    bool throttling_active = metrics.thermal.thermal_throttling_active;
    
    // Check for performance drops due to temperature
    bool performance_drop = false;
    if (recent_metrics_.size() > 10) {
        // Compare current clocks to recent average
        double recent_avg_clock = 0.0;
        size_t count = std::min(static_cast<size_t>(10), recent_metrics_.size());
        
        for (size_t i = recent_metrics_.size() - count; i < recent_metrics_.size(); ++i) {
            recent_avg_clock += recent_metrics_[i].clocks.graphics_clock_mhz;
        }
        recent_avg_clock /= count;
        
        // If current clock is significantly lower than recent average
        performance_drop = metrics.clocks.graphics_clock_mhz < (recent_avg_clock * 0.9);
    }
    
    return high_temp || throttling_active || performance_drop;
}

bool DataProcessor::detect_power_limiting(const GPUMetrics& metrics) {
    // Power limiting indicators:
    // 1. Power usage near limit
    // 2. Clocks dropping while temperature is reasonable
    
    bool near_power_limit = metrics.power.power_usage_percentage > (power_limit_threshold_ * 100);
    
    // Check for clock reduction without thermal issues
    bool clock_reduction = false;
    if (recent_metrics_.size() > 5) {
        double recent_avg_clock = 0.0;
        size_t count = std::min(static_cast<size_t>(5), recent_metrics_.size());
        
        for (size_t i = recent_metrics_.size() - count; i < recent_metrics_.size(); ++i) {
            recent_avg_clock += recent_metrics_[i].clocks.graphics_clock_mhz;
        }
        recent_avg_clock /= count;
        
        // Clock reduction without high temperature suggests power limiting
        bool clocks_dropped = metrics.clocks.graphics_clock_mhz < (recent_avg_clock * 0.95);
        bool temp_reasonable = metrics.thermal.gpu_temperature_c < (thermal_throttle_threshold_ - 5);
        
        clock_reduction = clocks_dropped && temp_reasonable;
    }
    
    return near_power_limit || clock_reduction;
}

bool DataProcessor::detect_compute_bottleneck(const GPUMetrics& metrics) {
    // Compute bottleneck indicators:
    // 1. High GPU utilization with low memory utilization
    // 2. High temperature and power usage
    
    double gpu_util = metrics.utilization.gpu_utilization_percentage / 100.0;
    double memory_util = metrics.utilization.memory_utilization_percentage / 100.0;
    
    // High GPU utilization suggests compute-bound workload
    bool high_gpu_util = gpu_util > 0.9;
    
    // Low memory utilization with high GPU utilization
    bool low_memory_util = memory_util < 0.5;
    
    // High power/temperature suggests intensive compute
    bool high_power = metrics.power.power_usage_percentage > 80.0;
    bool elevated_temp = metrics.thermal.gpu_temperature_c > 70;
    
    return high_gpu_util && (low_memory_util || (high_power && elevated_temp));
}

double DataProcessor::calculate_power_efficiency(const GPUMetrics& metrics) {
    // Power efficiency: Performance per watt
    // Higher values indicate better efficiency
    
    if (metrics.power.power_draw_watts <= 0) {
        return 0.0;
    }
    
    // Use GPU utilization as a proxy for performance
    double performance = metrics.utilization.gpu_utilization_percentage;
    
    return performance / metrics.power.power_draw_watts;
}

double DataProcessor::calculate_thermal_efficiency(const GPUMetrics& metrics) {
    // Thermal efficiency: Performance per degree Celsius
    // Higher values indicate better thermal management
    
    if (metrics.thermal.gpu_temperature_c <= 0) {
        return 0.0;
    }
    
    double performance = metrics.utilization.gpu_utilization_percentage;
    
    return performance / static_cast<double>(metrics.thermal.gpu_temperature_c);
}

double DataProcessor::calculate_memory_efficiency(const GPUMetrics& metrics) {
    // Memory efficiency: Memory bandwidth utilization
    // Higher values indicate better memory usage
    
    if (metrics.memory.memory_bandwidth_gb_s <= 0) {
        return metrics.utilization.memory_utilization_percentage / 100.0;
    }
    
    // This would require actual memory bandwidth measurements
    // For now, use memory utilization as a proxy
    return metrics.utilization.memory_utilization_percentage / 100.0;
}

DataProcessor::TrendAnalysis DataProcessor::analyze_trends(const std::vector<GPUMetrics>& historical_data) {
    TrendAnalysis analysis{};
    
    if (historical_data.size() < 10) {
        // Not enough data for trend analysis
        analysis.stable = false;
        return analysis;
    }
    
    // Extract time series data
    std::vector<double> timestamps;
    std::vector<double> temperatures;
    std::vector<double> power_values;
    std::vector<double> utilization_values;
    
    for (const auto& metrics : historical_data) {
        auto epoch = metrics.timestamp.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch).count();
        
        timestamps.push_back(static_cast<double>(seconds));
        temperatures.push_back(static_cast<double>(metrics.thermal.gpu_temperature_c));
        power_values.push_back(metrics.power.power_draw_watts);
        utilization_values.push_back(static_cast<double>(metrics.utilization.gpu_utilization_percentage));
    }
    
    // Calculate trends (slopes)
    analysis.temperature_trend = calculate_slope(temperatures, timestamps) * 60.0; // per minute
    analysis.power_trend = calculate_slope(power_values, timestamps) * 60.0; // per minute
    analysis.utilization_trend = calculate_slope(utilization_values, timestamps) * 60.0; // per minute
    
    // Determine stability
    double temp_stability = std::abs(analysis.temperature_trend);
    double power_stability = std::abs(analysis.power_trend);
    double util_stability = std::abs(analysis.utilization_trend);
    
    // Consider stable if trends are small
    analysis.stable = (temp_stability < 1.0) && (power_stability < 5.0) && (util_stability < 5.0);
    
    return analysis;
}

void DataProcessor::update_recent_metrics(const GPUMetrics& metrics) {
    recent_metrics_.push_back(metrics);
    
    // Maintain size limit
    if (recent_metrics_.size() > MAX_RECENT_METRICS) {
        recent_metrics_.erase(recent_metrics_.begin());
    }
}

double DataProcessor::calculate_moving_average(const std::vector<double>& values, size_t window_size) {
    if (values.empty() || window_size == 0) {
        return 0.0;
    }
    
    size_t start_idx = (values.size() > window_size) ? (values.size() - window_size) : 0;
    size_t count = values.size() - start_idx;
    
    double sum = 0.0;
    for (size_t i = start_idx; i < values.size(); ++i) {
        sum += values[i];
    }
    
    return sum / static_cast<double>(count);
}

double DataProcessor::calculate_slope(const std::vector<double>& values, const std::vector<double>& timestamps) {
    if (values.size() != timestamps.size() || values.size() < 2) {
        return 0.0;
    }
    
    size_t n = values.size();
    
    // Calculate means
    double mean_x = std::accumulate(timestamps.begin(), timestamps.end(), 0.0) / n;
    double mean_y = std::accumulate(values.begin(), values.end(), 0.0) / n;
    
    // Calculate slope using least squares
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double dx = timestamps[i] - mean_x;
        double dy = values[i] - mean_y;
        
        numerator += dx * dy;
        denominator += dx * dx;
    }
    
    if (std::abs(denominator) < 1e-10) {
        return 0.0;
    }
    
    return numerator / denominator;
}






#pragma once

#include <nvml.h>
#include "gpu_metrics.h"
#include "common/config.h"

class MetricsCollector {
public:
    explicit MetricsCollector(const Config& config);
    ~MetricsCollector();

    // Core collection methods
    GPUMetrics collect_metrics(int gpu_id);
    GPUMetrics collect_metrics(nvmlDevice_t device, int gpu_id);
    
    // Individual metric collection
    PowerMetrics collect_power_metrics(nvmlDevice_t device);
    ThermalMetrics collect_thermal_metrics(nvmlDevice_t device);
    MemoryMetrics collect_memory_metrics(nvmlDevice_t device);
    UtilizationMetrics collect_utilization_metrics(nvmlDevice_t device);
    ClockMetrics collect_clock_metrics(nvmlDevice_t device);
    std::vector<ProcessMetrics> collect_process_metrics(nvmlDevice_t device);

    // Utility methods
    bool is_device_valid(int gpu_id) const;
    nvmlDevice_t get_device_handle(int gpu_id) const;

private:
    Config config_;
    std::vector<nvmlDevice_t> device_handles_;
    
    // Helper methods
    void initialize_device_handles();
    double calculate_memory_bandwidth(const MemoryMetrics& memory);
    void update_derived_metrics(GPUMetrics& metrics);
    bool check_throttle_reasons(nvmlDevice_t device);
};

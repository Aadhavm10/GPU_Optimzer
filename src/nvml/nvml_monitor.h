#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <nvml.h>

#include "common/config.h"
#include "gpu_metrics.h"
#include "metrics_collector.h"
#include "data_processor.h"

class NVMLMonitor {
public:
    explicit NVMLMonitor(const Config& config);
    ~NVMLMonitor();

    // Monitoring control
    bool start_monitoring();
    void stop_monitoring();
    bool is_monitoring() const;

    // Data access
    GPUMetrics get_current_metrics() const;
    std::vector<GPUMetrics> get_historical_metrics(int hours) const;
    GPUMetrics get_average_metrics(int minutes) const;

    // GPU information
    int get_gpu_count() const;
    std::vector<GPUInfo> get_gpu_info() const;
    bool is_gpu_available(int gpu_id) const;

    // Callback system for real-time data
    using MetricsCallback = std::function<void(const GPUMetrics&)>;
    void register_callback(MetricsCallback callback);
    void unregister_all_callbacks();

    // Alert system
    void set_temperature_threshold(int celsius);
    void set_power_threshold(int watts);
    void set_memory_threshold(double percentage);
    
    // Export functionality
    bool export_metrics_to_csv(const std::string& filename, int hours) const;
    bool export_metrics_to_json(const std::string& filename, int hours) const;

private:
    Config config_;
    std::atomic<bool> monitoring_;
    std::atomic<bool> should_stop_;
    
    // Threading
    std::unique_ptr<std::thread> monitor_thread_;
    std::mutex metrics_mutex_;
    std::condition_variable metrics_cv_;
    
    // Components
    std::unique_ptr<MetricsCollector> collector_;
    std::unique_ptr<DataProcessor> processor_;
    
    // Data storage
    std::queue<GPUMetrics> metrics_queue_;
    static constexpr size_t MAX_QUEUE_SIZE = 86400; // 24 hours at 1Hz
    
    // Callbacks
    std::vector<MetricsCallback> callbacks_;
    std::mutex callback_mutex_;
    
    // Alert thresholds
    int temperature_threshold_;
    int power_threshold_;
    double memory_threshold_;
    
    // GPU information
    std::vector<GPUInfo> gpu_info_;
    int gpu_count_;
    
    // Private methods
    void monitoring_loop();
    void process_metrics(const GPUMetrics& metrics);
    void check_alerts(const GPUMetrics& metrics);
    void initialize_nvml();
    void cleanup_nvml();
    void collect_gpu_info();
    void notify_callbacks(const GPUMetrics& metrics);
    void maintain_queue_size();
};

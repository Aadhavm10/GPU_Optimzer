#include "nvml_monitor.h"
#include "common/logger.h"
#include <stdexcept>
#include <chrono>
#include <algorithm>

NVMLMonitor::NVMLMonitor(const Config& config) 
    : config_(config), monitoring_(false), should_stop_(false),
      temperature_threshold_(85), power_threshold_(300), memory_threshold_(0.9),
      gpu_count_(0) {
    
    initialize_nvml();
    collect_gpu_info();
    
    collector_ = std::make_unique<MetricsCollector>(config_);
    processor_ = std::make_unique<DataProcessor>(config_);
}

NVMLMonitor::~NVMLMonitor() {
    stop_monitoring();
    cleanup_nvml();
}

bool NVMLMonitor::start_monitoring() {
    if (monitoring_.load()) {
        Logger::warn("Monitoring already active");
        return true;
    }
    
    should_stop_.store(false);
    monitoring_.store(true);
    
    monitor_thread_ = std::make_unique<std::thread>(&NVMLMonitor::monitoring_loop, this);
    
    Logger::info("NVML monitoring started");
    return true;
}

void NVMLMonitor::stop_monitoring() {
    if (!monitoring_.load()) {
        return;
    }
    
    should_stop_.store(true);
    monitoring_.store(false);
    
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }
    
    Logger::info("NVML monitoring stopped");
}

bool NVMLMonitor::is_monitoring() const {
    return monitoring_.load();
}

GPUMetrics NVMLMonitor::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (metrics_queue_.empty()) {
        return GPUMetrics{};
    }
    
    return metrics_queue_.back();
}

std::vector<GPUMetrics> NVMLMonitor::get_historical_metrics(int hours) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(hours);
    std::vector<GPUMetrics> result;
    
    std::queue<GPUMetrics> temp_queue = metrics_queue_;
    while (!temp_queue.empty()) {
        const auto& metrics = temp_queue.front();
        if (metrics.timestamp >= cutoff_time) {
            result.push_back(metrics);
        }
        temp_queue.pop();
    }
    
    return result;
}

GPUMetrics NVMLMonitor::get_average_metrics(int minutes) const {
    auto historical = get_historical_metrics(1); // Get last hour
    auto cutoff_time = std::chrono::system_clock::now() - std::chrono::minutes(minutes);
    
    // Filter to desired time range
    std::vector<GPUMetrics> filtered;
    std::copy_if(historical.begin(), historical.end(), std::back_inserter(filtered),
        [cutoff_time](const GPUMetrics& m) { return m.timestamp >= cutoff_time; });
    
    if (filtered.empty()) {
        return GPUMetrics{};
    }
    
    // Calculate averages
    GPUMetrics avg{};
    avg.timestamp = std::chrono::system_clock::now();
    avg.device_id = filtered[0].device_id;
    avg.device_name = filtered[0].device_name;
    
    double count = static_cast<double>(filtered.size());
    
    for (const auto& metrics : filtered) {
        avg.power.power_draw_watts += metrics.power.power_draw_watts;
        avg.thermal.gpu_temperature_c += metrics.thermal.gpu_temperature_c;
        avg.memory.usage_percentage += metrics.memory.usage_percentage;
        avg.utilization.gpu_utilization_percentage += metrics.utilization.gpu_utilization_percentage;
        avg.utilization.memory_utilization_percentage += metrics.utilization.memory_utilization_percentage;
        avg.clocks.graphics_clock_mhz += metrics.clocks.graphics_clock_mhz;
        avg.clocks.memory_clock_mhz += metrics.clocks.memory_clock_mhz;
    }
    
    // Divide by count to get averages
    avg.power.power_draw_watts /= count;
    avg.thermal.gpu_temperature_c = static_cast<int>(avg.thermal.gpu_temperature_c / count);
    avg.memory.usage_percentage /= count;
    avg.utilization.gpu_utilization_percentage = static_cast<int>(avg.utilization.gpu_utilization_percentage / count);
    avg.utilization.memory_utilization_percentage = static_cast<int>(avg.utilization.memory_utilization_percentage / count);
    avg.clocks.graphics_clock_mhz = static_cast<int>(avg.clocks.graphics_clock_mhz / count);
    avg.clocks.memory_clock_mhz = static_cast<int>(avg.clocks.memory_clock_mhz / count);
    
    return avg;
}

int NVMLMonitor::get_gpu_count() const {
    return gpu_count_;
}

std::vector<GPUInfo> NVMLMonitor::get_gpu_info() const {
    return gpu_info_;
}

bool NVMLMonitor::is_gpu_available(int gpu_id) const {
    return gpu_id >= 0 && gpu_id < gpu_count_;
}

void NVMLMonitor::register_callback(MetricsCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callbacks_.push_back(callback);
}

void NVMLMonitor::unregister_all_callbacks() {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callbacks_.clear();
}

void NVMLMonitor::set_temperature_threshold(int celsius) {
    temperature_threshold_ = celsius;
    Logger::info("Temperature threshold set to {}°C", celsius);
}

void NVMLMonitor::set_power_threshold(int watts) {
    power_threshold_ = watts;
    Logger::info("Power threshold set to {}W", watts);
}

void NVMLMonitor::set_memory_threshold(double percentage) {
    memory_threshold_ = percentage;
    Logger::info("Memory threshold set to {:.1f}%", percentage * 100);
}

bool NVMLMonitor::export_metrics_to_csv(const std::string& filename, int hours) const {
    auto metrics = get_historical_metrics(hours);
    if (metrics.empty()) {
        Logger::warn("No metrics available for export");
        return false;
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        Logger::error("Failed to open file for export: {}", filename);
        return false;
    }
    
    // Write CSV header
    file << GPUMetrics::get_csv_header() << "\n";
    
    // Write data
    for (const auto& m : metrics) {
        file << m.to_csv_line() << "\n";
    }
    
    file.close();
    Logger::info("Exported {} metrics to {}", metrics.size(), filename);
    return true;
}

bool NVMLMonitor::export_metrics_to_json(const std::string& filename, int hours) const {
    auto metrics = get_historical_metrics(hours);
    if (metrics.empty()) {
        Logger::warn("No metrics available for export");
        return false;
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        Logger::error("Failed to open file for export: {}", filename);
        return false;
    }
    
    file << "[\n";
    for (size_t i = 0; i < metrics.size(); ++i) {
        file << metrics[i].to_json();
        if (i < metrics.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    file << "]\n";
    
    file.close();
    Logger::info("Exported {} metrics to {}", metrics.size(), filename);
    return true;
}

void NVMLMonitor::monitoring_loop() {
    Logger::info("Monitoring loop started");
    
    auto next_sample_time = std::chrono::steady_clock::now();
    const auto sample_interval = std::chrono::milliseconds(1000 / config_.nvml_sample_rate);
    
    while (!should_stop_.load()) {
        try {
            // Collect metrics for target GPU
            GPUMetrics metrics = collector_->collect_metrics(config_.target_gpu_id);
            
            // Process metrics (calculate derived values, detect bottlenecks)
            processor_->process_metrics(metrics);
            
            // Store metrics
            process_metrics(metrics);
            
            // Check for alerts
            check_alerts(metrics);
            
            // Notify callbacks
            notify_callbacks(metrics);
            
        } catch (const std::exception& e) {
            Logger::error("Error in monitoring loop: {}", e.what());
        }
        
        // Wait for next sample time
        next_sample_time += sample_interval;
        std::this_thread::sleep_until(next_sample_time);
    }
    
    Logger::info("Monitoring loop ended");
}

void NVMLMonitor::process_metrics(const GPUMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Add to queue
    metrics_queue_.push(metrics);
    
    // Maintain queue size
    maintain_queue_size();
}

void NVMLMonitor::check_alerts(const GPUMetrics& metrics) {
    // Temperature alert
    if (metrics.thermal.gpu_temperature_c > temperature_threshold_) {
        Logger::warn("High temperature alert: {}°C (threshold: {}°C)", 
                    metrics.thermal.gpu_temperature_c, temperature_threshold_);
    }
    
    // Power alert
    if (metrics.power.power_draw_watts > power_threshold_) {
        Logger::warn("High power draw alert: {:.1f}W (threshold: {}W)", 
                    metrics.power.power_draw_watts, power_threshold_);
    }
    
    // Memory alert
    if (metrics.memory.usage_percentage > memory_threshold_ * 100) {
        Logger::warn("High memory usage alert: {:.1f}% (threshold: {:.1f}%)", 
                    metrics.memory.usage_percentage, memory_threshold_ * 100);
    }
    
    // Throttling alerts
    if (metrics.is_throttled) {
        Logger::warn("GPU throttling detected");
    }
    
    if (metrics.is_overheating) {
        Logger::warn("GPU overheating detected");
    }
    
    if (metrics.is_power_limited) {
        Logger::warn("GPU power limiting detected");
    }
}

void NVMLMonitor::initialize_nvml() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        throw std::runtime_error("Failed to initialize NVML: " + std::string(nvmlErrorString(result)));
    }
    
    Logger::info("NVML initialized successfully");
}

void NVMLMonitor::cleanup_nvml() {
    nvmlShutdown();
    Logger::debug("NVML shutdown completed");
}

void NVMLMonitor::collect_gpu_info() {
    nvmlReturn_t result = nvmlDeviceGetCount(&gpu_count_);
    if (result != NVML_SUCCESS) {
        throw std::runtime_error("Failed to get GPU count: " + std::string(nvmlErrorString(result)));
    }
    
    Logger::info("Found {} GPU(s)", gpu_count_);
    
    gpu_info_.clear();
    gpu_info_.reserve(gpu_count_);
    
    for (unsigned int i = 0; i < gpu_count_; ++i) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            Logger::warn("Failed to get handle for GPU {}: {}", i, nvmlErrorString(result));
            continue;
        }
        
        GPUInfo info;
        info.device_id = i;
        
        // Get device name
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            info.name = std::string(name);
        }
        
        // Get UUID
        char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
        result = nvmlDeviceGetUUID(device, uuid, NVML_DEVICE_UUID_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            info.uuid = std::string(uuid);
        }
        
        // Get memory info
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (result == NVML_SUCCESS) {
            info.total_memory_mb = memory.total / (1024 * 1024);
        }
        
        // Get compute capability
        result = nvmlDeviceGetCudaComputeCapability(device, &info.major_version, &info.minor_version);
        if (result != NVML_SUCCESS) {
            Logger::warn("Failed to get compute capability for GPU {}", i);
        }
        
        // Get max clocks
        result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &info.max_clock_mhz);
        if (result != NVML_SUCCESS) {
            Logger::debug("Failed to get max graphics clock for GPU {}", i);
        }
        
        result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &info.max_mem_clock_mhz);
        if (result != NVML_SUCCESS) {
            Logger::debug("Failed to get max memory clock for GPU {}", i);
        }
        
        gpu_info_.push_back(info);
        
        Logger::info("GPU {}: {} (Compute {}.{}, {} MB)", 
                    i, info.name, info.major_version, info.minor_version, info.total_memory_mb);
    }
}

void NVMLMonitor::notify_callbacks(const GPUMetrics& metrics) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    
    for (auto& callback : callbacks_) {
        try {
            callback(metrics);
        } catch (const std::exception& e) {
            Logger::error("Error in metrics callback: {}", e.what());
        }
    }
}

void NVMLMonitor::maintain_queue_size() {
    // Remove old entries if queue is too large
    while (metrics_queue_.size() > MAX_QUEUE_SIZE) {
        metrics_queue_.pop();
    }
}






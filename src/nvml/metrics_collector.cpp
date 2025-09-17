#include "metrics_collector.h"
#include "common/logger.h"
#include <stdexcept>
#include <cstring>

MetricsCollector::MetricsCollector(const Config& config) : config_(config) {
    initialize_device_handles();
}

MetricsCollector::~MetricsCollector() {
    // NVML device handles don't need explicit cleanup
}

GPUMetrics MetricsCollector::collect_metrics(int gpu_id) {
    if (!is_device_valid(gpu_id)) {
        throw std::invalid_argument("Invalid GPU ID: " + std::to_string(gpu_id));
    }
    
    nvmlDevice_t device = get_device_handle(gpu_id);
    return collect_metrics(device, gpu_id);
}

GPUMetrics MetricsCollector::collect_metrics(nvmlDevice_t device, int gpu_id) {
    GPUMetrics metrics;
    metrics.timestamp = std::chrono::system_clock::now();
    metrics.device_id = gpu_id;
    
    // Get device name
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlReturn_t result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        metrics.device_name = std::string(name);
    }
    
    try {
        // Collect individual metric categories
        if (config_.enable_power_monitoring) {
            metrics.power = collect_power_metrics(device);
        }
        
        if (config_.enable_thermal_monitoring) {
            metrics.thermal = collect_thermal_metrics(device);
        }
        
        if (config_.enable_memory_monitoring) {
            metrics.memory = collect_memory_metrics(device);
        }
        
        metrics.utilization = collect_utilization_metrics(device);
        metrics.clocks = collect_clock_metrics(device);
        metrics.processes = collect_process_metrics(device);
        
        // Calculate derived metrics
        update_derived_metrics(metrics);
        
    } catch (const std::exception& e) {
        Logger::error("Error collecting metrics for GPU {}: {}", gpu_id, e.what());
        // Return partial metrics rather than throwing
    }
    
    return metrics;
}

PowerMetrics MetricsCollector::collect_power_metrics(nvmlDevice_t device) {
    PowerMetrics power{};
    nvmlReturn_t result;
    
    // Power draw
    unsigned int power_draw;
    result = nvmlDeviceGetPowerUsage(device, &power_draw);
    if (result == NVML_SUCCESS) {
        power.power_draw_watts = static_cast<double>(power_draw) / 1000.0; // mW to W
    }
    
    // Power limit
    unsigned int power_limit;
    result = nvmlDeviceGetPowerManagementLimitConstraints(device, nullptr, &power_limit);
    if (result == NVML_SUCCESS) {
        power.power_limit_watts = static_cast<double>(power_limit) / 1000.0; // mW to W
        
        if (power.power_limit_watts > 0) {
            power.power_usage_percentage = (power.power_draw_watts / power.power_limit_watts) * 100.0;
        }
    }
    
    // Power management mode
    nvmlEnableState_t pm_mode;
    result = nvmlDeviceGetPowerManagementMode(device, &pm_mode);
    if (result == NVML_SUCCESS) {
        power.power_management_enabled = (pm_mode == NVML_FEATURE_ENABLED);
    }
    
    return power;
}

ThermalMetrics MetricsCollector::collect_thermal_metrics(nvmlDevice_t device) {
    ThermalMetrics thermal{};
    nvmlReturn_t result;
    
    // GPU temperature
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        thermal.gpu_temperature_c = static_cast<int>(temp);
    }
    
    // Memory temperature (if available)
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_MEMORY, &temp);
    if (result == NVML_SUCCESS) {
        thermal.memory_temperature_c = static_cast<int>(temp);
    }
    
    // Temperature thresholds
    result = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &temp);
    if (result == NVML_SUCCESS) {
        thermal.shutdown_temperature_c = static_cast<int>(temp);
    }
    
    result = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &temp);
    if (result == NVML_SUCCESS) {
        thermal.slowdown_temperature_c = static_cast<int>(temp);
    }
    
    result = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_MAX_OPERATING, &temp);
    if (result == NVML_SUCCESS) {
        thermal.max_operating_temperature_c = static_cast<int>(temp);
    }
    
    // Check for thermal throttling
    thermal.thermal_throttling_active = check_throttle_reasons(device);
    
    return thermal;
}

MemoryMetrics MetricsCollector::collect_memory_metrics(nvmlDevice_t device) {
    MemoryMetrics memory{};
    nvmlReturn_t result;
    
    // Memory info
    nvmlMemory_t mem_info;
    result = nvmlDeviceGetMemoryInfo(device, &mem_info);
    if (result == NVML_SUCCESS) {
        memory.total_memory_mb = mem_info.total / (1024 * 1024);
        memory.used_memory_mb = mem_info.used / (1024 * 1024);
        memory.free_memory_mb = mem_info.free / (1024 * 1024);
        
        if (memory.total_memory_mb > 0) {
            memory.usage_percentage = (static_cast<double>(memory.used_memory_mb) / 
                                     static_cast<double>(memory.total_memory_mb)) * 100.0;
        }
    }
    
    // Memory clock
    unsigned int clock;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
    if (result == NVML_SUCCESS) {
        memory.memory_clock_mhz = static_cast<int>(clock);
    }
    
    // Memory bus width (not directly available through NVML, would need CUDA)
    memory.memory_bus_width = 0; // Placeholder
    
    // Calculate memory bandwidth
    memory.memory_bandwidth_gb_s = calculate_memory_bandwidth(memory);
    
    return memory;
}

UtilizationMetrics MetricsCollector::collect_utilization_metrics(nvmlDevice_t device) {
    UtilizationMetrics util{};
    nvmlReturn_t result;
    
    // GPU and memory utilization
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result == NVML_SUCCESS) {
        util.gpu_utilization_percentage = static_cast<int>(utilization.gpu);
        util.memory_utilization_percentage = static_cast<int>(utilization.memory);
    }
    
    // Encoder utilization
    unsigned int encoder_util, encoder_sample_period;
    result = nvmlDeviceGetEncoderUtilization(device, &encoder_util, &encoder_sample_period);
    if (result == NVML_SUCCESS) {
        util.encoder_utilization_percentage = static_cast<int>(encoder_util);
    }
    
    // Decoder utilization
    unsigned int decoder_util, decoder_sample_period;
    result = nvmlDeviceGetDecoderUtilization(device, &decoder_util, &decoder_sample_period);
    if (result == NVML_SUCCESS) {
        util.decoder_utilization_percentage = static_cast<int>(decoder_util);
    }
    
    return util;
}

ClockMetrics MetricsCollector::collect_clock_metrics(nvmlDevice_t device) {
    ClockMetrics clocks{};
    nvmlReturn_t result;
    
    // Current clocks
    unsigned int clock;
    
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
    if (result == NVML_SUCCESS) {
        clocks.graphics_clock_mhz = static_cast<int>(clock);
    }
    
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
    if (result == NVML_SUCCESS) {
        clocks.memory_clock_mhz = static_cast<int>(clock);
    }
    
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
    if (result == NVML_SUCCESS) {
        clocks.sm_clock_mhz = static_cast<int>(clock);
    }
    
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_VIDEO, &clock);
    if (result == NVML_SUCCESS) {
        clocks.video_clock_mhz = static_cast<int>(clock);
    }
    
    // Maximum clocks
    result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
    if (result == NVML_SUCCESS) {
        clocks.max_graphics_clock_mhz = static_cast<int>(clock);
    }
    
    result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &clock);
    if (result == NVML_SUCCESS) {
        clocks.max_memory_clock_mhz = static_cast<int>(clock);
    }
    
    return clocks;
}

std::vector<ProcessMetrics> MetricsCollector::collect_process_metrics(nvmlDevice_t device) {
    std::vector<ProcessMetrics> processes;
    
    // Get running processes
    unsigned int info_count = 0;
    nvmlReturn_t result = nvmlDeviceGetGraphicsRunningProcesses(device, &info_count, nullptr);
    
    if (result == NVML_ERROR_INSUFFICIENT_SIZE && info_count > 0) {
        std::vector<nvmlProcessInfo_t> infos(info_count);
        result = nvmlDeviceGetGraphicsRunningProcesses(device, &info_count, infos.data());
        
        if (result == NVML_SUCCESS) {
            for (const auto& info : infos) {
                ProcessMetrics proc;
                proc.pid = info.pid;
                proc.used_memory_mb = info.usedGpuMemory / (1024 * 1024);
                
                // Get process name (platform-specific implementation needed)
                proc.process_name = "process_" + std::to_string(info.pid);
                
                processes.push_back(proc);
            }
        }
    }
    
    // Also check compute processes
    info_count = 0;
    result = nvmlDeviceGetComputeRunningProcesses(device, &info_count, nullptr);
    
    if (result == NVML_ERROR_INSUFFICIENT_SIZE && info_count > 0) {
        std::vector<nvmlProcessInfo_t> infos(info_count);
        result = nvmlDeviceGetComputeRunningProcesses(device, &info_count, infos.data());
        
        if (result == NVML_SUCCESS) {
            for (const auto& info : infos) {
                // Check if process is already in the list
                auto it = std::find_if(processes.begin(), processes.end(),
                    [&info](const ProcessMetrics& p) { return p.pid == info.pid; });
                
                if (it == processes.end()) {
                    ProcessMetrics proc;
                    proc.pid = info.pid;
                    proc.used_memory_mb = info.usedGpuMemory / (1024 * 1024);
                    proc.process_name = "compute_" + std::to_string(info.pid);
                    
                    processes.push_back(proc);
                } else {
                    // Update existing process with compute memory usage
                    it->used_memory_mb += info.usedGpuMemory / (1024 * 1024);
                }
            }
        }
    }
    
    return processes;
}

bool MetricsCollector::is_device_valid(int gpu_id) const {
    return gpu_id >= 0 && static_cast<size_t>(gpu_id) < device_handles_.size();
}

nvmlDevice_t MetricsCollector::get_device_handle(int gpu_id) const {
    if (!is_device_valid(gpu_id)) {
        throw std::invalid_argument("Invalid GPU ID: " + std::to_string(gpu_id));
    }
    
    return device_handles_[gpu_id];
}

void MetricsCollector::initialize_device_handles() {
    unsigned int device_count;
    nvmlReturn_t result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        throw std::runtime_error("Failed to get device count: " + std::string(nvmlErrorString(result)));
    }
    
    device_handles_.clear();
    device_handles_.reserve(device_count);
    
    for (unsigned int i = 0; i < device_count; ++i) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result == NVML_SUCCESS) {
            device_handles_.push_back(device);
        } else {
            Logger::warn("Failed to get handle for device {}: {}", i, nvmlErrorString(result));
        }
    }
    
    Logger::info("Initialized {} device handles", device_handles_.size());
}

double MetricsCollector::calculate_memory_bandwidth(const MemoryMetrics& memory) {
    // Rough estimate based on memory clock and bus width
    // This is a simplified calculation and may not be accurate for all GPUs
    if (memory.memory_clock_mhz > 0 && memory.memory_bus_width > 0) {
        // Bandwidth (GB/s) = Clock (MHz) * Bus Width (bits) * 2 (DDR) / 8 (bits to bytes) / 1000 (MB to GB)
        return (static_cast<double>(memory.memory_clock_mhz) * memory.memory_bus_width * 2) / 8000.0;
    }
    
    return 0.0;
}

void MetricsCollector::update_derived_metrics(GPUMetrics& metrics) {
    // Compute efficiency (utilization vs power ratio)
    if (metrics.power.power_draw_watts > 0) {
        metrics.compute_efficiency = metrics.utilization.gpu_utilization_percentage / metrics.power.power_draw_watts;
    }
    
    // Memory efficiency (memory utilization vs memory bandwidth)
    if (metrics.memory.memory_bandwidth_gb_s > 0) {
        metrics.memory_efficiency = metrics.utilization.memory_utilization_percentage / metrics.memory.memory_bandwidth_gb_s;
    }
    
    // Thermal efficiency (utilization vs temperature)
    if (metrics.thermal.gpu_temperature_c > 0) {
        metrics.thermal_efficiency = metrics.utilization.gpu_utilization_percentage / static_cast<double>(metrics.thermal.gpu_temperature_c);
    }
    
    // Performance indicators
    metrics.is_overheating = metrics.thermal.gpu_temperature_c > 85; // Default threshold
    metrics.is_power_limited = metrics.power.power_usage_percentage > 95.0;
    metrics.is_memory_bottlenecked = (metrics.utilization.memory_utilization_percentage > 90 && 
                                     metrics.utilization.gpu_utilization_percentage < 50);
    metrics.is_throttled = metrics.thermal.thermal_throttling_active;
}

bool MetricsCollector::check_throttle_reasons(nvmlDevice_t device) {
    unsigned long long reasons;
    nvmlReturn_t result = nvmlDeviceGetCurrentClocksThrottleReasons(device, &reasons);
    
    if (result == NVML_SUCCESS) {
        // Check for thermal throttling specifically
        return (reasons & nvmlClocksThrottleReasonThermalSlowdown) != 0;
    }
    
    return false;
}






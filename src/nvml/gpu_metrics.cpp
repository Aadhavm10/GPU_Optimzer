#include "gpu_metrics.h"
#include <sstream>
#include <iomanip>
#include <ctime>

double GPUMetrics::get_age_seconds() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now - timestamp;
    return std::chrono::duration<double>(duration).count();
}

bool GPUMetrics::is_valid() const {
    // Check if metrics contain valid data
    return device_id >= 0 && 
           !device_name.empty() && 
           get_age_seconds() < 300; // Valid for 5 minutes
}

std::string GPUMetrics::to_json() const {
    std::ostringstream json;
    
    // Convert timestamp to ISO 8601 string
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    json << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    std::string timestamp_str = json.str();
    json.str("");
    
    json << std::fixed << std::setprecision(2);
    json << "{\n";
    json << "  \"timestamp\": \"" << timestamp_str << "\",\n";
    json << "  \"device_id\": " << device_id << ",\n";
    json << "  \"device_name\": \"" << device_name << "\",\n";
    
    // Power metrics
    json << "  \"power\": {\n";
    json << "    \"power_draw_watts\": " << power.power_draw_watts << ",\n";
    json << "    \"power_limit_watts\": " << power.power_limit_watts << ",\n";
    json << "    \"power_usage_percentage\": " << power.power_usage_percentage << ",\n";
    json << "    \"power_management_enabled\": " << (power.power_management_enabled ? "true" : "false") << "\n";
    json << "  },\n";
    
    // Thermal metrics
    json << "  \"thermal\": {\n";
    json << "    \"gpu_temperature_c\": " << thermal.gpu_temperature_c << ",\n";
    json << "    \"memory_temperature_c\": " << thermal.memory_temperature_c << ",\n";
    json << "    \"max_operating_temperature_c\": " << thermal.max_operating_temperature_c << ",\n";
    json << "    \"shutdown_temperature_c\": " << thermal.shutdown_temperature_c << ",\n";
    json << "    \"slowdown_temperature_c\": " << thermal.slowdown_temperature_c << ",\n";
    json << "    \"thermal_throttling_active\": " << (thermal.thermal_throttling_active ? "true" : "false") << "\n";
    json << "  },\n";
    
    // Memory metrics
    json << "  \"memory\": {\n";
    json << "    \"total_memory_mb\": " << memory.total_memory_mb << ",\n";
    json << "    \"used_memory_mb\": " << memory.used_memory_mb << ",\n";
    json << "    \"free_memory_mb\": " << memory.free_memory_mb << ",\n";
    json << "    \"usage_percentage\": " << memory.usage_percentage << ",\n";
    json << "    \"memory_clock_mhz\": " << memory.memory_clock_mhz << ",\n";
    json << "    \"memory_bus_width\": " << memory.memory_bus_width << ",\n";
    json << "    \"memory_bandwidth_gb_s\": " << memory.memory_bandwidth_gb_s << "\n";
    json << "  },\n";
    
    // Utilization metrics
    json << "  \"utilization\": {\n";
    json << "    \"gpu_utilization_percentage\": " << utilization.gpu_utilization_percentage << ",\n";
    json << "    \"memory_utilization_percentage\": " << utilization.memory_utilization_percentage << ",\n";
    json << "    \"encoder_utilization_percentage\": " << utilization.encoder_utilization_percentage << ",\n";
    json << "    \"decoder_utilization_percentage\": " << utilization.decoder_utilization_percentage << "\n";
    json << "  },\n";
    
    // Clock metrics
    json << "  \"clocks\": {\n";
    json << "    \"graphics_clock_mhz\": " << clocks.graphics_clock_mhz << ",\n";
    json << "    \"memory_clock_mhz\": " << clocks.memory_clock_mhz << ",\n";
    json << "    \"sm_clock_mhz\": " << clocks.sm_clock_mhz << ",\n";
    json << "    \"video_clock_mhz\": " << clocks.video_clock_mhz << ",\n";
    json << "    \"max_graphics_clock_mhz\": " << clocks.max_graphics_clock_mhz << ",\n";
    json << "    \"max_memory_clock_mhz\": " << clocks.max_memory_clock_mhz << "\n";
    json << "  },\n";
    
    // Derived metrics
    json << "  \"efficiency\": {\n";
    json << "    \"compute_efficiency\": " << compute_efficiency << ",\n";
    json << "    \"memory_efficiency\": " << memory_efficiency << ",\n";
    json << "    \"thermal_efficiency\": " << thermal_efficiency << "\n";
    json << "  },\n";
    
    // Performance indicators
    json << "  \"indicators\": {\n";
    json << "    \"is_throttled\": " << (is_throttled ? "true" : "false") << ",\n";
    json << "    \"is_overheating\": " << (is_overheating ? "true" : "false") << ",\n";
    json << "    \"is_power_limited\": " << (is_power_limited ? "true" : "false") << ",\n";
    json << "    \"is_memory_bottlenecked\": " << (is_memory_bottlenecked ? "true" : "false") << "\n";
    json << "  }\n";
    
    json << "}";
    
    return json.str();
}

std::string GPUMetrics::to_csv_line() const {
    std::ostringstream csv;
    
    // Convert timestamp to epoch seconds
    auto epoch = timestamp.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch).count();
    
    csv << std::fixed << std::setprecision(2);
    csv << seconds << ",";
    csv << device_id << ",";
    csv << "\"" << device_name << "\",";
    
    // Power metrics
    csv << power.power_draw_watts << ",";
    csv << power.power_limit_watts << ",";
    csv << power.power_usage_percentage << ",";
    csv << (power.power_management_enabled ? 1 : 0) << ",";
    
    // Thermal metrics
    csv << thermal.gpu_temperature_c << ",";
    csv << thermal.memory_temperature_c << ",";
    csv << thermal.max_operating_temperature_c << ",";
    csv << (thermal.thermal_throttling_active ? 1 : 0) << ",";
    
    // Memory metrics
    csv << memory.total_memory_mb << ",";
    csv << memory.used_memory_mb << ",";
    csv << memory.free_memory_mb << ",";
    csv << memory.usage_percentage << ",";
    csv << memory.memory_clock_mhz << ",";
    csv << memory.memory_bandwidth_gb_s << ",";
    
    // Utilization metrics
    csv << utilization.gpu_utilization_percentage << ",";
    csv << utilization.memory_utilization_percentage << ",";
    csv << utilization.encoder_utilization_percentage << ",";
    csv << utilization.decoder_utilization_percentage << ",";
    
    // Clock metrics
    csv << clocks.graphics_clock_mhz << ",";
    csv << clocks.memory_clock_mhz << ",";
    csv << clocks.sm_clock_mhz << ",";
    csv << clocks.max_graphics_clock_mhz << ",";
    csv << clocks.max_memory_clock_mhz << ",";
    
    // Derived metrics
    csv << compute_efficiency << ",";
    csv << memory_efficiency << ",";
    csv << thermal_efficiency << ",";
    
    // Performance indicators
    csv << (is_throttled ? 1 : 0) << ",";
    csv << (is_overheating ? 1 : 0) << ",";
    csv << (is_power_limited ? 1 : 0) << ",";
    csv << (is_memory_bottlenecked ? 1 : 0);
    
    return csv.str();
}

std::string GPUMetrics::get_csv_header() {
    return "timestamp,device_id,device_name,"
           "power_draw_watts,power_limit_watts,power_usage_percentage,power_management_enabled,"
           "gpu_temperature_c,memory_temperature_c,max_operating_temperature_c,thermal_throttling_active,"
           "total_memory_mb,used_memory_mb,free_memory_mb,memory_usage_percentage,memory_clock_mhz,memory_bandwidth_gb_s,"
           "gpu_utilization_percentage,memory_utilization_percentage,encoder_utilization_percentage,decoder_utilization_percentage,"
           "graphics_clock_mhz,memory_clock_mhz,sm_clock_mhz,max_graphics_clock_mhz,max_memory_clock_mhz,"
           "compute_efficiency,memory_efficiency,thermal_efficiency,"
           "is_throttled,is_overheating,is_power_limited,is_memory_bottlenecked";
}






#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <signal.h>

// #include "cuda/profiling/cuda_profiler.h"  // Disabled temporarily
#include "nvml/nvml_monitor.h"
#include "common/config.h"
#include "common/logger.h"

volatile bool g_running = true;

void signal_handler(int signal) {
    Logger::info("Received signal {}, shutting down gracefully...", signal);
    g_running = false;
}

void print_usage(const char* program_name) {
    std::cout << "GPU Utilization Optimizer\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -c, --config FILE       Configuration file path\n";
    std::cout << "  -v, --verbose           Enable verbose logging\n";
    std::cout << "  -d, --daemon            Run as daemon (background service)\n";
    std::cout << "  -p, --port PORT         Dashboard port (default: 8080)\n";
    std::cout << "  -s, --sample-rate HZ    NVML sampling rate (default: 10Hz)\n";
    std::cout << "  --benchmark             Run matrix multiplication benchmarks\n";
    std::cout << "  --profile-matrix SIZE   Profile matrix multiplication of given size\n";
    std::cout << "  --profile-vector SIZE   Profile vector operations of given size\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " --benchmark\n";
    std::cout << "  " << program_name << " --profile-matrix 1024\n";
    std::cout << "  " << program_name << " -d -p 8080 -s 100\n";
}

int main(int argc, char* argv[]) {
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    Config config;
    bool daemon_mode = false;
    bool run_benchmark = false;
    int matrix_size = 0;
    int vector_size = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verbose") {
            config.log_level = LogLevel::DEBUG;
        } else if (arg == "-d" || arg == "--daemon") {
            daemon_mode = true;
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                config.dashboard_port = std::stoi(argv[++i]);
            }
        } else if (arg == "-s" || arg == "--sample-rate") {
            if (i + 1 < argc) {
                config.nvml_sample_rate = std::stoi(argv[++i]);
            }
        } else if (arg == "--profile-matrix") {
            if (i + 1 < argc) {
                matrix_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--profile-vector") {
            if (i + 1 < argc) {
                vector_size = std::stoi(argv[++i]);
            }
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config.config_file = argv[++i];
            }
        }
    }

    // Initialize logger
    Logger::init(config.log_level);
    Logger::info("GPU Utilization Optimizer starting...");

    try {
        // Initialize CUDA profiler
        // auto cuda_profiler = std::make_unique<CudaProfiler>(config);  // Disabled temporarily
        
        // Initialize NVML monitor
        auto nvml_monitor = std::make_unique<NVMLMonitor>(config);

        if (run_benchmark) {
            Logger::info("Running matrix multiplication benchmarks...");
            // cuda_profiler->run_matrix_benchmarks();  // Disabled temporarily
            Logger::warn("CUDA profiling temporarily disabled - install Visual Studio C++ compiler");
            return 0;
        }

        if (matrix_size > 0) {
            Logger::info("Profiling matrix multiplication of size {}x{}", matrix_size, matrix_size);
            // cuda_profiler->profile_matrix_multiplication(matrix_size);  // Disabled temporarily
            Logger::warn("CUDA profiling temporarily disabled - install Visual Studio C++ compiler");
            return 0;
        }

        if (vector_size > 0) {
            Logger::info("Profiling vector operations of size {}", vector_size);
            // cuda_profiler->profile_vector_operations(vector_size);  // Disabled temporarily
            Logger::warn("CUDA profiling temporarily disabled - install Visual Studio C++ compiler");
            return 0;
        }

        if (daemon_mode) {
            Logger::info("Starting monitoring daemon on port {}", config.dashboard_port);
            
            // Start NVML monitoring in background thread
            nvml_monitor->start_monitoring();
            
            // Main monitoring loop
            while (g_running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            nvml_monitor->stop_monitoring();
        } else {
            // Interactive mode - run single profiling session
            Logger::info("Running interactive profiling session...");
            // cuda_profiler->run_interactive_session();  // Disabled temporarily
            Logger::warn("CUDA profiling temporarily disabled - running NVML monitoring only");
            nvml_monitor->start_monitoring();
            
            Logger::info("Press Ctrl+C to stop monitoring...");
            while (g_running) {
                auto metrics = nvml_monitor->get_current_metrics();
                if (metrics.is_valid()) {
                    Logger::info("GPU {}: {}% util, {:.1f}% mem, {}°C", 
                               metrics.device_id, 
                               metrics.utilization.gpu_utilization_percentage,
                               metrics.memory.usage_percentage,
                               metrics.thermal.gpu_temperature_c);
                }
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            
            nvml_monitor->stop_monitoring();
        }

    } catch (const std::exception& e) {
        Logger::error("Fatal error: {}", e.what());
        return 1;
    }

    Logger::info("GPU Utilization Optimizer shutting down...");
    return 0;
}

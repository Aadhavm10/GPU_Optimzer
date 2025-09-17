#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "common/config.h"
#include "timer.h"

struct KernelProfile {
    std::string name;
    double execution_time_ms;
    double gflops;
    double memory_bandwidth_gb_s;
    double occupancy;
    size_t shared_memory_bytes;
    size_t registers_per_thread;
    dim3 grid_size;
    dim3 block_size;
    bool correctness_passed;
};

struct BenchmarkResult {
    std::string operation;
    std::vector<KernelProfile> kernel_profiles;
    KernelProfile best_kernel;
    double cublas_time_ms;
    double speedup_vs_cublas;
    size_t problem_size;
};

class CudaProfiler {
public:
    explicit CudaProfiler(const Config& config);
    ~CudaProfiler();

    // Matrix operations profiling
    void profile_matrix_multiplication(int size);
    BenchmarkResult benchmark_matrix_multiplication(int m, int n, int k);
    void run_matrix_benchmarks();

    // Vector operations profiling  
    void profile_vector_operations(int size);
    BenchmarkResult benchmark_sgemv(int m, int n);
    BenchmarkResult benchmark_vector_reduction(int size);

    // Interactive session
    void run_interactive_session();

    // Utility functions
    void warm_up_gpu();
    bool verify_cuda_capability();
    void print_device_info();

private:
    Config config_;
    cublasHandle_t cublas_handle_;
    Timer timer_;
    
    // Memory management
    void* d_A_;
    void* d_B_;
    void* d_C_;
    void* d_result_;
    size_t max_matrix_size_;
    
    // Profiling utilities
    double measure_kernel_time(std::function<void()> kernel_func, int iterations = 10);
    double calculate_gflops(size_t operations, double time_ms);
    double calculate_memory_bandwidth(size_t bytes_transferred, double time_ms);
    double get_kernel_occupancy(const void* kernel_func, int block_size);
    
    // Matrix kernel variants
    void run_naive_matrix_multiply(float* A, float* B, float* C, int m, int n, int k);
    void run_tiled_matrix_multiply(float* A, float* B, float* C, int m, int n, int k);
    void run_shared_memory_matrix_multiply(float* A, float* B, float* C, int m, int n, int k);
    void run_vectorized_matrix_multiply(float* A, float* B, float* C, int m, int n, int k);
    void run_double_buffered_matrix_multiply(float* A, float* B, float* C, int m, int n, int k);
    
    // Vector kernel variants
    void run_naive_sgemv(float* A, float* x, float* y, int m, int n);
    void run_optimized_sgemv(float* A, float* x, float* y, int m, int n);
    void run_vector_reduction(float* input, float* output, int size);
    
    // Verification
    bool verify_matrix_result(const float* gpu_result, const float* cpu_result, int size);
    bool verify_vector_result(const float* gpu_result, const float* cpu_result, int size);
    void compute_reference_matrix_multiply(const float* A, const float* B, float* C, int m, int n, int k);
    void compute_reference_sgemv(const float* A, const float* x, float* y, int m, int n);
    
    // Memory allocation helpers
    void allocate_matrix_memory(int max_size);
    void deallocate_matrix_memory();
    void initialize_random_data(float* data, int size);
    
    // Reporting
    void print_benchmark_results(const BenchmarkResult& result);
    void export_results_to_json(const std::vector<BenchmarkResult>& results, const std::string& filename);
};






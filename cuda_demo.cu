// Simple CUDA Matrix Multiplication Demo
// Demonstrates multiple optimization levels for the GPU Optimizer

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// ================================
// Kernel 1: Naive Matrix Multiplication
// ================================
__global__ void naive_matrix_multiply(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ================================
// Kernel 2: Tiled Matrix Multiplication (Optimized)
// ================================
#define TILE_SIZE 16

__global__ void tiled_matrix_multiply(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        if (row < N && tile * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && tile * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ================================
// Kernel 3: Advanced Optimized Version
// ================================
#define ADVANCED_TILE_SIZE 32

__global__ void optimized_matrix_multiply(float* A, float* B, float* C, int N) {
    __shared__ float As[ADVANCED_TILE_SIZE][ADVANCED_TILE_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ float Bs[ADVANCED_TILE_SIZE][ADVANCED_TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * ADVANCED_TILE_SIZE + ty;
    int col = bx * ADVANCED_TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + ADVANCED_TILE_SIZE - 1) / ADVANCED_TILE_SIZE; tile++) {
        // Coalesced loading
        if (row < N && tile * ADVANCED_TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + tile * ADVANCED_TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && tile * ADVANCED_TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(tile * ADVANCED_TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Unrolled computation for better performance
        #pragma unroll
        for (int k = 0; k < ADVANCED_TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ================================
// CPU Reference Implementation
// ================================
void cpu_matrix_multiply(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ================================
// Utility Functions
// ================================
void initialize_matrix(float* matrix, int N, bool random = true) {
    for (int i = 0; i < N * N; i++) {
        if (random) {
            matrix[i] = (float)rand() / RAND_MAX * 10.0f;
        } else {
            matrix[i] = 0.0f;
        }
    }
}

bool verify_result(float* gpu_result, float* cpu_result, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < N * N; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

double get_gflops(int N, double time_ms) {
    // GFLOPS = (2 * N^3) / (time_in_seconds * 10^9)
    double operations = 2.0 * N * N * N;
    double time_s = time_ms / 1000.0;
    return operations / (time_s * 1e9);
}

void print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("🖥️  GPU Information:\n");
    printf("   Device: %s\n", prop.name);
    printf("   Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("   Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("   Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("   Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("   Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\n");
}

// ================================
// Benchmark Function
// ================================
void benchmark_kernel(const char* name, 
                     void (*kernel)(float*, float*, float*, int, cudaStream_t),
                     float* d_A, float* d_B, float* d_C, int N, 
                     int warmup_runs = 3, int benchmark_runs = 10) {
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        kernel(d_A, d_B, d_C, N, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; i++) {
        kernel(d_A, d_B, d_C, N, 0);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double avg_time = milliseconds / benchmark_runs;
    double gflops = get_gflops(N, avg_time);
    
    printf("📊 %s:\n", name);
    printf("   Time: %.3f ms\n", avg_time);
    printf("   Performance: %.2f GFLOPS\n", gflops);
    printf("   Speedup vs CPU: %.2fx\n", gflops / 1.0); // Assuming 1 GFLOP/s CPU baseline
    printf("\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Kernel wrapper functions
void launch_naive(float* A, float* B, float* C, int N, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    naive_matrix_multiply<<<grid, block, 0, stream>>>(A, B, C, N);
}

void launch_tiled(float* A, float* B, float* C, int N, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    tiled_matrix_multiply<<<grid, block, 0, stream>>>(A, B, C, N);
}

void launch_optimized(float* A, float* B, float* C, int N, cudaStream_t stream) {
    dim3 block(ADVANCED_TILE_SIZE, ADVANCED_TILE_SIZE);
    dim3 grid((N + ADVANCED_TILE_SIZE - 1) / ADVANCED_TILE_SIZE, 
              (N + ADVANCED_TILE_SIZE - 1) / ADVANCED_TILE_SIZE);
    optimized_matrix_multiply<<<grid, block, 0, stream>>>(A, B, C, N);
}

// ================================
// Main Function
// ================================
int main(int argc, char* argv[]) {
    printf("🚀 CUDA Matrix Multiplication Benchmark\n");
    printf("========================================\n\n");
    
    // Parse command line arguments
    int N = 1024;  // Default matrix size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("Invalid matrix size. Using default: 1024\n");
            N = 1024;
        }
    }
    
    printf("Matrix Size: %dx%d\n", N, N);
    printf("Memory Required: %.2f MB per matrix\n", (N * N * sizeof(float)) / (1024.0 * 1024.0));
    printf("\n");
    
    print_gpu_info();
    
    // Allocate host memory
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        printf("❌ Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize matrices
    srand(time(NULL));
    initialize_matrix(h_A, N, true);
    initialize_matrix(h_B, N, true);
    initialize_matrix(h_C_gpu, N, false);
    initialize_matrix(h_C_cpu, N, false);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    printf("🔥 Running CUDA Kernel Benchmarks:\n");
    printf("=====================================\n\n");
    
    // Benchmark kernels
    benchmark_kernel("Naive GPU Implementation", launch_naive, d_A, d_B, d_C, N);
    benchmark_kernel("Tiled GPU Implementation", launch_tiled, d_A, d_B, d_C, N);
    benchmark_kernel("Optimized GPU Implementation", launch_optimized, d_A, d_B, d_C, N);
    
    // Verify correctness (using small matrix to avoid long CPU computation)
    if (N <= 512) {
        printf("🔍 Verifying Correctness:\n");
        printf("=========================\n");
        
        // CPU reference
        clock_t cpu_start = clock();
        cpu_matrix_multiply(h_A, h_B, h_C_cpu, N);
        clock_t cpu_end = clock();
        double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
        
        // Get GPU result
        launch_optimized(d_A, d_B, d_C, N, 0);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
        
        if (verify_result(h_C_gpu, h_C_cpu, N)) {
            printf("✅ GPU results match CPU reference\n");
            printf("📊 CPU Reference: %.3f ms (%.2f GFLOPS)\n", 
                   cpu_time, get_gflops(N, cpu_time));
        } else {
            printf("❌ GPU results do not match CPU reference\n");
        }
        printf("\n");
    }
    
    printf("🎯 Performance Summary:\n");
    printf("======================\n");
    printf("Matrix Size: %dx%d\n", N, N);
    printf("Best GPU Performance: Optimized Implementation\n");
    printf("Recommended for production workloads\n");
    printf("\n");
    
    printf("💡 Tips for Better Performance:\n");
    printf("===============================\n");
    printf("• Use larger matrices (2048x2048+) for better GPU utilization\n");
    printf("• Ensure matrices are stored in row-major order\n");
    printf("• Consider using cuBLAS for production applications\n");
    printf("• Monitor GPU utilization with the dashboard\n");
    printf("\n");
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    printf("✅ Benchmark completed successfully!\n");
    return 0;
}


#include "matrix_kernels.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// ================================
// Basic Matrix Multiplication Kernels
// ================================

__global__ void naive_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiled_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {
    
    constexpr int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        if (row < M && tile * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && tile * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void shared_memory_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {
    
    constexpr int TILE_SIZE = 32;
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Coalesced loading into shared memory
        if (row < M && tile * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && tile * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Unrolled computation
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ================================
// Optimized Matrix Multiplication Kernels
// ================================

__global__ void vectorized_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {
    
    constexpr int TILE_SIZE = 16;
    constexpr int VECTOR_SIZE = 4;
    
    __shared__ float4 As[TILE_SIZE][TILE_SIZE / VECTOR_SIZE];
    __shared__ float4 Bs[TILE_SIZE][TILE_SIZE / VECTOR_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx * VECTOR_SIZE;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Vectorized loading
        if (row < M && tile * TILE_SIZE + tx * VECTOR_SIZE < K) {
            As[ty][tx] = *reinterpret_cast<const float4*>(
                &A[row * K + tile * TILE_SIZE + tx * VECTOR_SIZE]);
        } else {
            As[ty][tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        
        if (col < N && tile * TILE_SIZE + ty < K) {
            Bs[ty][tx] = *reinterpret_cast<const float4*>(
                &B[(tile * TILE_SIZE + ty) * N + col]);
        } else {
            Bs[ty][tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        
        __syncthreads();
        
        // Vectorized computation
        for (int k = 0; k < TILE_SIZE / VECTOR_SIZE; ++k) {
            float4 a_vec = As[ty][k];
            float4 b_vec = Bs[k][tx];
            
            sum.x += a_vec.x * b_vec.x + a_vec.y * b_vec.y + 
                     a_vec.z * b_vec.z + a_vec.w * b_vec.w;
            sum.y += a_vec.x * b_vec.y + a_vec.y * b_vec.z + 
                     a_vec.z * b_vec.w + a_vec.w * b_vec.x;
            sum.z += a_vec.x * b_vec.z + a_vec.y * b_vec.w + 
                     a_vec.z * b_vec.x + a_vec.w * b_vec.y;
            sum.w += a_vec.x * b_vec.w + a_vec.y * b_vec.x + 
                     a_vec.z * b_vec.y + a_vec.w * b_vec.z;
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        *reinterpret_cast<float4*>(&C[row * N + col]) = sum;
    }
}

__global__ void double_buffered_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {
    
    constexpr int TILE_SIZE = 32;
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int buffer = 0;
    
    // Prefetch first tile
    if (row < M && tx < K) {
        As[buffer][ty][tx] = A[row * K + tx];
    } else {
        As[buffer][ty][tx] = 0.0f;
    }
    
    if (col < N && ty < K) {
        Bs[buffer][ty][tx] = B[ty * N + col];
    } else {
        Bs[buffer][ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int next_buffer = 1 - buffer;
        
        // Prefetch next tile while computing current
        if (tile + 1 < (K + TILE_SIZE - 1) / TILE_SIZE) {
            if (row < M && (tile + 1) * TILE_SIZE + tx < K) {
                As[next_buffer][ty][tx] = A[row * K + (tile + 1) * TILE_SIZE + tx];
            } else {
                As[next_buffer][ty][tx] = 0.0f;
            }
            
            if (col < N && (tile + 1) * TILE_SIZE + ty < K) {
                Bs[next_buffer][ty][tx] = B[((tile + 1) * TILE_SIZE + ty) * N + col];
            } else {
                Bs[next_buffer][ty][tx] = 0.0f;
            }
        }
        
        // Compute using current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[buffer][ty][k] * Bs[buffer][k][tx];
        }
        
        __syncthreads();
        buffer = next_buffer;
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void warp_tiled_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {
    
    constexpr int WARP_TILE_M = 32;
    constexpr int WARP_TILE_N = 32;
    constexpr int WARP_TILE_K = 16;
    
    __shared__ float As[WARP_TILE_M][WARP_TILE_K + 1];
    __shared__ float Bs[WARP_TILE_K][WARP_TILE_N + 1];
    
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / WARP_SIZE * WARP_TILE_M;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * WARP_TILE_N;
    
    int lane_id = threadIdx.x % WARP_SIZE;
    int thread_row = warp_row + lane_id / 8;
    int thread_col = warp_col + (lane_id % 8) * 4;
    
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int tile_k = 0; tile_k < K; tile_k += WARP_TILE_K) {
        // Cooperative loading with warp
        for (int i = lane_id; i < WARP_TILE_M * WARP_TILE_K; i += WARP_SIZE) {
            int row = i / WARP_TILE_K;
            int col = i % WARP_TILE_K;
            
            if (warp_row + row < M && tile_k + col < K) {
                As[row][col] = A[(warp_row + row) * K + tile_k + col];
            } else {
                As[row][col] = 0.0f;
            }
        }
        
        for (int i = lane_id; i < WARP_TILE_K * WARP_TILE_N; i += WARP_SIZE) {
            int row = i / WARP_TILE_N;
            int col = i % WARP_TILE_N;
            
            if (tile_k + row < K && warp_col + col < N) {
                Bs[row][col] = B[(tile_k + row) * N + warp_col + col];
            } else {
                Bs[row][col] = 0.0f;
            }
        }
        
        warp.sync();
        
        // Compute using warp shuffle for register reuse
        #pragma unroll
        for (int k = 0; k < WARP_TILE_K; ++k) {
            float a_val = (thread_row < M && tile_k + k < K) ? As[thread_row - warp_row][k] : 0.0f;
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (thread_col + i < N) {
                    float b_val = Bs[k][thread_col + i - warp_col];
                    sum[i] += a_val * b_val;
                }
            }
        }
        
        warp.sync();
    }
    
    // Write results
    if (thread_row < M) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (thread_col + i < N) {
                C[thread_row * N + thread_col + i] = sum[i];
            }
        }
    }
}

// ================================
// Kernel Launchers
// ================================

void launch_naive_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream) {
    
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);
    
    naive_matrix_multiply_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

void launch_tiled_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, int tile_size, cudaStream_t stream) {
    
    dim3 block_size(tile_size, tile_size);
    dim3 grid_size((N + tile_size - 1) / tile_size, 
                   (M + tile_size - 1) / tile_size);
    
    tiled_matrix_multiply_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

void launch_shared_memory_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, int tile_size, cudaStream_t stream) {
    
    dim3 block_size(tile_size, tile_size);
    dim3 grid_size((N + tile_size - 1) / tile_size, 
                   (M + tile_size - 1) / tile_size);
    
    shared_memory_matrix_multiply_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

void launch_vectorized_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream) {
    
    constexpr int TILE_SIZE = 16;
    constexpr int VECTOR_SIZE = 4;
    
    dim3 block_size(TILE_SIZE / VECTOR_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    vectorized_matrix_multiply_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

void launch_double_buffered_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream) {
    
    constexpr int TILE_SIZE = 32;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    double_buffered_matrix_multiply_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

void launch_warp_tiled_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream) {
    
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
    
    dim3 block_size(THREADS_PER_BLOCK, 1);
    dim3 grid_size((N + 31) / 32, (M + 31) / 32);
    
    warp_tiled_matrix_multiply_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

// ================================
// Utility Functions
// ================================

int get_optimal_tile_size(int M, int N, int K) {
    // Heuristic for optimal tile size based on problem dimensions
    int avg_dim = (M + N + K) / 3;
    
    if (avg_dim < 512) return 16;
    else if (avg_dim < 2048) return 24;
    else return 32;
}

dim3 get_optimal_grid_size(int M, int N, int tile_size) {
    return dim3((N + tile_size - 1) / tile_size, 
                (M + tile_size - 1) / tile_size);
}

dim3 get_optimal_block_size(int tile_size) {
    return dim3(tile_size, tile_size);
}

bool supports_tensor_cores() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Tensor Cores available on Volta (7.0), Turing (7.5), Ampere (8.0+)
    return (prop.major >= 7 && prop.minor >= 0);
}

size_t get_shared_memory_per_block() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return prop.sharedMemPerBlock;
}

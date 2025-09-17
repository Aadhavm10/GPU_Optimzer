#include "vector_kernels.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

using namespace cooperative_groups;

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// ================================
// Matrix-Vector Multiplication (SGEMV)
// ================================

__global__ void naive_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

__global__ void coalesced_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        
        // Process in chunks for better memory coalescing
        constexpr int CHUNK_SIZE = 4;
        int chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        for (int chunk = 0; chunk < chunks; ++chunk) {
            int col_start = chunk * CHUNK_SIZE;
            int col_end = min(col_start + CHUNK_SIZE, N);
            
            for (int col = col_start; col < col_end; ++col) {
                sum += A[row * N + col] * x[col];
            }
        }
        
        y[row] = sum;
    }
}

__global__ void shared_memory_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N) {
    
    constexpr int TILE_SIZE = 256;
    __shared__ float x_shared[TILE_SIZE];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load x vector tile into shared memory
        int col = tile * TILE_SIZE + tid;
        if (col < N) {
            x_shared[tid] = x[col];
        } else {
            x_shared[tid] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        if (row < M) {
            int tile_end = min(TILE_SIZE, N - tile * TILE_SIZE);
            for (int k = 0; k < tile_end; ++k) {
                int col = tile * TILE_SIZE + k;
                sum += A[row * N + col] * x_shared[k];
            }
        }
        
        __syncthreads();
    }
    
    if (row < M) {
        y[row] = sum;
    }
}

__global__ void warp_reduce_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    __shared__ float warp_sums[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    
    if (row < M) {
        float sum = 0.0f;
        
        // Each thread processes multiple elements
        for (int col = lane; col < N; col += WARP_SIZE) {
            sum += A[row * N + col] * x[col];
        }
        
        // Warp-level reduction
        auto warp = tiled_partition<WARP_SIZE>(this_thread_block());
        sum = reduce(warp, sum, plus<float>());
        
        // Store warp result
        if (lane == 0) {
            warp_sums[warp_id] = sum;
        }
        
        __syncthreads();
        
        // Final reduction across warps
        if (warp_id == 0 && lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
            sum = warp_sums[lane];
            sum = reduce(warp, sum, plus<float>());
            
            if (lane == 0) {
                y[row] = sum;
            }
        }
    }
}

// ================================
// Vector Reductions
// ================================

__global__ void naive_vector_sum_kernel(
    const float* input, float* output, int N) {
    
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void shared_memory_reduce_kernel(
    const float* input, float* output, int N) {
    
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and perform first level of reduction
    sdata[tid] = 0.0f;
    if (i < N) sdata[tid] += input[i];
    if (i + blockDim.x < N) sdata[tid] += input[i + blockDim.x];
    
    __syncthreads();
    
    // Unrolled reduction for better performance
    if (blockDim.x >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Warp-level reduction
    if (tid < 32) {
        if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
        if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
        if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
        if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
        if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
        if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void warp_reduce_kernel(
    const float* input, float* output, int N) {
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (i < N) ? input[i] : 0.0f;
    
    // Warp-level reduction using shuffle
    auto warp = tiled_partition<WARP_SIZE>(this_thread_block());
    val = reduce(warp, val, plus<float>());
    
    // Store warp results in shared memory
    __shared__ float warp_results[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        val = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? warp_results[tid] : 0.0f;
        val = reduce(warp, val, plus<float>());
        
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

__global__ void cooperative_reduce_kernel(
    const float* input, float* output, int N) {
    
    // Use CUB for highly optimized cooperative reduction
    typedef cub::BlockReduce<float, MAX_THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (i < N) ? input[i] : 0.0f;
    
    // Perform block-wide reduction
    float block_sum = BlockReduce(temp_storage).Sum(val);
    
    if (tid == 0) {
        output[blockIdx.x] = block_sum;
    }
}

// ================================
// Element-wise Operations
// ================================

__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_multiply_kernel(
    const float* a, const float* b, float* c, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        c[i] = a[i] * b[i];
    }
}

__global__ void vector_scale_kernel(
    const float* input, float scalar, float* output, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        output[i] = input[i] * scalar;
    }
}

__global__ void vector_saxpy_kernel(
    float alpha, const float* x, float* y, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        y[i] = alpha * x[i] + y[i];
    }
}

// ================================
// Advanced Vector Operations
// ================================

__global__ void vector_norm_kernel(
    const float* input, float* output, int N) {
    
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute squared values
    float val = (i < N) ? input[i] : 0.0f;
    sdata[tid] = val * val;
    
    __syncthreads();
    
    // Reduction to compute sum of squares
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void vector_dot_product_kernel(
    const float* a, const float* b, float* result, int N) {
    
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute products
    sdata[tid] = (i < N) ? a[i] * b[i] : 0.0f;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

__global__ void vector_max_kernel(
    const float* input, float* output, int* index, int N) {
    
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    __shared__ int sindex[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (i < N) {
        sdata[tid] = input[i];
        sindex[tid] = i;
    } else {
        sdata[tid] = -FLT_MAX;
        sindex[tid] = -1;
    }
    
    __syncthreads();
    
    // Reduction to find maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
        index[blockIdx.x] = sindex[0];
    }
}

__global__ void vector_min_kernel(
    const float* input, float* output, int* index, int N) {
    
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    __shared__ int sindex[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (i < N) {
        sdata[tid] = input[i];
        sindex[tid] = i;
    } else {
        sdata[tid] = FLT_MAX;
        sindex[tid] = -1;
    }
    
    __syncthreads();
    
    // Reduction to find minimum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
        index[blockIdx.x] = sindex[0];
    }
}

// ================================
// Vectorized Operations
// ================================

__global__ void vectorized_add_kernel(
    const float4* a, const float4* b, float4* c, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float4 a_val = a[i];
        float4 b_val = b[i];
        
        c[i] = make_float4(
            a_val.x + b_val.x,
            a_val.y + b_val.y,
            a_val.z + b_val.z,
            a_val.w + b_val.w
        );
    }
}

__global__ void vectorized_multiply_kernel(
    const float4* a, const float4* b, float4* c, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float4 a_val = a[i];
        float4 b_val = b[i];
        
        c[i] = make_float4(
            a_val.x * b_val.x,
            a_val.y * b_val.y,
            a_val.z * b_val.z,
            a_val.w * b_val.w
        );
    }
}

__global__ void vectorized_reduce_kernel(
    const float4* input, float* output, int N) {
    
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and sum float4 components
    float sum = 0.0f;
    if (i < N) {
        float4 val = input[i];
        sum = val.x + val.y + val.z + val.w;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Standard reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ================================
// Kernel Launchers
// ================================

void launch_naive_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    naive_sgemv_kernel<<<grid_size, block_size, 0, stream>>>(A, x, y, M, N);
}

void launch_coalesced_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    coalesced_sgemv_kernel<<<grid_size, block_size, 0, stream>>>(A, x, y, M, N);
}

void launch_shared_memory_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    shared_memory_sgemv_kernel<<<grid_size, block_size, 0, stream>>>(A, x, y, M, N);
}

void launch_warp_reduce_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    warp_reduce_sgemv_kernel<<<grid_size, block_size, 0, stream>>>(A, x, y, M, N);
}

void launch_vector_reduction(
    const float* input, float* output, int N, 
    const std::string& method, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    if (method == "naive") {
        naive_vector_sum_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N);
    } else if (method == "shared") {
        shared_memory_reduce_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N);
    } else if (method == "warp") {
        warp_reduce_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N);
    } else if (method == "cooperative") {
        cooperative_reduce_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N);
    } else {
        // Default to warp reduction
        warp_reduce_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N);
    }
}

void launch_vector_add(
    const float* a, const float* b, float* c, int N, 
    bool vectorized, cudaStream_t stream) {
    
    if (vectorized && N % 4 == 0) {
        int block_size = 256;
        int grid_size = (N / 4 + block_size - 1) / block_size;
        
        vectorized_add_kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            N / 4
        );
    } else {
        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;
        
        vector_add_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, N);
    }
}

void launch_vector_multiply(
    const float* a, const float* b, float* c, int N, 
    bool vectorized, cudaStream_t stream) {
    
    if (vectorized && N % 4 == 0) {
        int block_size = 256;
        int grid_size = (N / 4 + block_size - 1) / block_size;
        
        vectorized_multiply_kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            N / 4
        );
    } else {
        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;
        
        vector_multiply_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, N);
    }
}

// Additional launcher implementations...
void launch_vector_scale(
    const float* input, float scalar, float* output, int N, 
    cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    vector_scale_kernel<<<grid_size, block_size, 0, stream>>>(input, scalar, output, N);
}

void launch_vector_saxpy(
    float alpha, const float* x, float* y, int N, 
    cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    vector_saxpy_kernel<<<grid_size, block_size, 0, stream>>>(alpha, x, y, N);
}

void launch_vector_norm(
    const float* input, float* output, int N, 
    cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    vector_norm_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N);
}

void launch_vector_dot_product(
    const float* a, const float* b, float* result, int N, 
    cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    vector_dot_product_kernel<<<grid_size, block_size, 0, stream>>>(a, b, result, N);
}

void launch_vector_max(
    const float* input, float* output, int* index, int N, 
    cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    vector_max_kernel<<<grid_size, block_size, 0, stream>>>(input, output, index, N);
}

void launch_vector_min(
    const float* input, float* output, int* index, int N, 
    cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    vector_min_kernel<<<grid_size, block_size, 0, stream>>>(input, output, index, N);
}

// ================================
// Utility Functions
// ================================

int get_optimal_block_size_for_reduction(int N) {
    // Use powers of 2 for better warp efficiency
    if (N < 512) return 128;
    else if (N < 2048) return 256;
    else return 512;
}

int get_optimal_block_size_for_sgemv(int M, int N) {
    // Optimize based on matrix dimensions
    if (M < 1024) return 128;
    else if (M < 4096) return 256;
    else return 512;
}

dim3 get_sgemv_grid_size(int M, int N, int block_size) {
    return dim3((M + block_size - 1) / block_size);
}

bool should_use_vectorized_operations(int N) {
    // Use vectorized operations for large arrays that are 4-aligned
    return (N >= 1024 && N % 4 == 0);
}

size_t get_reduction_shared_memory_size(int block_size) {
    return block_size * sizeof(float);
}






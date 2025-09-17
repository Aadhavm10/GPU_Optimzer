#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Vector operation kernel declarations
extern "C" {

// Matrix-Vector multiplication (SGEMV)
__global__ void naive_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N);

__global__ void coalesced_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N);

__global__ void shared_memory_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N);

__global__ void warp_reduce_sgemv_kernel(
    const float* A, const float* x, float* y, 
    int M, int N);

// Vector reductions
__global__ void naive_vector_sum_kernel(
    const float* input, float* output, int N);

__global__ void shared_memory_reduce_kernel(
    const float* input, float* output, int N);

__global__ void warp_reduce_kernel(
    const float* input, float* output, int N);

__global__ void cooperative_reduce_kernel(
    const float* input, float* output, int N);

// Element-wise operations
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int N);

__global__ void vector_multiply_kernel(
    const float* a, const float* b, float* c, int N);

__global__ void vector_scale_kernel(
    const float* input, float scalar, float* output, int N);

__global__ void vector_saxpy_kernel(
    float alpha, const float* x, float* y, int N);

// Advanced vector operations
__global__ void vector_norm_kernel(
    const float* input, float* output, int N);

__global__ void vector_dot_product_kernel(
    const float* a, const float* b, float* result, int N);

__global__ void vector_max_kernel(
    const float* input, float* output, int* index, int N);

__global__ void vector_min_kernel(
    const float* input, float* output, int* index, int N);

// Vectorized operations
__global__ void vectorized_add_kernel(
    const float4* a, const float4* b, float4* c, int N);

__global__ void vectorized_multiply_kernel(
    const float4* a, const float4* b, float4* c, int N);

__global__ void vectorized_reduce_kernel(
    const float4* input, float* output, int N);

}

// Template kernels for different data types
template<typename T, int BLOCK_SIZE>
__global__ void templated_reduce_kernel(
    const T* input, T* output, int N);

template<typename T>
__global__ void templated_vector_add_kernel(
    const T* a, const T* b, T* c, int N);

// Kernel launcher functions
void launch_naive_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream = 0);

void launch_coalesced_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream = 0);

void launch_shared_memory_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream = 0);

void launch_warp_reduce_sgemv(
    const float* A, const float* x, float* y, 
    int M, int N, cudaStream_t stream = 0);

void launch_vector_reduction(
    const float* input, float* output, int N, 
    const std::string& method = "warp", cudaStream_t stream = 0);

void launch_vector_add(
    const float* a, const float* b, float* c, int N, 
    bool vectorized = true, cudaStream_t stream = 0);

void launch_vector_multiply(
    const float* a, const float* b, float* c, int N, 
    bool vectorized = true, cudaStream_t stream = 0);

void launch_vector_scale(
    const float* input, float scalar, float* output, int N, 
    cudaStream_t stream = 0);

void launch_vector_saxpy(
    float alpha, const float* x, float* y, int N, 
    cudaStream_t stream = 0);

void launch_vector_norm(
    const float* input, float* output, int N, 
    cudaStream_t stream = 0);

void launch_vector_dot_product(
    const float* a, const float* b, float* result, int N, 
    cudaStream_t stream = 0);

void launch_vector_max(
    const float* input, float* output, int* index, int N, 
    cudaStream_t stream = 0);

void launch_vector_min(
    const float* input, float* output, int* index, int N, 
    cudaStream_t stream = 0);

// Utility functions
int get_optimal_block_size_for_reduction(int N);
int get_optimal_block_size_for_sgemv(int M, int N);
dim3 get_sgemv_grid_size(int M, int N, int block_size);
bool should_use_vectorized_operations(int N);
size_t get_reduction_shared_memory_size(int block_size);

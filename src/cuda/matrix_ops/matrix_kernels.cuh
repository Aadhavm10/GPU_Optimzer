#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Kernel function declarations
extern "C" {

// Basic kernels
__global__ void naive_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

__global__ void tiled_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

__global__ void shared_memory_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

// Optimized kernels
__global__ void vectorized_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

__global__ void double_buffered_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

__global__ void warp_tiled_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

__global__ void bank_conflict_free_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

// Advanced kernels
__global__ void tensor_core_matrix_multiply_kernel(
    const __half* A, const __half* B, float* C, 
    int M, int N, int K);

__global__ void mixed_precision_matrix_multiply_kernel(
    const __half* A, const __half* B, float* C, 
    int M, int N, int K);

__global__ void prefetch_matrix_multiply_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K);

}

// Template kernels for different data types
template<typename T, int TILE_SIZE>
__global__ void templated_matrix_multiply_kernel(
    const T* A, const T* B, T* C, 
    int M, int N, int K);

// Kernel launcher functions
void launch_naive_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

void launch_tiled_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, int tile_size = 16, cudaStream_t stream = 0);

void launch_shared_memory_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, int tile_size = 32, cudaStream_t stream = 0);

void launch_vectorized_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

void launch_double_buffered_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

void launch_warp_tiled_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

void launch_bank_conflict_free_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

void launch_tensor_core_matrix_multiply(
    const __half* A, const __half* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

void launch_mixed_precision_matrix_multiply(
    const __half* A, const __half* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

void launch_prefetch_matrix_multiply(
    const float* A, const float* B, float* C, 
    int M, int N, int K, cudaStream_t stream = 0);

// Utility functions
int get_optimal_tile_size(int M, int N, int K);
dim3 get_optimal_grid_size(int M, int N, int tile_size);
dim3 get_optimal_block_size(int tile_size);
bool supports_tensor_cores();
size_t get_shared_memory_per_block();






# CUDA profiling library
add_library(cuda_profiling
    profiling/cuda_profiler.cpp
    profiling/cuda_profiler.cu
    profiling/timer.cpp
    matrix_ops/matrix_kernels.cu
    matrix_ops/optimized_sgemm.cu
    vector_ops/vector_kernels.cu
    vector_ops/sgemv_kernels.cu
    utils/cuda_utils.cu
    utils/device_info.cu
)

target_include_directories(cuda_profiling PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(cuda_profiling
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Set CUDA properties
set_target_properties(cuda_profiling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)






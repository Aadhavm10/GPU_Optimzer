cmake_minimum_required(VERSION 3.20)
project(gpu_utilization_optimizer LANGUAGES CXX)
# CUDA disabled temporarily - add back after fixing toolchain
# project(gpu_utilization_optimizer LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
# find_package(CUDA REQUIRED)  # Disabled temporarily
find_package(Threads REQUIRED)

# Set CUDA architectures (Compute Capability 7.0+)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(src)

# NVML library
find_library(NVML_LIBRARY nvidia-ml HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
if(NOT NVML_LIBRARY)
    message(FATAL_ERROR "NVML library not found")
endif()

# Add subdirectories
# add_subdirectory(src/cuda)  # Disabled temporarily until CUDA toolchain fixed
add_subdirectory(src/nvml)
add_subdirectory(tests)

# Main executable
add_executable(gpu_optimizer
    src/main.cpp
)

target_link_libraries(gpu_optimizer
    # cuda_profiling  # Disabled temporarily
    nvml_monitoring
    ${NVML_LIBRARY}
    # ${CUDA_LIBRARIES}  # Disabled temporarily
    Threads::Threads
)

# Python bindings (optional)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        add_subdirectory(src/python_bindings)
    else()
        message(WARNING "pybind11 not found, skipping Python bindings")
    endif()
endif()

# Testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_subdirectory(tests)
    else()
        message(WARNING "GTest not found, skipping tests")
    endif()
endif()

# Installation
install(TARGETS gpu_optimizer DESTINATION bin)
install(DIRECTORY src/ DESTINATION include/gpu_optimizer 
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")

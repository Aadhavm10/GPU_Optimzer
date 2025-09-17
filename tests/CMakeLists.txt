# Test configuration
if(BUILD_TESTS)
    # Find GTest
    find_package(GTest REQUIRED)
    
    # Include directories
    include_directories(${CMAKE_SOURCE_DIR}/src)
    
    # Test executable
    add_executable(gpu_optimizer_tests
        test_main.cpp
        test_matrix_kernels.cpp
        test_vector_kernels.cpp
        test_nvml_monitor.cpp
        test_config.cpp
        test_timer.cpp
    )
    
    target_link_libraries(gpu_optimizer_tests
        cuda_profiling
        nvml_monitoring
        GTest::GTest
        GTest::Main
        ${CUDA_LIBRARIES}
        ${NVML_LIBRARY}
    )
    
    # Register tests with CTest
    add_test(NAME MatrixKernelTests COMMAND gpu_optimizer_tests --gtest_filter="MatrixKernel*")
    add_test(NAME VectorKernelTests COMMAND gpu_optimizer_tests --gtest_filter="VectorKernel*")
    add_test(NAME NVMLMonitorTests COMMAND gpu_optimizer_tests --gtest_filter="NVMLMonitor*")
    add_test(NAME ConfigTests COMMAND gpu_optimizer_tests --gtest_filter="Config*")
    add_test(NAME TimerTests COMMAND gpu_optimizer_tests --gtest_filter="Timer*")
endif()

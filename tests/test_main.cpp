#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "common/logger.h"

class GPUOptimizerTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // Initialize CUDA
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            GTEST_SKIP() << "CUDA not available: " << cudaGetErrorString(error);
        }
        
        // Initialize logger for tests
        Logger::init(LogLevel::ERROR); // Suppress most logging during tests
        
        // Verify GPU is available
        int device_count;
        error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        
        std::cout << "Running tests with " << device_count << " CUDA device(s)" << std::endl;
    }
    
    void TearDown() override {
        // Cleanup CUDA
        cudaDeviceReset();
    }
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new GPUOptimizerTestEnvironment);
    return RUN_ALL_TESTS();
}

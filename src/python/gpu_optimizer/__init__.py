"""
GPU Utilization Optimizer

A comprehensive CUDA-based profiling and optimization tool for GPU workloads.
"""

__version__ = "1.0.0"
__author__ = "GPU Optimizer Team"
__email__ = "gpu-optimizer@example.com"

# Core imports
try:
    from .cuda_profiling import CudaProfiler, MatrixKernels, VectorKernels
    from .nvml_monitoring import NVMLMonitor, GPUMetrics
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Python-only components (always available)
from .dashboard import create_dashboard, DashboardApp
from .analysis import PerformanceAnalyzer, BenchmarkRunner
from .utils import GPUInfo, SystemInfo
from .config import Config, load_config

# High-level API
def profile_matrix_operations(sizes=None, algorithms=None, **kwargs):
    """
    High-level function to profile matrix operations.
    
    Args:
        sizes: List of matrix sizes to test (default: [512, 1024, 2048])
        algorithms: List of algorithms to test (default: all available)
        **kwargs: Additional configuration options
    
    Returns:
        BenchmarkResults object with profiling data
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA profiling not available. Please install with CUDA support.")
    
    profiler = CudaProfiler(**kwargs)
    return profiler.profile_matrix_operations(sizes, algorithms)

def profile_vector_operations(sizes=None, operations=None, **kwargs):
    """
    High-level function to profile vector operations.
    
    Args:
        sizes: List of vector sizes to test (default: [1M, 10M, 100M])
        operations: List of operations to test (default: all available)
        **kwargs: Additional configuration options
    
    Returns:
        BenchmarkResults object with profiling data
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA profiling not available. Please install with CUDA support.")
    
    profiler = CudaProfiler(**kwargs)
    return profiler.profile_vector_operations(sizes, operations)

def start_gpu_monitoring(config=None, callback=None):
    """
    Start real-time GPU monitoring.
    
    Args:
        config: Configuration object or dict
        callback: Optional callback function for real-time data
    
    Returns:
        NVMLMonitor instance
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("NVML monitoring not available. Please install with CUDA support.")
    
    monitor = NVMLMonitor(config or {})
    if callback:
        monitor.register_callback(callback)
    monitor.start_monitoring()
    return monitor

def create_performance_dashboard(port=8080, **kwargs):
    """
    Create and launch performance dashboard.
    
    Args:
        port: Port to run dashboard on (default: 8080)
        **kwargs: Additional dashboard configuration
    
    Returns:
        DashboardApp instance
    """
    app = create_dashboard(port=port, **kwargs)
    return app

# Utility functions
def get_gpu_info():
    """Get information about available GPUs."""
    return GPUInfo.get_all_gpus()

def get_system_info():
    """Get system information including CUDA availability."""
    return {
        "cuda_available": CUDA_AVAILABLE,
        "gpu_info": get_gpu_info(),
        "system_info": SystemInfo.get_system_info(),
        "version": __version__
    }

def check_requirements():
    """Check if all requirements are met for GPU profiling."""
    issues = []
    
    if not CUDA_AVAILABLE:
        issues.append("CUDA profiling modules not available")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count == 0:
            issues.append("No NVIDIA GPUs detected")
    except Exception as e:
        issues.append(f"NVML initialization failed: {e}")
    
    if issues:
        print("Requirements check failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("All requirements met!")
    return True

# Module-level constants
DEFAULT_CONFIG = {
    "cuda": {
        "warmup_iterations": 5,
        "benchmark_iterations": 100,
        "enable_correctness_checks": True,
    },
    "nvml": {
        "sample_rate": 10,
        "enable_power_monitoring": True,
        "enable_thermal_monitoring": True,
    },
    "dashboard": {
        "port": 8080,
        "host": "localhost",
        "debug": False,
    }
}

# Export public API
__all__ = [
    # Core classes
    "CudaProfiler", "NVMLMonitor", "DashboardApp",
    "MatrixKernels", "VectorKernels", "GPUMetrics",
    
    # High-level functions
    "profile_matrix_operations", "profile_vector_operations",
    "start_gpu_monitoring", "create_performance_dashboard",
    
    # Utilities
    "get_gpu_info", "get_system_info", "check_requirements",
    "load_config", "Config",
    
    # Analysis
    "PerformanceAnalyzer", "BenchmarkRunner",
    
    # Constants
    "CUDA_AVAILABLE", "DEFAULT_CONFIG",
    "__version__", "__author__"
]

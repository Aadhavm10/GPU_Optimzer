#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import subprocess

# Get CUDA installation path
def find_cuda():
    """Find CUDA installation path"""
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path and os.path.exists(cuda_path):
        return cuda_path
    
    # Common CUDA installation paths
    common_paths = [
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/local/cuda-12.0',
        '/usr/local/cuda-11.8',
        'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0',
        'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    raise RuntimeError("CUDA installation not found. Please set CUDA_PATH environment variable.")

# Get project version
def get_version():
    """Extract version from CMakeLists.txt or use default"""
    try:
        with open('CMakeLists.txt', 'r') as f:
            content = f.read()
            # Look for version in CMakeLists.txt
            for line in content.split('\n'):
                if 'project(' in line and 'VERSION' in line:
                    return line.split('VERSION')[1].split()[0].strip()
    except:
        pass
    return "1.0.0"

# Check if build directory exists and has been configured
def check_cmake_build():
    """Check if CMake build has been configured"""
    build_dir = os.path.join(os.path.dirname(__file__), 'build')
    if not os.path.exists(build_dir):
        print("Warning: CMake build directory not found.")
        print("Please run: mkdir build && cd build && cmake .. && make")
        return False
    
    if not os.path.exists(os.path.join(build_dir, 'CMakeCache.txt')):
        print("Warning: CMake not configured.")
        print("Please run: cd build && cmake .. && make")
        return False
    
    return True

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
            requirements = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove comments and optional dependencies
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line and not line.startswith('cupy') and not line.startswith('numba'):
                        requirements.append(line)
            return requirements
    except FileNotFoundError:
        return []

# CUDA extension setup
cuda_path = find_cuda()
check_cmake_build()

# Define extensions
ext_modules = []

# Only build extensions if we can find CUDA and pybind11
try:
    ext_modules = [
        Pybind11Extension(
            "gpu_optimizer.cuda_profiling",
            [
                "src/python_bindings/cuda_profiling_bindings.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
                os.path.join(cuda_path, "include"),
                "src",
                "src/cuda",
                "src/common"
            ],
            libraries=["cudart", "cublas", "cuda"],
            library_dirs=[
                os.path.join(cuda_path, "lib64"),
                os.path.join(cuda_path, "lib", "x64"),
                "build/src/cuda"
            ],
            language='c++',
            cxx_std=17,
        ),
        Pybind11Extension(
            "gpu_optimizer.nvml_monitoring",
            [
                "src/python_bindings/nvml_monitoring_bindings.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
                os.path.join(cuda_path, "include"),
                "src",
                "src/nvml",
                "src/common"
            ],
            libraries=["nvidia-ml"],
            library_dirs=[
                os.path.join(cuda_path, "lib64"),
                os.path.join(cuda_path, "lib", "x64"),
                "build/src/nvml"
            ],
            language='c++',
            cxx_std=17,
        ),
    ]
except Exception as e:
    print(f"Warning: Could not setup C++ extensions: {e}")
    print("GPU Optimizer will be installed in Python-only mode.")
    ext_modules = []

setup(
    name="gpu-utilization-optimizer",
    version=get_version(),
    author="GPU Optimizer Team",
    author_email="gpu-optimizer@example.com",
    description="Comprehensive CUDA-based GPU profiling and optimization tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gpu-utilization-optimizer",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
        "cuda": [
            "cupy-cuda11x>=9.0.0",
            "numba>=0.54.0",
        ],
        "ml": [
            "tensorflow>=2.7.0",
            "torch>=1.10.0",
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpu-optimizer=gpu_optimizer.cli:main",
            "gpu-dashboard=gpu_optimizer.dashboard.app:main",
            "gpu-benchmark=gpu_optimizer.benchmarks.runner:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gpu_optimizer": [
            "dashboard/templates/*.html",
            "dashboard/static/css/*.css",
            "dashboard/static/js/*.js",
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
    # Ensure wheel is built with CUDA support tags
    options={
        "bdist_wheel": {
            "plat_name": "linux_x86_64",  # Adjust based on platform
        }
    },
)

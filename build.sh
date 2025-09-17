#!/bin/bash

# GPU Utilization Optimizer Build Script
# This script automates the build process for the GPU Optimizer

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check CUDA installation
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_success "CUDA found: version $CUDA_VERSION"
        return 0
    else
        print_error "CUDA not found. Please install CUDA Toolkit 12.x or 11.x"
        print_status "Download from: https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
}

# Function to check CMake
check_cmake() {
    print_status "Checking CMake installation..."
    
    if command_exists cmake; then
        CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
        print_success "CMake found: version $CMAKE_VERSION"
        
        # Check if version is >= 3.20
        if [[ $(echo "$CMAKE_VERSION 3.20" | tr " " "\n" | sort -V | head -n1) == "3.20" ]]; then
            return 0
        else
            print_warning "CMake version $CMAKE_VERSION found, but 3.20+ is recommended"
            return 0
        fi
    else
        print_error "CMake not found. Please install CMake 3.20+"
        return 1
    fi
}

# Function to check Python
check_python() {
    print_status "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_success "Python found: version $PYTHON_VERSION"
        return 0
    elif command_exists python; then
        PYTHON_VERSION=$(python --version | awk '{print $2}')
        print_success "Python found: version $PYTHON_VERSION"
        return 0
    else
        print_error "Python not found. Please install Python 3.8+"
        return 1
    fi
}

# Function to check GPU drivers
check_gpu_drivers() {
    print_status "Checking GPU drivers..."
    
    if command_exists nvidia-smi; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)
        print_success "NVIDIA drivers found: version $DRIVER_VERSION"
        
        # Check GPU count
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_status "Found $GPU_COUNT GPU(s)"
        
        return 0
    else
        print_error "nvidia-smi not found. Please install NVIDIA drivers"
        return 1
    fi
}

# Function to setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_warning "requirements.txt not found"
    fi
}

# Function to configure CMake build
configure_build() {
    print_status "Configuring CMake build..."
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Detect build system
    local CMAKE_GENERATOR=""
    if command_exists ninja; then
        CMAKE_GENERATOR="-GNinja"
        print_status "Using Ninja build system"
    else
        print_status "Using Make build system"
    fi
    
    # Configure with CMake
    cmake .. \
        $CMAKE_GENERATOR \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    print_success "CMake configuration completed"
    cd ..
}

# Function to build the project
build_project() {
    print_status "Building the project..."
    
    cd build
    
    # Determine number of parallel jobs
    local JOBS=""
    if command_exists nproc; then
        JOBS="-j$(nproc)"
    elif [[ $(detect_os) == "macos" ]]; then
        JOBS="-j$(sysctl -n hw.ncpu)"
    else
        JOBS="-j4"  # Default fallback
    fi
    
    # Build
    if command_exists ninja && [[ -f "build.ninja" ]]; then
        ninja
    else
        make $JOBS
    fi
    
    print_success "Build completed successfully"
    cd ..
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd build
    
    if command_exists ctest; then
        ctest --output-on-failure
        print_success "All tests passed"
    else
        print_warning "CTest not available, skipping tests"
    fi
    
    cd ..
}

# Function to install Python package
install_python_package() {
    print_status "Installing Python package..."
    
    # Activate virtual environment if it exists
    if [[ -d "venv" ]]; then
        source venv/bin/activate
    fi
    
    # Install in development mode
    pip install -e .
    
    print_success "Python package installed"
}

# Function to run basic verification
verify_installation() {
    print_status "Verifying installation..."
    
    # Test basic functionality
    if [[ -x "build/gpu_optimizer" ]]; then
        print_status "Testing basic functionality..."
        ./build/gpu_optimizer --help > /dev/null
        print_success "Basic functionality test passed"
    else
        print_error "gpu_optimizer executable not found"
        return 1
    fi
    
    # Test GPU detection
    if command_exists nvidia-smi; then
        print_status "Testing GPU detection..."
        ./build/gpu_optimizer --profile-matrix 64 > /dev/null 2>&1 || true
        print_success "GPU detection test completed"
    fi
}

# Function to display usage information
show_usage() {
    echo "GPU Utilization Optimizer Build Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --check-only     Only check dependencies, don't build"
    echo "  --no-python      Skip Python environment setup"
    echo "  --no-tests       Skip running tests"
    echo "  --clean          Clean build directory before building"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full build with all components"
    echo "  $0 --check-only      # Only check if dependencies are available"
    echo "  $0 --clean           # Clean build and rebuild everything"
    echo "  $0 --no-python       # Build C++/CUDA components only"
}

# Main build function
main() {
    local CHECK_ONLY=false
    local NO_PYTHON=false
    local NO_TESTS=false
    local CLEAN_BUILD=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --no-python)
                NO_PYTHON=true
                shift
                ;;
            --no-tests)
                NO_TESTS=true
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo "=================================================================="
    echo "          GPU Utilization Optimizer Build Script"
    echo "=================================================================="
    echo ""
    
    # Check system requirements
    print_status "Checking system requirements..."
    
    local REQUIREMENTS_OK=true
    
    check_cuda || REQUIREMENTS_OK=false
    check_cmake || REQUIREMENTS_OK=false
    check_python || REQUIREMENTS_OK=false
    check_gpu_drivers || REQUIREMENTS_OK=false
    
    if [[ "$REQUIREMENTS_OK" == false ]]; then
        print_error "Some requirements are missing. Please install them and try again."
        exit 1
    fi
    
    print_success "All requirements satisfied"
    
    # Exit if only checking
    if [[ "$CHECK_ONLY" == true ]]; then
        print_success "Dependency check completed successfully"
        exit 0
    fi
    
    # Clean build if requested
    if [[ "$CLEAN_BUILD" == true ]]; then
        print_status "Cleaning previous build..."
        rm -rf build
        print_success "Build directory cleaned"
    fi
    
    # Setup Python environment
    if [[ "$NO_PYTHON" == false ]]; then
        setup_python_env
    fi
    
    # Configure and build
    configure_build
    build_project
    
    # Run tests
    if [[ "$NO_TESTS" == false ]]; then
        run_tests
    fi
    
    # Install Python package
    if [[ "$NO_PYTHON" == false ]]; then
        install_python_package
    fi
    
    # Verify installation
    verify_installation
    
    echo ""
    echo "=================================================================="
    print_success "Build completed successfully!"
    echo "=================================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run basic benchmark:    ./build/gpu_optimizer --benchmark"
    echo "  2. Start monitoring:       ./build/gpu_optimizer --daemon"
    echo "  3. Profile operations:     ./build/gpu_optimizer --profile-matrix 1024"
    echo ""
    if [[ "$NO_PYTHON" == false ]]; then
        echo "  4. Start Python dashboard: source venv/bin/activate && gpu-dashboard"
        echo "  5. Use Python API:         python -c \"import gpu_optimizer; gpu_optimizer.check_requirements()\""
        echo ""
    fi
    echo "For more information, see README.md"
}

# Run main function
main "$@"

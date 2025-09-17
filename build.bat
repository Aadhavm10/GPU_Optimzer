@echo off
REM GPU Utilization Optimizer Build Script for Windows
REM This script automates the build process for the GPU Optimizer on Windows

setlocal enabledelayedexpansion

REM Colors are not easily available in batch, so we'll use simple text
set "INFO_PREFIX=[INFO]"
set "SUCCESS_PREFIX=[SUCCESS]"
set "WARNING_PREFIX=[WARNING]"
set "ERROR_PREFIX=[ERROR]"

REM Function to check if command exists
where /q %1 >nul 2>&1
if %errorlevel% equ 0 (
    set "COMMAND_EXISTS=1"
) else (
    set "COMMAND_EXISTS=0"
)

echo ==================================================================
echo           GPU Utilization Optimizer Build Script
echo ==================================================================
echo.

echo %INFO_PREFIX% Checking system requirements...

REM Check CUDA installation
echo %INFO_PREFIX% Checking CUDA installation...
where /q nvcc >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=6" %%i in ('nvcc --version ^| findstr "release"') do (
        set "CUDA_VERSION=%%i"
        set "CUDA_VERSION=!CUDA_VERSION:~1!"
    )
    echo %SUCCESS_PREFIX% CUDA found: version !CUDA_VERSION!
) else (
    echo %ERROR_PREFIX% CUDA not found. Please install CUDA Toolkit 12.x or 11.x
    echo %INFO_PREFIX% Download from: https://developer.nvidia.com/cuda-downloads
    goto :error_exit
)

REM Check CMake
echo %INFO_PREFIX% Checking CMake installation...
where /q cmake >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('cmake --version ^| findstr "cmake version"') do (
        set "CMAKE_VERSION=%%i"
    )
    echo %SUCCESS_PREFIX% CMake found: version !CMAKE_VERSION!
) else (
    echo %ERROR_PREFIX% CMake not found. Please install CMake 3.20+
    goto :error_exit
)

REM Check Python
echo %INFO_PREFIX% Checking Python installation...
where /q python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version') do (
        set "PYTHON_VERSION=%%i"
    )
    echo %SUCCESS_PREFIX% Python found: version !PYTHON_VERSION!
) else (
    echo %ERROR_PREFIX% Python not found. Please install Python 3.8+
    goto :error_exit
)

REM Check NVIDIA drivers
echo %INFO_PREFIX% Checking GPU drivers...
where /q nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "skip=1" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader^,nounits') do (
        set "DRIVER_VERSION=%%i"
        goto :driver_found
    )
    :driver_found
    echo %SUCCESS_PREFIX% NVIDIA drivers found: version !DRIVER_VERSION!
    
    REM Count GPUs
    for /f %%i in ('nvidia-smi --list-gpus ^| find /c "GPU"') do (
        set "GPU_COUNT=%%i"
    )
    echo %INFO_PREFIX% Found !GPU_COUNT! GPU^(s^)
) else (
    echo %ERROR_PREFIX% nvidia-smi not found. Please install NVIDIA drivers
    goto :error_exit
)

echo %SUCCESS_PREFIX% All requirements satisfied

REM Parse command line arguments
set "CHECK_ONLY=0"
set "NO_PYTHON=0"
set "NO_TESTS=0"
set "CLEAN_BUILD=0"

:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--check-only" (
    set "CHECK_ONLY=1"
    shift
    goto :parse_args
)
if "%~1"=="--no-python" (
    set "NO_PYTHON=1"
    shift
    goto :parse_args
)
if "%~1"=="--no-tests" (
    set "NO_TESTS=1"
    shift
    goto :parse_args
)
if "%~1"=="--clean" (
    set "CLEAN_BUILD=1"
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_usage
)
echo %ERROR_PREFIX% Unknown option: %~1
goto :show_usage

:done_parsing

REM Exit if only checking
if %CHECK_ONLY% equ 1 (
    echo %SUCCESS_PREFIX% Dependency check completed successfully
    exit /b 0
)

REM Clean build if requested
if %CLEAN_BUILD% equ 1 (
    echo %INFO_PREFIX% Cleaning previous build...
    if exist build rmdir /s /q build
    echo %SUCCESS_PREFIX% Build directory cleaned
)

REM Setup Python virtual environment
if %NO_PYTHON% equ 0 (
    echo %INFO_PREFIX% Setting up Python virtual environment...
    
    if not exist venv (
        python -m venv venv
        echo %SUCCESS_PREFIX% Virtual environment created
    ) else (
        echo %INFO_PREFIX% Virtual environment already exists
    )
    
    REM Activate virtual environment
    call venv\Scripts\activate.bat
    
    REM Upgrade pip
    python -m pip install --upgrade pip
    
    REM Install requirements
    if exist requirements.txt (
        echo %INFO_PREFIX% Installing Python dependencies...
        pip install -r requirements.txt
        echo %SUCCESS_PREFIX% Python dependencies installed
    ) else (
        echo %WARNING_PREFIX% requirements.txt not found
    )
)

REM Configure CMake build
echo %INFO_PREFIX% Configuring CMake build...

if not exist build mkdir build
cd build

REM Check for Ninja
where /q ninja >nul 2>&1
if %errorlevel% equ 0 (
    set "CMAKE_GENERATOR=-G Ninja"
    echo %INFO_PREFIX% Using Ninja build system
) else (
    set "CMAKE_GENERATOR="
    echo %INFO_PREFIX% Using Visual Studio build system
)

REM Configure with CMake
cmake .. ^
    %CMAKE_GENERATOR% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_TESTS=ON ^
    -DBUILD_PYTHON_BINDINGS=ON ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if %errorlevel% neq 0 (
    echo %ERROR_PREFIX% CMake configuration failed
    cd ..
    goto :error_exit
)

echo %SUCCESS_PREFIX% CMake configuration completed

REM Build the project
echo %INFO_PREFIX% Building the project...

REM Determine number of parallel jobs
for /f %%i in ('echo %NUMBER_OF_PROCESSORS%') do set "JOBS=%%i"

REM Build
if exist build.ninja (
    ninja
) else (
    cmake --build . --config Release --parallel %JOBS%
)

if %errorlevel% neq 0 (
    echo %ERROR_PREFIX% Build failed
    cd ..
    goto :error_exit
)

echo %SUCCESS_PREFIX% Build completed successfully
cd ..

REM Run tests
if %NO_TESTS% equ 0 (
    echo %INFO_PREFIX% Running tests...
    cd build
    where /q ctest >nul 2>&1
    if %errorlevel% equ 0 (
        ctest --output-on-failure
        if %errorlevel% equ 0 (
            echo %SUCCESS_PREFIX% All tests passed
        ) else (
            echo %WARNING_PREFIX% Some tests failed
        )
    ) else (
        echo %WARNING_PREFIX% CTest not available, skipping tests
    )
    cd ..
)

REM Install Python package
if %NO_PYTHON% equ 0 (
    echo %INFO_PREFIX% Installing Python package...
    
    REM Activate virtual environment if it exists
    if exist venv call venv\Scripts\activate.bat
    
    REM Install in development mode
    pip install -e .
    
    echo %SUCCESS_PREFIX% Python package installed
)

REM Verify installation
echo %INFO_PREFIX% Verifying installation...

if exist build\gpu_optimizer.exe (
    echo %INFO_PREFIX% Testing basic functionality...
    build\gpu_optimizer.exe --help >nul 2>&1
    if %errorlevel% equ 0 (
        echo %SUCCESS_PREFIX% Basic functionality test passed
    ) else (
        echo %WARNING_PREFIX% Basic functionality test failed
    )
) else (
    echo %ERROR_PREFIX% gpu_optimizer.exe not found
    goto :error_exit
)

echo.
echo ==================================================================
echo %SUCCESS_PREFIX% Build completed successfully!
echo ==================================================================
echo.
echo Next steps:
echo   1. Run basic benchmark:    build\gpu_optimizer.exe --benchmark
echo   2. Start monitoring:       build\gpu_optimizer.exe --daemon
echo   3. Profile operations:     build\gpu_optimizer.exe --profile-matrix 1024
echo.
if %NO_PYTHON% equ 0 (
    echo   4. Start Python dashboard: venv\Scripts\activate ^&^& gpu-dashboard
    echo   5. Use Python API:         python -c "import gpu_optimizer; gpu_optimizer.check_requirements()"
    echo.
)
echo For more information, see README.md

exit /b 0

:show_usage
echo GPU Utilization Optimizer Build Script
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --check-only     Only check dependencies, don't build
echo   --no-python      Skip Python environment setup
echo   --no-tests       Skip running tests
echo   --clean          Clean build directory before building
echo   --help           Show this help message
echo.
echo Examples:
echo   %~nx0                    # Full build with all components
echo   %~nx0 --check-only      # Only check if dependencies are available
echo   %~nx0 --clean           # Clean build and rebuild everything
echo   %~nx0 --no-python       # Build C++/CUDA components only
exit /b 0

:error_exit
echo.
echo %ERROR_PREFIX% Build failed. Please check the errors above and try again.
exit /b 1






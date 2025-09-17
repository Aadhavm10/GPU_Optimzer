#!/usr/bin/env python3
"""
Simple GPU Monitor - Quick Start Version
A simplified GPU monitoring tool using pynvml to get you started immediately.
"""

import time
try:
    import nvidia_ml_py as pynvml
except ImportError:
    import pynvml
from datetime import datetime
import sys

def initialize_nvml():
    """Initialize NVIDIA Management Library"""
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"❌ Failed to initialize NVML: {e}")
        print("Make sure you have NVIDIA drivers installed and a GPU available.")
        return False

def get_gpu_info():
    """Get basic GPU information"""
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"🔍 Found {device_count} GPU(s)")
        
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = mem_info.total // (1024**2)  # Convert to MB
            
            gpus.append({
                'id': i,
                'name': name,
                'handle': handle,
                'total_memory_mb': total_mem
            })
            
            print(f"  GPU {i}: {name} ({total_mem} MB)")
        
        return gpus
    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")
        return []

def get_gpu_metrics(handle):
    """Get current GPU metrics"""
    try:
        metrics = {}
        
        # GPU utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics['gpu_util'] = util.gpu
            metrics['memory_util'] = util.memory
        except:
            metrics['gpu_util'] = 0
            metrics['memory_util'] = 0
        
        # Memory usage
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics['memory_used_mb'] = mem_info.used // (1024**2)
            metrics['memory_total_mb'] = mem_info.total // (1024**2)
            metrics['memory_percent'] = (mem_info.used / mem_info.total) * 100
        except:
            metrics['memory_used_mb'] = 0
            metrics['memory_total_mb'] = 0
            metrics['memory_percent'] = 0
        
        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics['temperature'] = temp
        except:
            metrics['temperature'] = 0
        
        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert mW to W
            metrics['power_watts'] = power
        except:
            metrics['power_watts'] = 0
        
        # Clock speeds
        try:
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            metrics['graphics_clock_mhz'] = graphics_clock
            metrics['memory_clock_mhz'] = memory_clock
        except:
            metrics['graphics_clock_mhz'] = 0
            metrics['memory_clock_mhz'] = 0
        
        return metrics
    except Exception as e:
        print(f"❌ Error getting metrics: {e}")
        return {}

def format_metrics(gpu_id, gpu_name, metrics):
    """Format metrics for display"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    return (
        f"[{timestamp}] 🖥️  GPU {gpu_id} ({gpu_name})\n"
        f"  📊 Utilization: {metrics.get('gpu_util', 0):3d}% GPU, {metrics.get('memory_util', 0):3d}% Memory\n"
        f"  🧠 Memory: {metrics.get('memory_used_mb', 0):,} MB / {metrics.get('memory_total_mb', 0):,} MB ({metrics.get('memory_percent', 0):.1f}%)\n"
        f"  🌡️  Temperature: {metrics.get('temperature', 0)}°C\n"
        f"  ⚡ Power: {metrics.get('power_watts', 0)} W\n"
        f"  🔄 Clocks: {metrics.get('graphics_clock_mhz', 0)} MHz GPU, {metrics.get('memory_clock_mhz', 0)} MHz Memory\n"
    )

def monitor_gpus(gpus, duration_seconds=None, update_interval=2):
    """Monitor GPUs continuously"""
    print(f"\n🚀 Starting GPU monitoring (update every {update_interval}s)")
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    
    try:
        while True:
            # Clear screen (works on Windows)
            print("\033[2J\033[H", end="")
            
            print("=" * 80)
            print("🖥️  GPU UTILIZATION OPTIMIZER - REAL-TIME MONITORING")
            print("=" * 80)
            
            for gpu in gpus:
                metrics = get_gpu_metrics(gpu['handle'])
                if metrics:
                    print(format_metrics(gpu['id'], gpu['name'], metrics))
            
            # Check duration limit
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                print(f"✅ Monitoring completed ({duration_seconds}s)")
                break
            
            print(f"⏱️  Running for {int(time.time() - start_time)}s - Press Ctrl+C to stop")
            print("=" * 80)
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print(f"\n✅ Monitoring stopped by user after {int(time.time() - start_time)}s")

def main():
    """Main function"""
    print("🖥️  GPU Utilization Optimizer - Simple Monitor")
    print("=" * 50)
    
    # Initialize NVML
    if not initialize_nvml():
        sys.exit(1)
    
    # Get GPU information
    gpus = get_gpu_info()
    if not gpus:
        print("❌ No GPUs found or accessible")
        sys.exit(1)
    
    print(f"\n✅ Successfully detected {len(gpus)} GPU(s)")
    
    # Parse command line arguments
    duration = None
    interval = 2
    
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "--help" or sys.argv[1] == "-h":
                print("\nUsage:")
                print("  python simple_gpu_monitor.py [duration] [interval]")
                print("  python simple_gpu_monitor.py 60 1    # Monitor for 60s, update every 1s")
                print("  python simple_gpu_monitor.py         # Monitor indefinitely, update every 2s")
                sys.exit(0)
            
            duration = int(sys.argv[1])
            if len(sys.argv) > 2:
                interval = float(sys.argv[2])
        except ValueError:
            print("❌ Invalid arguments. Use numbers for duration and interval.")
            sys.exit(1)
    
    # Start monitoring
    try:
        monitor_gpus(gpus, duration, interval)
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
    finally:
        pynvml.nvmlShutdown()
        print("👋 GPU Optimizer shutting down...")

if __name__ == "__main__":
    main()

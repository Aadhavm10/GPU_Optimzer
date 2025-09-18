# Screenshots Directory

This directory contains screenshots for the GPU Utilization Optimizer project.

## Required Screenshots

### 1. Dashboard Overview (`dashboard-overview.png`)
- **What to capture**: Full web dashboard showing all metrics
- **How to get**: Run `python simple_dashboard.py`, open http://localhost:8050, take full browser screenshot
- **Should show**: GPU utilization, temperature, memory usage, power consumption, clock speeds

### 2. CUDA Performance Demo (`cuda-performance-demo.png`)
- **What to capture**: Terminal output showing performance comparisons
- **How to get**: Run `python simple_cuda_demo.py`, capture terminal output
- **Should show**: CPU vs GPU performance comparisons, speedup calculations

### 3. Command Line Monitoring (`command-line-monitor.png`)
- **What to capture**: Terminal output of GPU monitoring
- **How to get**: Run `python simple_gpu_monitor.py`, capture terminal output
- **Should show**: Real-time GPU metrics in text format

## Optional Screenshots

### 4. Installation Process (`installation.png`)
- **What to capture**: Setup script running successfully
- **How to get**: Run `.\setup_windows.ps1`, capture the successful completion

### 5. GPU Load Test (`gpu-load-test.png`)
- **What to capture**: Dashboard showing increased GPU activity
- **How to get**: Run `python gpu_load_test.py` in one terminal, dashboard in another, capture both

## Screenshot Guidelines

- **Resolution**: Use at least 1920x1080 for clear readability
- **Format**: PNG preferred for sharp text, JPEG acceptable for photos
- **File size**: Keep under 2MB per image
- **Annotations**: Add arrows or highlights to important features if needed
- **Consistency**: Use similar browser/terminal styling across screenshots

## How to Take Screenshots

### Windows:
1. Use **Snipping Tool** or **Win + Shift + S**
2. Capture full window or custom area
3. Save as PNG for best quality

### Browser Screenshots:
1. **Chrome**: F12 → Device toolbar → Responsive → Capture screenshot
2. **Firefox**: F12 → Responsive Design Mode → Screenshot
3. **Edge**: F12 → Toggle device toolbar → Screenshot

### Terminal Screenshots:
1. **Windows Terminal**: Right-click → Copy → Paste to image editor
2. **PowerShell**: Use `Get-Clipboard` and save as image
3. **CMD**: Use third-party tools like Greenshot

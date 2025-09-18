#!/usr/bin/env python3
"""
Simple GPU Dashboard - Minimal version without pandas dependency
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta
import threading
import time
import collections
try:
    import nvidia_ml_py as pynvml
except ImportError:
    import pynvml

# Global data storage (using collections.deque for efficient circular buffer)
MAX_DATA_POINTS = 300  # 5 minutes at 1-second intervals
metrics_data = {
    'timestamps': collections.deque(maxlen=MAX_DATA_POINTS),
    'gpu_util': collections.deque(maxlen=MAX_DATA_POINTS),
    'memory_util': collections.deque(maxlen=MAX_DATA_POINTS),
    'temperature': collections.deque(maxlen=MAX_DATA_POINTS),
    'power': collections.deque(maxlen=MAX_DATA_POINTS),
    'memory_used': collections.deque(maxlen=MAX_DATA_POINTS),
    'memory_total': 0
}

gpu_handle = None
gpu_info = {}

def initialize_gpu():
    """Initialize GPU monitoring"""
    global gpu_handle, gpu_info
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
        
        # Get GPU info
        name = pynvml.nvmlDeviceGetName(gpu_handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        
        gpu_info = {
            'name': name,
            'memory_total_mb': mem_info.total // (1024**2)
        }
        
        metrics_data['memory_total'] = gpu_info['memory_total_mb']
        
        return True
    except Exception as e:
        print(f"GPU initialization error: {e}")
        return False

def collect_metrics():
    """Background thread to collect GPU metrics"""
    global gpu_handle, metrics_data
    
    while True:
        try:
            if gpu_handle is None:
                time.sleep(1)
                continue
                
            # Collect metrics
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) // 1000
            except:
                power = 0
            
            # Add to circular buffers
            now = datetime.now()
            metrics_data['timestamps'].append(now)
            metrics_data['gpu_util'].append(util.gpu)
            metrics_data['memory_util'].append(util.memory)
            metrics_data['temperature'].append(temp)
            metrics_data['power'].append(power)
            metrics_data['memory_used'].append(mem_info.used // (1024**2))
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
        
        time.sleep(1)  # Collect every second

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "🖥️ GPU Optimizer Dashboard"

# Layout
app.layout = html.Div([
    html.Div([
        html.H1(" GPU Utilization Optimizer", 
                style={'textAlign': 'center', 'color': '#2E86AB', 'margin': '20px'}),
        html.H3(id="gpu-name", 
                style={'textAlign': 'center', 'color': '#A23B72', 'margin': '10px'}),
    ]),
    
    # Current metrics row
    html.Div([
        html.Div([
            html.H4("GPU", style={'textAlign': 'center', 'margin': '5px'}),
            html.H2(id="gpu-util-display", 
                   style={'textAlign': 'center', 'color': '#F18F01', 'margin': '10px'})
        ], style={'width': '24%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 
                 'margin': '0.5%', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #dee2e6'}),
        
        html.Div([
            html.H4("Memory", style={'textAlign': 'center', 'margin': '5px'}),
            html.H2(id="memory-display", 
                   style={'textAlign': 'center', 'color': '#C73E1D', 'margin': '10px'})
        ], style={'width': '24%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 
                 'margin': '0.5%', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #dee2e6'}),
        
        html.Div([
            html.H4("Temperature", style={'textAlign': 'center', 'margin': '5px'}),
            html.H2(id="temp-display", 
                   style={'textAlign': 'center', 'color': '#2E86AB', 'margin': '10px'})
        ], style={'width': '24%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 
                 'margin': '0.5%', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #dee2e6'}),
        
        html.Div([
            html.H4("Power", style={'textAlign': 'center', 'margin': '5px'}),
            html.H2(id="power-display", 
                   style={'textAlign': 'center', 'color': '#A23B72', 'margin': '10px'})
        ], style={'width': '24%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 
                 'margin': '0.5%', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #dee2e6'}),
    ], style={'margin': '20px 0'}),
    
    # Charts
    html.Div([
        dcc.Graph(id="utilization-chart", style={'height': '400px'})
    ], style={'margin': '20px'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id="temperature-chart", style={'height': '300px'})
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id="memory-chart", style={'height': '300px'})
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    ),
    
    html.Footer([
        html.P(" Real-time GPU monitoring dashboard - Press F5 to refresh", 
               style={'textAlign': 'center', 'margin': '20px', 'color': '#6c757d'})
    ])
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# Callbacks
@app.callback(
    Output('gpu-name', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_gpu_name(n):
    if gpu_info:
        return f" {gpu_info['name']} ({gpu_info['memory_total_mb']:,} MB)"
    return "Detecting GPU..."

@app.callback(
    [Output('gpu-util-display', 'children'),
     Output('memory-display', 'children'),
     Output('temp-display', 'children'),
     Output('power-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_current_metrics(n):
    if not metrics_data['timestamps']:
        return "N/A", "N/A", "N/A", "N/A"
    
    # Get latest values
    gpu_util = metrics_data['gpu_util'][-1] if metrics_data['gpu_util'] else 0
    memory_used = metrics_data['memory_used'][-1] if metrics_data['memory_used'] else 0
    memory_util = metrics_data['memory_util'][-1] if metrics_data['memory_util'] else 0
    temp = metrics_data['temperature'][-1] if metrics_data['temperature'] else 0
    power = metrics_data['power'][-1] if metrics_data['power'] else 0
    
    return (
        f"{gpu_util}%",
        f"{memory_used:,} MB ({memory_util}%)",
        f"{temp}°C",
        f"{power} W"
    )

@app.callback(
    Output('utilization-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_utilization_chart(n):
    if not metrics_data['timestamps']:
        return go.Figure().add_annotation(text="🔄 Collecting data...", x=0.5, y=0.5, showarrow=False)
    
    timestamps = list(metrics_data['timestamps'])
    gpu_utils = list(metrics_data['gpu_util'])
    memory_utils = list(metrics_data['memory_util'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=gpu_utils, 
                            mode='lines', name='GPU Utilization (%)', 
                            line=dict(color='#F18F01', width=3)))
    fig.add_trace(go.Scatter(x=timestamps, y=memory_utils, 
                            mode='lines', name='Memory Utilization (%)', 
                            line=dict(color='#C73E1D', width=3)))
    
    fig.update_layout(
        title=' GPU & Memory Utilization',
        xaxis_title='Time', 
        yaxis_title='Utilization (%)',
        yaxis=dict(range=[0, 100]),
        legend=dict(x=0.7, y=0.95),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    return fig

@app.callback(
    Output('temperature-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_temperature_chart(n):
    if not metrics_data['timestamps']:
        return go.Figure().add_annotation(text="🔄 Loading...", x=0.5, y=0.5, showarrow=False)
    
    timestamps = list(metrics_data['timestamps'])
    temperatures = list(metrics_data['temperature'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=temperatures, 
                            mode='lines', name='Temperature', 
                            line=dict(color='#2E86AB', width=3)))
    
    fig.update_layout(
        title=' GPU Temperature',
        xaxis_title='Time', 
        yaxis_title='Temperature (°C)',
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    return fig

@app.callback(
    Output('memory-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_memory_chart(n):
    if not metrics_data['timestamps']:
        return go.Figure().add_annotation(text="🔄 Loading...", x=0.5, y=0.5, showarrow=False)
    
    timestamps = list(metrics_data['timestamps'])
    memory_used = list(metrics_data['memory_used'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=memory_used, 
                            mode='lines', name='Memory Used (MB)', 
                            line=dict(color='#A23B72', width=3)))
    
    # Add total memory line
    if metrics_data['memory_total'] > 0:
        fig.add_hline(y=metrics_data['memory_total'], 
                     line_dash="dash", line_color="red",
                     annotation_text=f"Total: {metrics_data['memory_total']:,} MB")
    
    fig.update_layout(
        title='GPU Memory Usage',
        xaxis_title='Time', 
        yaxis_title='Memory (MB)',
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    return fig

if __name__ == '__main__':
    print("GPU Utilization Optimizer Dashboard")
    print("=" * 50)
    
    # Initialize GPU
    if not initialize_gpu():
        print("Failed to initialize GPU monitoring")
        print("Make sure you have NVIDIA drivers and pynvml installed")
        exit(1)
    
    print(f"GPU detected: {gpu_info['name']}")
    print(f"Memory: {gpu_info['memory_total_mb']:,} MB")
    
    # Start background metrics collection
    metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
    metrics_thread.start()
    print("Background monitoring started")
    
    print("\n🌐 Starting web dashboard...")
    print("URL: http://localhost:8050")
    print("Updates every 2 seconds")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Run dashboard
        app.run(debug=False, host='localhost', port=8050)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"\nDashboard error: {e}")

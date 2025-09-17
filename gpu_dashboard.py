#!/usr/bin/env python3
"""
GPU Dashboard - Web-based real-time GPU monitoring
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import queue
try:
    import nvidia_ml_py as pynvml
except ImportError:
    import pynvml

# Global data storage
metrics_queue = queue.Queue(maxsize=1000)
gpu_handle = None

def initialize_gpu():
    """Initialize GPU monitoring"""
    global gpu_handle
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
        return True
    except:
        return False

def collect_metrics():
    """Background thread to collect GPU metrics"""
    global gpu_handle, metrics_queue
    
    while True:
        try:
            if gpu_handle is None:
                time.sleep(1)
                continue
                
            # Collect metrics
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) // 1000
            
            metrics = {
                'timestamp': datetime.now(),
                'gpu_util': util.gpu,
                'memory_util': util.memory,
                'memory_used_mb': mem_info.used // (1024**2),
                'memory_total_mb': mem_info.total // (1024**2),
                'temperature': temp,
                'power': power
            }
            
            # Add to queue (remove old if full)
            if metrics_queue.full():
                try:
                    metrics_queue.get_nowait()
                except:
                    pass
            
            metrics_queue.put(metrics)
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
        
        time.sleep(1)  # Collect every second

def get_recent_metrics(minutes=5):
    """Get recent metrics as DataFrame"""
    metrics_list = []
    temp_queue = queue.Queue()
    
    # Extract all metrics from queue
    while not metrics_queue.empty():
        try:
            metric = metrics_queue.get_nowait()
            metrics_list.append(metric)
            temp_queue.put(metric)
        except:
            break
    
    # Put metrics back
    while not temp_queue.empty():
        try:
            metrics_queue.put(temp_queue.get_nowait())
        except:
            break
    
    if not metrics_list:
        return pd.DataFrame()
    
    # Filter to recent data
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent_metrics = [m for m in metrics_list if m['timestamp'] > cutoff_time]
    
    return pd.DataFrame(recent_metrics)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "GPU Utilization Optimizer Dashboard"

# Layout
app.layout = html.Div([
    html.H1("🖥️ GPU Utilization Optimizer", style={'textAlign': 'center', 'color': '#2E86AB'}),
    html.H3("Real-time GPU Performance Dashboard", style={'textAlign': 'center', 'color': '#A23B72'}),
    
    # Current metrics cards
    html.Div([
        html.Div([
            html.H4("GPU Utilization", style={'textAlign': 'center'}),
            html.H2(id="gpu-util-display", style={'textAlign': 'center', 'color': '#F18F01'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4("Memory Usage", style={'textAlign': 'center'}),
            html.H2(id="memory-display", style={'textAlign': 'center', 'color': '#C73E1D'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4("Temperature", style={'textAlign': 'center'}),
            html.H2(id="temp-display", style={'textAlign': 'center', 'color': '#2E86AB'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '10px'}),
        
        html.Div([
            html.H4("Power Draw", style={'textAlign': 'center'}),
            html.H2(id="power-display", style={'textAlign': 'center', 'color': '#A23B72'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '10px'}),
    ]),
    
    # Charts
    html.Div([
        html.Div([
            dcc.Graph(id="utilization-chart")
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id="temperature-chart")
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(id="memory-chart")
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id="power-chart")
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    ),
    
    html.Footer([
        html.P("GPU Utilization Optimizer Dashboard - Real-time monitoring", 
               style={'textAlign': 'center', 'marginTop': '20px', 'color': '#666'})
    ])
])

# Callbacks
@app.callback(
    [Output('gpu-util-display', 'children'),
     Output('memory-display', 'children'),
     Output('temp-display', 'children'),
     Output('power-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_current_metrics(n):
    df = get_recent_metrics(1)  # Last minute
    
    if df.empty:
        return "N/A", "N/A", "N/A", "N/A"
    
    latest = df.iloc[-1]
    return (
        f"{latest['gpu_util']}%",
        f"{latest['memory_used_mb']:,} MB ({latest['memory_util']}%)",
        f"{latest['temperature']}°C",
        f"{latest['power']} W"
    )

@app.callback(
    Output('utilization-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_utilization_chart(n):
    df = get_recent_metrics()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['gpu_util'], 
                            mode='lines', name='GPU Utilization', line=dict(color='#F18F01')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['memory_util'], 
                            mode='lines', name='Memory Utilization', line=dict(color='#C73E1D')))
    
    fig.update_layout(title='GPU & Memory Utilization (%)', 
                     xaxis_title='Time', yaxis_title='Utilization (%)',
                     yaxis=dict(range=[0, 100]))
    return fig

@app.callback(
    Output('temperature-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_temperature_chart(n):
    df = get_recent_metrics()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'], 
                            mode='lines', name='Temperature', line=dict(color='#2E86AB')))
    
    fig.update_layout(title='GPU Temperature (°C)', 
                     xaxis_title='Time', yaxis_title='Temperature (°C)')
    return fig

@app.callback(
    Output('memory-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_memory_chart(n):
    df = get_recent_metrics()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['memory_used_mb'], 
                            mode='lines', name='Memory Used', line=dict(color='#A23B72')))
    
    fig.update_layout(title='GPU Memory Usage (MB)', 
                     xaxis_title='Time', yaxis_title='Memory (MB)')
    return fig

@app.callback(
    Output('power-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_power_chart(n):
    df = get_recent_metrics()
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['power'], 
                            mode='lines', name='Power Draw', line=dict(color='#F18F01')))
    
    fig.update_layout(title='GPU Power Draw (W)', 
                     xaxis_title='Time', yaxis_title='Power (W)')
    return fig

if __name__ == '__main__':
    # Initialize GPU
    if not initialize_gpu():
        print("❌ Failed to initialize GPU monitoring")
        exit(1)
    
    print("✅ GPU monitoring initialized")
    
    # Start background metrics collection
    metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
    metrics_thread.start()
    
    print("🚀 Starting dashboard on http://localhost:8050")
    print("Press Ctrl+C to stop")
    
    # Run dashboard
    app.run_server(debug=False, host='localhost', port=8050)




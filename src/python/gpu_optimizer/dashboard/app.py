"""
Main dashboard application using Dash/Plotly for real-time GPU monitoring.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
from typing import Optional, Dict, Any, List

from ..config import Config
from .components import (
    GPUUtilizationChart,
    MemoryUsageChart,
    TemperatureChart,
    PowerChart,
    KernelComparisonChart
)
from .data_source import LiveDataSource

logger = logging.getLogger(__name__)

class DashboardApp:
    """Main dashboard application class."""
    
    def __init__(self, config: Config = None, data_source: LiveDataSource = None):
        self.config = config or Config()
        self.data_source = data_source or LiveDataSource(self.config)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ])
        
        self.app.title = "GPU Utilization Optimizer"
        
        # Initialize components
        self.gpu_util_chart = GPUUtilizationChart()
        self.memory_chart = MemoryUsageChart()
        self.temp_chart = TemperatureChart()
        self.power_chart = PowerChart()
        self.kernel_chart = KernelComparisonChart()
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        # Start data collection
        self.data_source.start()
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1([
                    html.I(className="fas fa-microchip", style={'margin-right': '10px'}),
                    "GPU Utilization Optimizer"
                ], className="header-title"),
                html.Div([
                    html.Span("Real-time GPU Performance Monitoring", className="header-subtitle"),
                    html.Div(id="last-update", className="last-update")
                ])
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("GPU Selection:"),
                    dcc.Dropdown(
                        id="gpu-selector",
                        options=[{"label": f"GPU {i}", "value": i} for i in range(4)],
                        value=0,
                        clearable=False
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id="time-range",
                        options=[
                            {"label": "Last 1 minute", "value": 1},
                            {"label": "Last 5 minutes", "value": 5},
                            {"label": "Last 15 minutes", "value": 15},
                            {"label": "Last 1 hour", "value": 60},
                        ],
                        value=5,
                        clearable=False
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Button([
                        html.I(className="fas fa-play"),
                        " Start Monitoring"
                    ], id="start-btn", className="btn btn-primary"),
                    html.Button([
                        html.I(className="fas fa-pause"),
                        " Pause"
                    ], id="pause-btn", className="btn btn-secondary"),
                    html.Button([
                        html.I(className="fas fa-download"),
                        " Export"
                    ], id="export-btn", className="btn btn-success")
                ], className="control-buttons")
            ], className="control-panel"),
            
            # Main metrics row
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("GPU Utilization"),
                        html.Div(id="gpu-util-value", className="metric-value"),
                        html.Div("Current GPU compute usage", className="metric-description")
                    ], className="metric-card")
                ], className="metric-col"),
                
                html.Div([
                    html.Div([
                        html.H3("Memory Usage"),
                        html.Div(id="memory-value", className="metric-value"),
                        html.Div("GPU memory utilization", className="metric-description")
                    ], className="metric-card")
                ], className="metric-col"),
                
                html.Div([
                    html.Div([
                        html.H3("Temperature"),
                        html.Div(id="temp-value", className="metric-value"),
                        html.Div("GPU core temperature", className="metric-description")
                    ], className="metric-card")
                ], className="metric-col"),
                
                html.Div([
                    html.Div([
                        html.H3("Power Draw"),
                        html.Div(id="power-value", className="metric-value"),
                        html.Div("Current power consumption", className="metric-description")
                    ], className="metric-card")
                ], className="metric-col")
            ], className="metrics-row"),
            
            # Charts row
            html.Div([
                html.Div([
                    html.H3("GPU Utilization Over Time"),
                    dcc.Graph(id="gpu-utilization-chart", config={'displayModeBar': False})
                ], className="chart-container"),
                
                html.Div([
                    html.H3("Memory Usage"),
                    dcc.Graph(id="memory-usage-chart", config={'displayModeBar': False})
                ], className="chart-container")
            ], className="charts-row"),
            
            html.Div([
                html.Div([
                    html.H3("Temperature Monitoring"),
                    dcc.Graph(id="temperature-chart", config={'displayModeBar': False})
                ], className="chart-container"),
                
                html.Div([
                    html.H3("Power Consumption"),
                    dcc.Graph(id="power-chart", config={'displayModeBar': False})
                ], className="chart-container")
            ], className="charts-row"),
            
            # Kernel comparison section
            html.Div([
                html.H2("Kernel Performance Comparison"),
                html.Div([
                    html.Div([
                        html.Label("Matrix Size:"),
                        dcc.Slider(
                            id="matrix-size-slider",
                            min=6,  # 2^6 = 64
                            max=13,  # 2^13 = 8192
                            step=1,
                            value=10,  # 2^10 = 1024
                            marks={i: f"{2**i}" for i in range(6, 14)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="slider-container"),
                    
                    html.Div([
                        html.Button("Run Benchmark", id="benchmark-btn", className="btn btn-primary"),
                        html.Div(id="benchmark-status", className="benchmark-status")
                    ], className="benchmark-controls")
                ]),
                
                dcc.Graph(id="kernel-comparison-chart")
            ], className="benchmark-section"),
            
            # Auto-refresh component
            dcc.Interval(
                id="interval-component",
                interval=1000,  # Update every second
                n_intervals=0
            ),
            
            # Store components for data
            dcc.Store(id="gpu-data-store"),
            dcc.Store(id="monitoring-state", data={"active": True})
            
        ], className="dashboard-container")
    
    def _setup_callbacks(self):
        """Setup all dashboard callbacks."""
        
        @self.app.callback(
            [Output("gpu-util-value", "children"),
             Output("memory-value", "children"),
             Output("temp-value", "children"),
             Output("power-value", "children"),
             Output("last-update", "children")],
            [Input("interval-component", "n_intervals"),
             Input("gpu-selector", "value")],
            [State("monitoring-state", "data")]
        )
        def update_metrics(n_intervals, gpu_id, monitoring_state):
            if not monitoring_state.get("active", True):
                return dash.no_update
            
            try:
                metrics = self.data_source.get_latest_metrics(gpu_id)
                if metrics:
                    return (
                        f"{metrics.utilization.gpu_utilization_percentage}%",
                        f"{metrics.memory.usage_percentage:.1f}%",
                        f"{metrics.thermal.gpu_temperature_c}°C",
                        f"{metrics.power.power_draw_watts:.1f}W",
                        f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                    )
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
            
            return "N/A", "N/A", "N/A", "N/A", "Connection lost"
        
        @self.app.callback(
            [Output("gpu-utilization-chart", "figure"),
             Output("memory-usage-chart", "figure"),
             Output("temperature-chart", "figure"),
             Output("power-chart", "figure")],
            [Input("interval-component", "n_intervals"),
             Input("gpu-selector", "value"),
             Input("time-range", "value")],
            [State("monitoring-state", "data")]
        )
        def update_charts(n_intervals, gpu_id, time_range_minutes, monitoring_state):
            if not monitoring_state.get("active", True):
                return dash.no_update
            
            try:
                # Get historical data
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=time_range_minutes)
                data = self.data_source.get_historical_data(gpu_id, start_time, end_time)
                
                if not data:
                    # Return empty charts
                    empty_fig = go.Figure()
                    empty_fig.update_layout(title="No data available")
                    return empty_fig, empty_fig, empty_fig, empty_fig
                
                # Convert to DataFrame for easier plotting
                df = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'gpu_util': d.utilization.gpu_utilization_percentage,
                    'memory_util': d.utilization.memory_utilization_percentage,
                    'memory_used': d.memory.used_memory_mb,
                    'memory_total': d.memory.total_memory_mb,
                    'temperature': d.thermal.gpu_temperature_c,
                    'power': d.power.power_draw_watts
                } for d in data])
                
                # Create charts
                gpu_util_fig = self.gpu_util_chart.create_figure(df)
                memory_fig = self.memory_chart.create_figure(df)
                temp_fig = self.temp_chart.create_figure(df)
                power_fig = self.power_chart.create_figure(df)
                
                return gpu_util_fig, memory_fig, temp_fig, power_fig
                
            except Exception as e:
                logger.error(f"Error updating charts: {e}")
                empty_fig = go.Figure()
                empty_fig.update_layout(title=f"Error: {str(e)}")
                return empty_fig, empty_fig, empty_fig, empty_fig
        
        @self.app.callback(
            [Output("monitoring-state", "data"),
             Output("start-btn", "children"),
             Output("pause-btn", "disabled")],
            [Input("start-btn", "n_clicks"),
             Input("pause-btn", "n_clicks")],
            [State("monitoring-state", "data")]
        )
        def control_monitoring(start_clicks, pause_clicks, current_state):
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_state, dash.no_update, False
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-btn":
                self.data_source.resume()
                return {"active": True}, [html.I(className="fas fa-pause"), " Pause"], False
            elif button_id == "pause-btn":
                self.data_source.pause()
                return {"active": False}, [html.I(className="fas fa-play"), " Resume"], True
            
            return current_state, dash.no_update, False
        
        @self.app.callback(
            [Output("kernel-comparison-chart", "figure"),
             Output("benchmark-status", "children")],
            [Input("benchmark-btn", "n_clicks")],
            [State("matrix-size-slider", "value")]
        )
        def run_kernel_benchmark(n_clicks, matrix_size_exp):
            if not n_clicks:
                return go.Figure(), ""
            
            matrix_size = 2 ** matrix_size_exp
            
            try:
                # This would call the CUDA profiler
                # For now, return mock data
                kernels = ["Naive", "Tiled", "Shared Memory", "Vectorized", "Double Buffered"]
                performance = [15.2, 45.6, 67.8, 78.4, 85.2]  # Mock GFLOPS
                
                fig = self.kernel_chart.create_comparison_figure(kernels, performance, matrix_size)
                status = f"Benchmark completed for {matrix_size}x{matrix_size} matrices"
                
                return fig, status
                
            except Exception as e:
                logger.error(f"Benchmark error: {e}")
                return go.Figure(), f"Error: {str(e)}"
    
    def run(self, host: str = "localhost", port: int = 8080, debug: bool = False):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)
    
    def stop(self):
        """Stop the dashboard and data collection."""
        self.data_source.stop()

def create_dashboard(config: Config = None, **kwargs) -> DashboardApp:
    """
    Factory function to create a dashboard instance.
    
    Args:
        config: Configuration object
        **kwargs: Additional configuration options
        
    Returns:
        DashboardApp instance
    """
    if config is None:
        config = Config()
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DashboardApp(config)

def main():
    """Main entry point for the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Utilization Optimizer Dashboard")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.config:
        config.load_from_file(args.config)
    
    # Create and run dashboard
    app = create_dashboard(config)
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")
    finally:
        app.stop()

if __name__ == "__main__":
    main()

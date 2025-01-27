"""Visualization module for pipeline performance monitoring."""

from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class PipelineVisualizerMixin:
    """Mixin class for pipeline performance visualization."""
    
    def plot_pipeline_performance(
        self,
        time_window: Optional[int] = None,  # days
        model_filter: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
        interactive: bool = True
    ) -> go.Figure:
        """Plot pipeline performance metrics.
        
        Parameters
        ----------
        time_window : Optional[int]
            Number of days to include in visualization
        model_filter : Optional[List[str]]
            List of model names to include in visualization
        min_quality : Optional[float]
            Minimum quality score to include
        interactive : bool
            Whether to add interactive features
            
        Returns
        -------
        go.Figure
            Interactive figure with performance visualizations
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Execution Time by Model',
                'Quality Metrics Distribution',
                'Token Usage Distribution',
                'Model Selection Distribution',
                'Performance Trends',
                'Quality-Speed Tradeoff',
                'Quality Metrics Radar',
                'Quality Metrics Timeline'
            ),
            specs=[
                [{"type": "box"}, {"type": "violin"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatterpolar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12
        )
        
        # Process metrics data
        metrics_data = self._process_metrics_data(
            time_window=time_window,
            model_filter=model_filter,
            min_quality=min_quality
        )
        
        if not metrics_data:
            return self._create_empty_figure()
            
        # Add standard plots
        self._add_execution_time_plot(fig, metrics_data)
        self._add_quality_metrics_plot(fig, metrics_data)
        self._add_token_usage_plot(fig, metrics_data)
        self._add_model_selection_plot(fig, metrics_data)
        self._add_performance_trends(fig, metrics_data)
        self._add_tradeoff_plot(fig, metrics_data)
        
        # Add new plots
        self._add_quality_radar_plot(fig, metrics_data)
        self._add_quality_timeline_plot(fig, metrics_data)
        
        # Update layout
        fig.update_layout(
            height=1400,
            width=1200,
            showlegend=True,
            title_text="Pipeline Performance Analysis",
            title_x=0.5,
            template="plotly_white",
            hovermode='closest'
        )
        
        if interactive:
            self._add_interactive_features(fig)
        
        return fig
        
    def plot_quality_details(
        self,
        model_name: str,
        time_window: Optional[int] = None
    ) -> go.Figure:
        """Plot detailed quality metrics for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to analyze
        time_window : Optional[int]
            Number of days to include
            
        Returns
        -------
        go.Figure
            Detailed quality metrics visualization
        """
        metrics_data = self._process_metrics_data(
            time_window=time_window,
            model_filter=[model_name]
        )
        
        if not metrics_data or model_name not in metrics_data:
            return self._create_empty_figure()
            
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Quality Metrics Distribution',
                'Quality Metrics Correlation',
                'Quality vs Complexity',
                'Quality vs Token Usage',
                'Quality Metrics Timeline',
                'Quality Components Analysis'
            ),
            specs=[
                [{"type": "violin"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        data = metrics_data[model_name]
        
        # Add quality metrics distribution
        metrics = ['comprehensiveness', 'coherence', 'insight_depth', 'relevance', 'clarity']
        for metric in metrics:
            values = [d['metrics']['quality_metrics'][metric] for d in data]
            fig.add_trace(
                go.Violin(
                    y=values,
                    name=metric,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=1, col=1
            )
            
        # Add correlation heatmap
        corr_matrix = self._compute_quality_correlations(data)
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=metrics,
                y=metrics,
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=2
        )
        
        # Add quality vs complexity scatter
        complexities = [d['context']['complexity'] for d in data]
        quality_scores = [d['metrics']['quality_metrics']['overall_score'] for d in data]
        fig.add_trace(
            go.Scatter(
                x=complexities,
                y=quality_scores,
                mode='markers',
                name='Quality vs Complexity',
                marker=dict(
                    size=8,
                    color=quality_scores,
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=2, col=1
        )
        
        # Add quality vs token usage
        token_counts = [d['metrics']['token_count'] for d in data]
        fig.add_trace(
            go.Scatter(
                x=token_counts,
                y=quality_scores,
                mode='markers',
                name='Quality vs Tokens',
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        # Add quality metrics timeline
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        for metric in metrics:
            values = [d['metrics']['quality_metrics'][metric] for d in data]
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=values,
                    name=metric,
                    mode='lines+markers'
                ),
                row=3, col=1
            )
            
        # Add quality components analysis
        avg_metrics = {
            metric: np.mean([d['metrics']['quality_metrics'][metric] for d in data])
            for metric in metrics
        }
        
        fig.add_trace(
            go.Bar(
                x=list(avg_metrics.keys()),
                y=list(avg_metrics.values()),
                name='Average Quality Components'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            showlegend=True,
            title_text=f"Detailed Quality Analysis - {model_name}",
            title_x=0.5,
            template="plotly_white",
            hovermode='closest'
        )
        
        return fig
        
    def _process_metrics_data(
        self,
        time_window: Optional[int],
        model_filter: Optional[List[str]] = None,
        min_quality: Optional[float] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Process and filter metrics data."""
        processed_data = {}
        
        for model, metrics in self.performance_metrics.items():
            if not metrics:
                continue
                
            if model_filter and model not in model_filter:
                continue
                
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Apply time window filter if specified
            if time_window:
                cutoff = datetime.now() - timedelta(days=time_window)
                df = df[df['timestamp'] >= cutoff]
                
            # Apply quality filter if specified
            if min_quality is not None:
                df = df[df['metrics'].apply(
                    lambda x: x['quality_metrics']['overall_score'] >= min_quality
                )]
                
            if not df.empty:
                processed_data[model] = df.to_dict('records')
                
        return processed_data
        
    def _add_execution_time_plot(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add execution time box plots."""
        for model, data in metrics_data.items():
            exec_times = [d['metrics']['execution_time'] for d in data]
            fig.add_trace(
                go.Box(
                    y=exec_times,
                    name=model,
                    boxpoints='outliers'
                ),
                row=1, col=1
            )
            
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Execution Time (s)", row=1, col=1)
        
    def _add_quality_metrics_plot(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add quality metrics violin plots."""
        for model, data in metrics_data.items():
            metrics = ['comprehensiveness', 'coherence', 'insight_depth', 'relevance', 'clarity']
            for metric in metrics:
                values = [d['metrics']['quality_metrics'][metric] for d in data]
                fig.add_trace(
                    go.Violin(
                        y=values,
                        name=f"{model} - {metric}",
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="Quality Score", row=1, col=2)
        
    def _add_token_usage_plot(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add token usage distribution."""
        models = list(metrics_data.keys())
        avg_tokens = []
        
        for model in models:
            token_counts = [d['metrics']['token_count'] for d in metrics_data[model]]
            avg_tokens.append(np.mean(token_counts))
            
        fig.add_trace(
            go.Bar(
                x=models,
                y=avg_tokens,
                name="Average Token Usage"
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_yaxes(title_text="Average Token Count", row=2, col=1)
        
    def _add_model_selection_plot(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add model selection distribution pie chart."""
        model_counts = {model: len(data) for model, data in metrics_data.items()}
        
        fig.add_trace(
            go.Pie(
                labels=list(model_counts.keys()),
                values=list(model_counts.values()),
                name="Model Selection"
            ),
            row=2, col=2
        )
        
    def _add_performance_trends(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add performance trends over time."""
        for model, data in metrics_data.items():
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate rolling averages
            window = min(10, len(df))
            quality_ma = df['metrics'].apply(lambda x: x['quality_score']).rolling(window).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=quality_ma,
                    name=f"{model} Quality Trend",
                    mode='lines'
                ),
                row=3, col=1
            )
            
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Rolling Average Quality", row=3, col=1)
        
    def _add_tradeoff_plot(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add quality-speed tradeoff scatter plot."""
        for model, data in metrics_data.items():
            exec_times = [d['metrics']['execution_time'] for d in data]
            quality_scores = [d['metrics']['quality_score'] for d in data]
            
            fig.add_trace(
                go.Scatter(
                    x=exec_times,
                    y=quality_scores,
                    name=model,
                    mode='markers',
                    marker=dict(size=8)
                ),
                row=3, col=2
            )
            
        fig.update_xaxes(title_text="Execution Time (s)", row=3, col=2)
        fig.update_yaxes(title_text="Quality Score", row=3, col=2)
        
    def _add_quality_radar_plot(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add quality metrics radar plot."""
        metrics = ['comprehensiveness', 'coherence', 'insight_depth', 'relevance', 'clarity']
        
        for model, data in metrics_data.items():
            avg_metrics = {
                metric: np.mean([d['metrics']['quality_metrics'][metric] for d in data])
                for metric in metrics
            }
            
            fig.add_trace(
                go.Scatterpolar(
                    r=list(avg_metrics.values()),
                    theta=list(avg_metrics.keys()),
                    fill='toself',
                    name=model
                ),
                row=4, col=1
            )
            
    def _add_quality_timeline_plot(
        self,
        fig: go.Figure,
        metrics_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add quality metrics timeline plot."""
        for model, data in metrics_data.items():
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            metrics = ['comprehensiveness', 'coherence', 'insight_depth', 'relevance', 'clarity']
            for metric in metrics:
                values = [d['metrics']['quality_metrics'][metric] for d in data]
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=values,
                        name=f"{model} - {metric}",
                        mode='lines',
                        opacity=0.7
                    ),
                    row=4, col=2
                )
                
        fig.update_xaxes(title_text="Time", row=4, col=2)
        fig.update_yaxes(title_text="Quality Score", row=4, col=2)
        
    def _compute_quality_correlations(
        self,
        data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Compute correlation matrix for quality metrics."""
        metrics = ['comprehensiveness', 'coherence', 'insight_depth', 'relevance', 'clarity']
        values = np.array([
            [d['metrics']['quality_metrics'][metric] for metric in metrics]
            for d in data
        ])
        return np.corrcoef(values.T)
        
    def _add_interactive_features(self, fig: go.Figure) -> None:
        """Add interactive features to the figure."""
        # Add range slider for timeline plots
        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
        fig.update_xaxes(rangeslider_visible=True, row=4, col=2)
        
        # Add buttons for different time ranges
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="All Time",
                            method="restyle"
                        ),
                        dict(
                            args=[{
                                "visible": [
                                    i < len(fig.data)/2 for i in range(len(fig.data))
                                ]
                            }],
                            label="Recent",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        
        # Add hover templates
        for trace in fig.data:
            if isinstance(trace, go.Scatter):
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Time: %{x}<br>" +
                    "Value: %{y:.3f}<br>" +
                    "<extra></extra>"
                )
                
    def _create_empty_figure(self) -> go.Figure:
        """Create empty figure when no data is available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No performance data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig 
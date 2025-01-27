"""Visualization functions for emotion analysis."""

from typing import Dict, Any, List
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_emotion_alignment(results: Dict[str, Any]) -> None:
    """Plot emotion alignment analysis results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Emotion analysis results
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Emotion Intensities (Audio)',
            'Emotion Intensities (Video)',
            'Emotion Coherence Matrix',
            'Temporal Patterns',
            'Peak Detection',
            'Stability Analysis'
        )
    )
    
    # Plot audio emotions
    audio_emotions = results['raw_emotions']['audio']
    for emotion, intensities in audio_emotions['emotions'].items():
        fig.add_trace(
            go.Scatter(
                x=audio_emotions['timestamps'],
                y=intensities,
                name=f'Audio - {emotion}',
                mode='lines'
            ),
            row=1, col=1
        )
    
    # Plot video emotions
    video_emotions = results['raw_emotions']['video']
    for emotion, intensities in video_emotions['emotions'].items():
        fig.add_trace(
            go.Scatter(
                x=video_emotions['timestamps'],
                y=intensities,
                name=f'Video - {emotion}',
                mode='lines'
            ),
            row=1, col=2
        )
    
    # Plot coherence matrix
    coherence_matrix = np.array([
        [results['emotion_alignment']['correlation'][e]['coefficient']
         for e in results['emotion_alignment']['correlation']]
    ])
    
    fig.add_trace(
        go.Heatmap(
            z=coherence_matrix,
            x=list(results['emotion_alignment']['correlation'].keys()),
            y=['Coherence'],
            colorscale='RdBu',
            zmid=0
        ),
        row=2, col=1
    )
    
    # Plot temporal patterns
    temporal = results['temporal_patterns']
    fig.add_trace(
        go.Scatter(
            x=temporal['timestamps'],
            y=temporal['pattern_strength'],
            name='Pattern Strength',
            mode='lines'
        ),
        row=2, col=2
    )
    
    # Plot peak detection
    for emotion in results['emotion_alignment']['correlation']:
        peaks = temporal['peaks'].get(emotion, [])
        if peaks:
            fig.add_trace(
                go.Scatter(
                    x=[p['index'] for p in peaks],
                    y=[p['intensity'] for p in peaks],
                    name=f'{emotion} peaks',
                    mode='markers'
                ),
                row=3, col=1
            )
    
    # Plot stability analysis
    stability_scores = [
        temporal['stability'].get(emotion, 0)
        for emotion in results['emotion_alignment']['correlation']
    ]
    
    fig.add_trace(
        go.Bar(
            x=list(results['emotion_alignment']['correlation'].keys()),
            y=stability_scores,
            name='Stability'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        title_text='Cross-Modal Emotion Analysis',
        showlegend=True
    )
    
    # Show figure
    fig.show()

def plot_emotion_timeline(
    results: Dict[str, Any],
    emotion: str
) -> None:
    """Plot detailed timeline for specific emotion.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Emotion analysis results
    emotion : str
        Emotion to plot
    """
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'{emotion} Intensity Across Modalities',
            'Coherence Analysis'
        )
    )
    
    # Plot emotion intensities
    audio_intensity = results['raw_emotions']['audio']['emotions'][emotion]
    video_intensity = results['raw_emotions']['video']['emotions'][emotion]
    timestamps = results['raw_emotions']['audio']['timestamps']
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=audio_intensity,
            name='Audio',
            mode='lines'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=video_intensity,
            name='Video',
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Plot coherence metrics
    metrics = {
        'Correlation': results['emotion_alignment']['correlation'][emotion]['coefficient'],
        'Mutual Information': results['emotion_alignment']['mutual_information'][emotion],
        'DTW Distance': results['emotion_alignment']['dtw_distance'][emotion]
    }
    
    fig.add_trace(
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            name='Coherence Metrics'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text=f'Detailed Analysis: {emotion}',
        showlegend=True
    )
    
    # Show figure
    fig.show()

def plot_emotion_changepoints(
    results: Dict[str, Any],
    title: str = "Emotion Change Point Analysis"
) -> go.Figure:
    """Plot emotion change point analysis results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Change point analysis results
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Audio Change Scores", "Video Change Scores",
            "Audio Regime Statistics", "Video Regime Statistics",
            "Audio Emotion States", "Video Emotion States"
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, modality in enumerate(['audio', 'video']):
        # Plot change scores
        fig.add_trace(
            go.Scatter(
                x=results['timestamps'][:-1],
                y=results['change_scores'][modality],
                name=f"{modality.title()} Change Score",
                line=dict(width=2),
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Add changepoint markers
        cp_times = [cp['time'] for cp in results['changepoints'][modality]]
        cp_scores = [cp['score'] for cp in results['changepoints'][modality]]
        
        fig.add_trace(
            go.Scatter(
                x=cp_times,
                y=cp_scores,
                mode='markers',
                name=f"{modality.title()} Changepoints",
                marker=dict(
                    size=10,
                    symbol='diamond',
                    color='red'
                ),
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Plot regime statistics
        regimes = results['regime_statistics'][modality]
        
        fig.add_trace(
            go.Scatter(
                x=[r['start_time'] for r in regimes],
                y=[r['stability'] for r in regimes],
                name="Stability",
                line=dict(width=2),
                showlegend=False
            ),
            row=2, col=i+1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[r['start_time'] for r in regimes],
                y=[r['complexity'] for r in regimes],
                name="Complexity",
                line=dict(width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=i+1
        )
        
        # Plot emotion states
        for cp in results['changepoints'][modality]:
            # Before state
            fig.add_trace(
                go.Bar(
                    x=list(cp['before_state'].keys()),
                    y=list(cp['before_state'].values()),
                    name=f"Before {cp['time']:.1f}s",
                    showlegend=False
                ),
                row=3, col=i+1
            )
            
            # After state
            fig.add_trace(
                go.Bar(
                    x=list(cp['after_state'].keys()),
                    y=list(cp['after_state'].values()),
                    name=f"After {cp['time']:.1f}s",
                    showlegend=False
                ),
                row=3, col=i+1
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=1000,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1)
    fig.update_xaxes(title_text="Time (s)", row=2)
    fig.update_xaxes(title_text="Emotion", row=3)
    
    fig.update_yaxes(title_text="Change Score", row=1)
    fig.update_yaxes(title_text="Metric Value", row=2)
    fig.update_yaxes(title_text="Probability", row=3)
    
    return fig

def plot_emotion_trends(
    results: Dict[str, Any],
    title: str = "Emotion Trend Analysis"
) -> go.Figure:
    """Plot emotion trend analysis results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Trend analysis results
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Audio Trends", "Video Trends",
            "Audio Seasonality", "Video Seasonality",
            "Audio Momentum", "Video Momentum"
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, modality in enumerate(['audio', 'video']):
        # Plot trends
        for emotion in results['trends'][modality].keys():
            fig.add_trace(
                go.Scatter(
                    x=results['timestamps'],
                    y=results['trends'][modality][emotion],
                    name=emotion,
                    line=dict(width=2),
                    showlegend=True if i == 0 else False
                ),
                row=1, col=i+1
            )
        
        # Plot seasonality
        for emotion in results['seasonality'][modality].keys():
            fig.add_trace(
                go.Scatter(
                    x=results['timestamps'],
                    y=results['seasonality'][modality][emotion]['residuals'],
                    name=f"{emotion} (seasonal)",
                    line=dict(width=1, dash='dot'),
                    showlegend=False
                ),
                row=2, col=i+1
            )
        
        # Plot momentum indicators
        for emotion in results['momentum'][modality].keys():
            momentum = results['momentum'][modality][emotion]
            
            # Rate of change
            fig.add_trace(
                go.Scatter(
                    x=results['timestamps'],
                    y=momentum['rate_of_change'],
                    name=f"{emotion} RoC",
                    line=dict(width=1),
                    showlegend=False
                ),
                row=3, col=i+1
            )
            
            # Add trend strength annotation
            fig.add_annotation(
                x=0.1, y=0.9,
                text=f"Trend Strength: {momentum['trend_strength']:.3f}",
                showarrow=False,
                xref=f"x{i+5}",
                yref=f"y{i+5}"
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=1000,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    for row in range(1, 4):
        fig.update_xaxes(title_text="Time (s)", row=row)
    
    fig.update_yaxes(title_text="Emotion Intensity", row=1)
    fig.update_yaxes(title_text="Seasonal Component", row=2)
    fig.update_yaxes(title_text="Rate of Change", row=3)
    
    return fig

def plot_emotion_summary(
    results: Dict[str, Any],
    title: str = "Emotion Analysis Summary"
) -> go.Figure:
    """Plot comprehensive emotion analysis summary.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Combined analysis results
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "Emotion Intensities (Audio)", "Emotion Intensities (Video)",
            "Change Points", "Trends",
            "Cross-Modal Coherence", "Seasonality Patterns",
            "Emotion Transitions", "Momentum Analysis"
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Plot raw emotions
    for i, modality in enumerate(['audio', 'video']):
        for emotion in results['raw_emotions'][modality]['emotions'].keys():
            fig.add_trace(
                go.Scatter(
                    x=results['timestamps'],
                    y=results['raw_emotions'][modality]['emotions'][emotion],
                    name=emotion,
                    line=dict(width=2),
                    showlegend=True if i == 0 else False
                ),
                row=1, col=i+1
            )
    
    # Plot change points
    for modality in ['audio', 'video']:
        fig.add_trace(
            go.Scatter(
                x=results['timestamps'][:-1],
                y=results['changepoints']['change_scores'][modality],
                name=f"{modality.title()} Changes",
                line=dict(width=2)
            ),
            row=2, col=1
        )
    
    # Plot trends
    for modality in ['audio', 'video']:
        for emotion in results['trends']['trends'][modality].keys():
            fig.add_trace(
                go.Scatter(
                    x=results['timestamps'],
                    y=results['trends']['trends'][modality][emotion],
                    name=f"{emotion} ({modality})",
                    line=dict(width=2, dash='dot')
                ),
                row=2, col=2
            )
    
    # Plot coherence
    for emotion in results['synchronization']['coherence'].keys():
        fig.add_trace(
            go.Scatter(
                x=results['timestamps'],
                y=results['synchronization']['coherence'][emotion],
                name=f"{emotion} Coherence",
                line=dict(width=2)
            ),
            row=3, col=1
        )
    
    # Plot seasonality patterns
    for modality in ['audio', 'video']:
        for emotion in results['trends']['seasonality'][modality].keys():
            patterns = results['trends']['seasonality'][modality][emotion]['patterns']
            for pattern in patterns:
                fig.add_trace(
                    go.Scatter(
                        x=[pattern['frequency']],
                        y=[pattern['amplitude']],
                        mode='markers+text',
                        name=f"{emotion} ({modality})",
                        text=[emotion],
                        textposition="top center",
                        marker=dict(size=10)
                    ),
                    row=3, col=2
                )
    
    # Plot transitions
    for modality in ['audio', 'video']:
        transitions = results['context']['emotion_sequences'][modality]
        durations = [t['duration'] for t in transitions]
        emotions = [t['emotion'] for t in transitions]
        
        fig.add_trace(
            go.Bar(
                x=emotions,
                y=durations,
                name=f"{modality.title()} Durations"
            ),
            row=4, col=1
        )
    
    # Plot momentum
    for modality in ['audio', 'video']:
        for emotion in results['trends']['momentum'][modality].keys():
            momentum = results['trends']['momentum'][modality][emotion]
            fig.add_trace(
                go.Scatter(
                    x=results['timestamps'],
                    y=momentum['acceleration'],
                    name=f"{emotion} ({modality})",
                    line=dict(width=1)
                ),
                row=4, col=2
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=1400,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig 
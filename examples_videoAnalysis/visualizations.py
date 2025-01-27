"""Visualization functions for video analysis results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import cv2

def plot_emotion_timeline(
    emotion_data: Dict[str, List[float]],
    output_path: Optional[str] = None
):
    """Plot emotion intensities over time.
    
    Parameters
    ----------
    emotion_data : Dict[str, List[float]]
        Dictionary mapping emotions to intensity values over time
    output_path : Optional[str]
        Path to save plot, if None displays plot
    """
    plt.figure(figsize=(12, 6))
    for emotion, values in emotion_data.items():
        plt.plot(values, label=emotion)
    
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Emotion Intensity Timeline")
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_scene_segmentation(
    scene_data: List[Dict[str, Any]],
    output_path: Optional[str] = None
):
    """Plot scene segmentation results.
    
    Parameters
    ----------
    scene_data : List[Dict[str, Any]]
        List of scene dictionaries with start_time, end_time, and content
    output_path : Optional[str]
        Path to save plot, if None displays plot
    """
    fig = go.Figure()
    
    for i, scene in enumerate(scene_data):
        start = scene['start_time']
        end = scene['end_time']
        content = scene['content'][:50] + '...' if len(scene['content']) > 50 else scene['content']
        
        fig.add_trace(go.Bar(
            x=[end - start],
            y=[i],
            orientation='h',
            name=f'Scene {i+1}',
            text=[content],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title='Video Scene Segmentation',
        xaxis_title='Time (seconds)',
        yaxis_title='Scene Number',
        showlegend=False,
        height=max(400, len(scene_data) * 40)
    )
    
    if output_path:
        fig.write_html(output_path)
    else:
        fig.show()

def plot_action_recognition(
    action_data: Dict[str, List[float]],
    output_path: Optional[str] = None
):
    """Plot action recognition confidence scores.
    
    Parameters
    ----------
    action_data : Dict[str, List[float]]
        Dictionary mapping actions to confidence scores over time
    output_path : Optional[str]
        Path to save plot, if None displays plot
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        np.array([scores for scores in action_data.values()]),
        yticklabels=list(action_data.keys()),
        cmap='YlOrRd',
        cbar_kws={'label': 'Confidence Score'}
    )
    
    plt.xlabel("Frame")
    plt.title("Action Recognition Confidence Scores")
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_object_tracking(
    tracking_data: Dict[str, List[Dict[str, Any]]],
    frame: np.ndarray,
    frame_idx: int,
    output_path: Optional[str] = None
):
    """Plot object tracking results on a frame.
    
    Parameters
    ----------
    tracking_data : Dict[str, List[Dict[str, Any]]]
        Dictionary mapping object IDs to lists of tracking data
    frame : np.ndarray
        Video frame to plot on
    frame_idx : int
        Current frame index
    output_path : Optional[str]
        Path to save plot, if None displays plot
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tracking_data)))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    for (obj_id, tracks), color in zip(tracking_data.items(), colors):
        if frame_idx < len(tracks):
            bbox = tracks[frame_idx]['bbox']
            x, y, w, h = bbox
            rect = plt.Rectangle(
                (x, y), w, h,
                fill=False,
                color=color,
                linewidth=2
            )
            plt.gca().add_patch(rect)
            plt.text(
                x, y-5,
                f'ID: {obj_id}',
                color=color,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )
    
    plt.axis('off')
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_temporal_patterns(
    temporal_data: Dict[str, Any],
    output_path: Optional[str] = None
):
    """Plot temporal pattern analysis results.
    
    Parameters
    ----------
    temporal_data : Dict[str, Any]
        Dictionary containing temporal analysis results
    output_path : Optional[str]
        Path to save plot, if None displays plot
    """
    fig = go.Figure()
    
    # Motion magnitude
    fig.add_trace(go.Scatter(
        y=temporal_data['motion_magnitude'],
        name='Motion Magnitude',
        line=dict(color='blue')
    ))
    
    # Transition points
    for transition in temporal_data['transitions']:
        fig.add_vline(
            x=transition,
            line_dash="dash",
            line_color="red",
            annotation_text="Transition"
        )
    
    fig.update_layout(
        title='Temporal Pattern Analysis',
        xaxis_title='Frame',
        yaxis_title='Magnitude',
        showlegend=True
    )
    
    if output_path:
        fig.write_html(output_path)
    else:
        fig.show()

def create_analysis_dashboard(
    analysis_results: Dict[str, Any],
    output_dir: str
):
    """Create a comprehensive dashboard of all analysis results.
    
    Parameters
    ----------
    analysis_results : Dict[str, Any]
        Dictionary containing all analysis results
    output_dir : str
        Directory to save dashboard files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot individual visualizations
    if 'emotions' in analysis_results:
        plot_emotion_timeline(
            analysis_results['emotions'],
            str(output_path / 'emotions.png')
        )
    
    if 'scenes' in analysis_results:
        plot_scene_segmentation(
            analysis_results['scenes'],
            str(output_path / 'scenes.html')
        )
    
    if 'actions' in analysis_results:
        plot_action_recognition(
            analysis_results['actions'],
            str(output_path / 'actions.png')
        )
    
    if 'temporal' in analysis_results:
        plot_temporal_patterns(
            analysis_results['temporal'],
            str(output_path / 'temporal.html')
        )
    
    # Create index.html combining all visualizations
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Analysis Dashboard</title>
        <style>
            .dashboard {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                padding: 20px;
            }
            .visualization {
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
            }
            img {
                max-width: 100%;
                height: auto;
            }
            iframe {
                width: 100%;
                height: 400px;
                border: none;
            }
        </style>
    </head>
    <body>
        <h1>Video Analysis Dashboard</h1>
        <div class="dashboard">
    """
    
    # Add available visualizations
    for viz_type in ['emotions', 'actions']:
        if viz_type in analysis_results:
            html_content += f"""
            <div class="visualization">
                <h2>{viz_type.title()} Analysis</h2>
                <img src="{viz_type}.png" alt="{viz_type} analysis">
            </div>
            """
    
    for viz_type in ['scenes', 'temporal']:
        if viz_type in analysis_results:
            html_content += f"""
            <div class="visualization">
                <h2>{viz_type.title()} Analysis</h2>
                <iframe src="{viz_type}.html"></iframe>
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path / 'index.html', 'w') as f:
        f.write(html_content) 
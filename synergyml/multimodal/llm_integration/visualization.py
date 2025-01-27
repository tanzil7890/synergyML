"""Visualization module for LLM-enhanced analysis results."""

from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import pandas as pd

class LLMVisualizerMixin:
    """Mixin class providing visualization methods for LLM analysis results."""
    
    def plot_emotion_llm_insights(
        self,
        results: Dict[str, Any],
        title: Optional[str] = None
    ) -> go.Figure:
        """Create interactive visualization of emotion analysis with LLM insights.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Analysis results from LLMEnhancedAnalyzer
        title : Optional[str]
            Custom title for the plot
            
        Returns
        -------
        go.Figure
            Interactive Plotly figure
        """
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Audio Emotions Over Time',
                'Video Emotions Over Time',
                'Emotion Correlation Matrix',
                'Similar Analysis Patterns',
                'LLM Key Insights',
                'Memory Timeline'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "table", "colspan": 2}, None]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot audio emotions
        audio_emotions = np.array(results['emotion_analysis']['raw_emotions']['audio'])
        for i, emotion in enumerate(self.emotion_categories):
            fig.add_trace(
                go.Scatter(
                    y=audio_emotions[:, i],
                    name=f"Audio {emotion}",
                    line=dict(width=1),
                    hovertemplate=f"{emotion}: %{{y:.2f}}<br>Time: %{{x}}",
                ),
                row=1, col=1
            )
            
        # Plot video emotions
        video_emotions = np.array(results['emotion_analysis']['raw_emotions']['video'])
        for i, emotion in enumerate(self.emotion_categories):
            fig.add_trace(
                go.Scatter(
                    y=video_emotions[:, i],
                    name=f"Video {emotion}",
                    line=dict(width=1),
                    hovertemplate=f"{emotion}: %{{y:.2f}}<br>Time: %{{x}}",
                ),
                row=1, col=2
            )
            
        # Plot correlation matrix
        correlation_matrix = np.corrcoef(
            audio_emotions.mean(axis=0),
            video_emotions.mean(axis=0)
        )
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=self.emotion_categories,
                y=self.emotion_categories,
                colorscale='RdBu',
                zmid=0,
                hoverongaps=False,
                hovertemplate="Audio: %{y}<br>Video: %{x}<br>Correlation: %{z:.2f}"
            ),
            row=2, col=1
        )
        
        # Plot similar analysis patterns
        if results.get('similar_analyses'):
            timestamps = []
            similarities = []
            for analysis in results['similar_analyses']:
                timestamps.append(datetime.fromisoformat(analysis['timestamp']))
                similarities.append(analysis.get('similarity', 0.5))
                
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=similarities,
                    mode='markers+lines',
                    name='Similar Analyses',
                    hovertemplate="Time: %{x}<br>Similarity: %{y:.2f}"
                ),
                row=2, col=2
            )
            
        # Add LLM insights table
        insights = self._extract_key_insights(results['llm_insights'])
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Category', 'Insight'],
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=[
                        list(insights.keys()),
                        list(insights.values())
                    ],
                    align="left"
                )
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title=title or "LLM-Enhanced Emotion Analysis",
            showlegend=True,
            template="plotly_white",
            hovermode='closest'
        )
        
        # Add range sliders for time series
        fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=True, row=1, col=2)
        
        return fig
    
    def _extract_key_insights(self, llm_insights: str) -> Dict[str, str]:
        """Extract structured insights from LLM analysis."""
        categories = [
            "Emotional Narrative",
            "Key Transitions",
            "Audio-Visual Relationship",
            "Psychological Patterns",
            "Recommendations"
        ]
        
        insights = {}
        current_category = None
        current_text = []
        
        for line in llm_insights.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number (e.g., "1.", "2.")
            if line[0].isdigit() and line[1] == '.':
                if current_category and current_text:
                    insights[current_category] = ' '.join(current_text)
                current_category = categories[int(line[0]) - 1]
                current_text = [line[3:].strip()]
            else:
                if current_category:
                    current_text.append(line)
                    
        # Add last category
        if current_category and current_text:
            insights[current_category] = ' '.join(current_text)
            
        return insights
    
    @property
    def emotion_categories(self) -> List[str]:
        """List of emotion categories."""
        return [
            'happiness', 'sadness', 'anger', 'fear',
            'surprise', 'disgust', 'neutral'
        ] 
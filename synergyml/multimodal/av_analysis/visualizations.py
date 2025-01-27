"""Visualization tools for audio analysis features."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import librosa.display
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class AudioFeatureVisualizer:
    """Visualization tools for audio features."""
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer.
        
        Parameters
        ----------
        style : str
            Matplotlib style to use
        """
        plt.style.use(style)
    
    def plot_waveform(
        self,
        audio: np.ndarray,
        sr: int,
        title: str = "Waveform"
    ) -> Figure:
        """Plot audio waveform.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        title : str
            Plot title
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        times = np.arange(len(audio)) / sr
        ax.plot(times, audio)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        return fig
    
    def plot_spectrogram(
        self,
        audio: np.ndarray,
        sr: int,
        title: str = "Spectrogram"
    ) -> Figure:
        """Plot spectrogram.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        title : str
            Plot title
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='hz',
            ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.set_title(title)
        return fig

class SpectralVisualizer:
    """Visualization tools for spectral features."""
    
    def plot_spectral_features(
        self,
        features: Dict[str, np.ndarray],
        sr: int
    ) -> Figure:
        """Plot spectral features.
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Spectral features
        sr : int
            Sample rate
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Centroid
        ax1 = fig.add_subplot(gs[0, 0])
        times = librosa.times_like(features['spectral_centroid'])
        ax1.plot(times, features['spectral_centroid'][0])
        ax1.set_title('Spectral Centroid')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        
        # Bandwidth
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, features['spectral_bandwidth'][0])
        ax2.set_title('Spectral Bandwidth')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        
        # Rolloff
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(times, features['spectral_rolloff'][0])
        ax3.set_title('Spectral Rolloff')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        
        # Flatness
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(times, features['spectral_flatness'][0])
        ax4.set_title('Spectral Flatness')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Flatness')
        
        # Contrast
        ax5 = fig.add_subplot(gs[2, :])
        img = librosa.display.specshow(
            features['spectral_contrast'],
            x_axis='time',
            ax=ax5
        )
        fig.colorbar(img, ax=ax5)
        ax5.set_title('Spectral Contrast')
        
        plt.tight_layout()
        return fig
    
    def plot_wavelet_features(
        self,
        features: Dict[str, np.ndarray]
    ) -> Figure:
        """Plot wavelet features.
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Wavelet features
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        # Get number of levels
        levels = len([k for k in features.keys() if 'energy' in k])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Wavelet Features Analysis')
        
        # Energy distribution
        energies = [features[f'wavelet_energy_level_{i}'] for i in range(levels)]
        axes[0, 0].bar(range(levels), energies)
        axes[0, 0].set_title('Wavelet Energy Distribution')
        axes[0, 0].set_xlabel('Decomposition Level')
        axes[0, 0].set_ylabel('Energy')
        
        # Mean values
        means = [features[f'wavelet_mean_level_{i}'] for i in range(levels)]
        axes[0, 1].plot(range(levels), means, 'o-')
        axes[0, 1].set_title('Mean Coefficient Values')
        axes[0, 1].set_xlabel('Decomposition Level')
        axes[0, 1].set_ylabel('Mean')
        
        # Standard deviation
        stds = [features[f'wavelet_std_level_{i}'] for i in range(levels)]
        axes[1, 0].plot(range(levels), stds, 'o-')
        axes[1, 0].set_title('Coefficient Standard Deviation')
        axes[1, 0].set_xlabel('Decomposition Level')
        axes[1, 0].set_ylabel('Std')
        
        # Kurtosis
        kurtosis = [features[f'wavelet_kurtosis_level_{i}'] for i in range(levels)]
        axes[1, 1].plot(range(levels), kurtosis, 'o-')
        axes[1, 1].set_title('Coefficient Kurtosis')
        axes[1, 1].set_xlabel('Decomposition Level')
        axes[1, 1].set_ylabel('Kurtosis')
        
        plt.tight_layout()
        return fig

class EmotionVisualizer:
    """Visualization tools for emotion features."""
    
    def plot_prosodic_features(
        self,
        features: Dict[str, np.ndarray]
    ) -> Figure:
        """Plot prosodic features.
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Prosodic features
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                'Pitch Features',
                'Energy Features',
                'Voice Quality'
            )
        )
        
        # Pitch features
        pitch_features = {
            'Mean': features['pitch_mean'],
            'Std': features['pitch_std'],
            'Range': features['pitch_range']
        }
        fig.add_trace(
            go.Bar(
                x=list(pitch_features.keys()),
                y=list(pitch_features.values()),
                name='Pitch'
            ),
            row=1,
            col=1
        )
        
        # Energy features
        energy_features = {
            'Mean': features['energy_mean'],
            'Std': features['energy_std'],
            'Range': features['energy_range']
        }
        fig.add_trace(
            go.Bar(
                x=list(energy_features.keys()),
                y=list(energy_features.values()),
                name='Energy'
            ),
            row=2,
            col=1
        )
        
        # Voice quality
        voice_features = {
            'ZCR Mean': features['zcr_mean'],
            'ZCR Std': features['zcr_std'],
            'Speech Rate': features['speech_rate']
        }
        fig.add_trace(
            go.Bar(
                x=list(voice_features.keys()),
                y=list(voice_features.values()),
                name='Voice Quality'
            ),
            row=3,
            col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Prosodic Features Analysis"
        )
        return fig
    
    def plot_emotion_dynamics(
        self,
        features: Dict[str, np.ndarray]
    ) -> Figure:
        """Plot emotion dynamics.
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Emotion dynamics features
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Temporal Variance',
                'Feature Derivatives',
                'Emotion Trajectory',
                'Feature Correlation'
            )
        )
        
        # Temporal variance
        fig.add_trace(
            go.Bar(
                y=features['temporal_variance'],
                name='Variance'
            ),
            row=1,
            col=1
        )
        
        # Feature derivatives
        fig.add_trace(
            go.Heatmap(
                z=features['temporal_derivatives'],
                colorscale='Viridis',
                name='Derivatives'
            ),
            row=1,
            col=2
        )
        
        # Emotion trajectory
        fig.add_trace(
            go.Heatmap(
                z=features['emotion_trajectory'],
                colorscale='Viridis',
                name='Trajectory'
            ),
            row=2,
            col=1
        )
        
        # Feature correlation
        corr = np.corrcoef(features['temporal_derivatives'].T)
        fig.add_trace(
            go.Heatmap(
                z=corr,
                colorscale='RdBu',
                name='Correlation'
            ),
            row=2,
            col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Emotion Dynamics Analysis"
        )
        return fig

class MusicVisualizer:
    """Visualization tools for music features."""
    
    def plot_rhythm_features(
        self,
        features: Dict[str, Any]
    ) -> Figure:
        """Plot rhythm features.
        
        Parameters
        ----------
        features : Dict[str, Any]
            Rhythm features
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Beat Times',
                'Beat Histogram',
                'Rhythm Patterns',
                'Tempo Analysis'
            )
        )
        
        # Beat times
        fig.add_trace(
            go.Scatter(
                x=features['beat_times'],
                y=np.ones_like(features['beat_times']),
                mode='markers',
                name='Beat Times'
            ),
            row=1,
            col=1
        )
        
        # Beat histogram
        if 'beat_histogram' in features:
            fig.add_trace(
                go.Bar(
                    y=features['beat_histogram'],
                    name='Beat Distribution'
                ),
                row=1,
                col=2
            )
        
        # Rhythm patterns
        fig.add_trace(
            go.Heatmap(
                z=features['rhythm_patterns'],
                colorscale='Viridis',
                name='Patterns'
            ),
            row=2,
            col=1
        )
        
        # Tempo info
        tempo_data = {
            'Tempo': features['tempo'],
            'Regularity': features.get('beat_regularity', 0)
        }
        fig.add_trace(
            go.Bar(
                x=list(tempo_data.keys()),
                y=list(tempo_data.values()),
                name='Tempo Info'
            ),
            row=2,
            col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Rhythm Analysis"
        )
        return fig
    
    def plot_harmony_features(
        self,
        features: Dict[str, Any]
    ) -> Figure:
        """Plot harmony features.
        
        Parameters
        ----------
        features : Dict[str, Any]
            Harmony features
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Chromagram',
                'Key Detection',
                'Chord Features',
                'Tonal Centroids'
            )
        )
        
        # Chromagram
        chroma = librosa.feature.chroma_cqt(y=features['chroma'])
        fig.add_trace(
            go.Heatmap(
                z=chroma,
                colorscale='Viridis',
                name='Chromagram'
            ),
            row=1,
            col=1
        )
        
        # Key detection
        fig.add_trace(
            go.Bar(
                y=features['estimated_key'],
                name='Key Strength'
            ),
            row=1,
            col=2
        )
        
        # Chord features
        fig.add_trace(
            go.Heatmap(
                z=features['chord_features'],
                colorscale='Viridis',
                name='Chords'
            ),
            row=2,
            col=1
        )
        
        # Tonal centroids
        fig.add_trace(
            go.Heatmap(
                z=features['tonal_centroids'],
                colorscale='Viridis',
                name='Centroids'
            ),
            row=2,
            col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Harmony Analysis"
        )
        return fig 
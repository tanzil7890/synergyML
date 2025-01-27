"""Examples demonstrating the audio analysis capabilities of SynergyML."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from synergyml.multimodal.fusion import (
    MultimodalMediaAnalyzer,
    WaveletAnalyzer,
    CausalityAnalyzer,
    FusionVisualizer
)
from synergyml.config import SynergyMLConfig

def setup_audio_analysis():
    """Set up the audio analysis environment."""
    # Initialize analyzers
    media_analyzer = MultimodalMediaAnalyzer()
    wavelet_analyzer = WaveletAnalyzer()
    causality_analyzer = CausalityAnalyzer()
    visualizer = FusionVisualizer()
    
    return media_analyzer, wavelet_analyzer, causality_analyzer, visualizer

def basic_audio_analysis(audio_path: str):
    """Demonstrate basic audio analysis capabilities.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio file
    y, sr = librosa.load(audio_path)
    print(f"Loaded audio file: {audio_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} seconds")
    
    # Basic feature extraction
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    print("\nBasic Audio Features:")
    print(f"Estimated tempo: {tempo:.2f} BPM")
    print(f"Number of detected beats: {len(beats)}")
    print(f"Average spectral centroid: {np.mean(spectral_centroid):.2f} Hz")
    print(f"Chroma feature shape: {chroma.shape}")
    
    return {
        'signal': y,
        'sr': sr,
        'tempo': tempo,
        'beats': beats,
        'spectral_centroid': spectral_centroid,
        'chroma': chroma
    }

def advanced_spectral_analysis(audio_path: str):
    """Demonstrate advanced spectral analysis.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio and initialize analyzer
    y, sr = librosa.load(audio_path)
    _, wavelet_analyzer, _, visualizer = setup_audio_analysis()
    
    # Compute wavelet transform
    scales = np.arange(1, 128)
    wavelet_results = wavelet_analyzer.compute_wavelet_synchrosqueezed(
        y,
        fs=sr,
        freq_range=(20, sr/2)
    )
    
    # Compute bispectrum for nonlinear analysis
    bispectrum_results = wavelet_analyzer.compute_wavelet_bispectrum(
        y,
        fs=sr,
        freq_range=(20, sr/2)
    )
    
    # Visualize results
    visualizer.plot_synchrosqueezed_transform(wavelet_results)
    visualizer.plot_wavelet_bispectrum(bispectrum_results)
    
    return {
        'wavelet_results': wavelet_results,
        'bispectrum_results': bispectrum_results
    }

def temporal_pattern_analysis(audio_path: str):
    """Analyze temporal patterns in audio.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Extract onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # Compute beat synchronous features
    y_beats = librosa.effects.remix(y, beats)
    mfcc = librosa.feature.mfcc(y=y_beats, sr=sr)
    
    # Compute recurrence matrix
    rec_matrix = librosa.segment.recurrence_matrix(
        mfcc,
        mode='affinity',
        metric='cosine'
    )
    
    print("\nTemporal Pattern Analysis:")
    print(f"Number of beats: {len(beats)}")
    print(f"Beat-synced MFCC shape: {mfcc.shape}")
    print(f"Recurrence matrix shape: {rec_matrix.shape}")
    
    return {
        'onset_env': onset_env,
        'beats': beats,
        'mfcc': mfcc,
        'recurrence_matrix': rec_matrix
    }

def audio_event_detection(audio_path: str):
    """Detect and analyze audio events.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    media_analyzer, _, _, visualizer = setup_audio_analysis()
    
    # Detect onset events
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Compute event features
    event_features = []
    for start_time in onset_times:
        start_frame = int(start_time * sr)
        frame = y[start_frame:start_frame + sr//10]  # 100ms window
        if len(frame) == sr//10:
            spectral = np.mean(np.abs(librosa.stft(frame)))
            centroid = np.mean(librosa.feature.spectral_centroid(y=frame, sr=sr))
            event_features.append([spectral, centroid])
    
    event_features = np.array(event_features)
    
    # Create event detection results
    event_results = {
        'change_points': {
            'time': onset_times,
            'signal': y,
            'points': [{'time': t, 'score': s} for t, s in zip(onset_times, event_features[:, 0])]
        },
        'event_clusters': {
            'features': event_features,
            'labels': np.zeros(len(event_features))  # Placeholder for clustering
        }
    }
    
    # Visualize events
    visualizer.plot_event_detection(event_results)
    
    return event_results

def audio_correlation_analysis(audio_path: str):
    """Analyze correlations in audio features.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    _, _, causality_analyzer, visualizer = setup_audio_analysis()
    
    # Extract multiple features
    spectral = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Compute correlations
    correlation_results = {
        'dynamic_correlation': {
            'matrix': np.corrcoef(spectral),
            'time_points': np.arange(spectral.shape[1]),
            'variables': [f'Freq_{i}' for i in range(spectral.shape[0])]
        },
        'time_freq_correlation': {
            'correlation': np.corrcoef(mfcc, chroma)[:mfcc.shape[0], mfcc.shape[0]:],
            'time': np.arange(mfcc.shape[1]),
            'frequency': np.arange(mfcc.shape[0])
        }
    }
    
    # Visualize correlations
    visualizer.plot_advanced_correlation(correlation_results)
    
    return correlation_results

def main():
    """Run all audio analysis examples."""
    # Set up configuration
    SynergyMLConfig.set_audio_backend('librosa')
    
    # Example audio file path
    audio_path = "example.wav"
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"Please provide a valid audio file path. Could not find: {audio_path}")
        return
    
    print("Running Audio Analysis Examples...")
    
    # Run examples
    basic_results = basic_audio_analysis(audio_path)
    print("\nCompleted basic audio analysis")
    
    spectral_results = advanced_spectral_analysis(audio_path)
    print("Completed advanced spectral analysis")
    
    temporal_results = temporal_pattern_analysis(audio_path)
    print("Completed temporal pattern analysis")
    
    event_results = audio_event_detection(audio_path)
    print("Completed audio event detection")
    
    correlation_results = audio_correlation_analysis(audio_path)
    print("Completed correlation analysis")
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main() 
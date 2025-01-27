"""Examples demonstrating music-specific analysis capabilities of SynergyML."""

import numpy as np
import librosa
from pathlib import Path
from synergyml.multimodal.fusion import (
    MultimodalMediaAnalyzer,
    WaveletAnalyzer,
    FusionVisualizer
)

def analyze_musical_structure(audio_path: str):
    """Analyze musical structure including harmony, rhythm, and form.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Harmonic analysis
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    
    # Chord detection
    chord_features = librosa.feature.chroma_cens(y=harmonic, sr=sr)
    
    # Structural segmentation
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rec_matrix = librosa.segment.recurrence_matrix(mfcc, mode='affinity')
    segments = librosa.segment.agglomerative(rec_matrix, len(mfcc))
    
    # Rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    results = {
        'chroma': chroma,
        'chord_features': chord_features,
        'segments': segments,
        'tempo': tempo,
        'beats': beats,
        'structure_matrix': rec_matrix
    }
    
    # Visualize results
    visualizer.plot_multimodal_summary({
        'wavelet': {'coherence': chroma},
        'time_freq_correlation': {'correlation': rec_matrix}
    })
    
    return results

def analyze_performance_dynamics(audio_path: str):
    """Analyze performance dynamics and expression.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Extract dynamics features
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Tempo variation analysis
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo_var = np.diff(beat_times)
    
    # Spectral contrast for timbral analysis
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    results = {
        'dynamics': {
            'rms': rms,
            'onset_strength': onset_env
        },
        'tempo_variation': tempo_var,
        'timbral_contrast': contrast,
        'beats': beat_times
    }
    
    # Create visualization data
    event_results = {
        'change_points': {
            'time': np.arange(len(rms)) * librosa.get_duration(y=y, sr=sr) / len(rms),
            'signal': rms,
            'points': [{'time': t, 'score': s} for t, s in zip(beat_times, onset_env[beats])]
        }
    }
    
    visualizer.plot_event_detection(event_results)
    
    return results

def analyze_genre_characteristics(audio_path: str):
    """Analyze genre-specific characteristics.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Extract genre-relevant features
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Rhythm patterns
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    
    results = {
        'spectral_features': {
            'rolloff': spectral_rolloff,
            'bandwidth': spectral_bandwidth,
            'zcr': zero_crossing_rate
        },
        'mfcc': mfcc,
        'rhythm': {
            'tempo': tempo,
            'pulse': pulse
        }
    }
    
    # Create correlation visualization
    correlation_results = {
        'dynamic_correlation': {
            'matrix': np.corrcoef(mfcc),
            'time_points': np.arange(mfcc.shape[1]),
            'variables': [f'MFCC_{i+1}' for i in range(mfcc.shape[0])]
        }
    }
    
    visualizer.plot_advanced_correlation(correlation_results)
    
    return results

def main():
    """Run music analysis examples."""
    # Example audio file path
    audio_path = "example_music.wav"
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"Please provide a valid audio file path. Could not find: {audio_path}")
        return
    
    print("Running Music Analysis Examples...")
    
    # Run examples
    structure_results = analyze_musical_structure(audio_path)
    print("Completed musical structure analysis")
    
    dynamics_results = analyze_performance_dynamics(audio_path)
    print("Completed performance dynamics analysis")
    
    genre_results = analyze_genre_characteristics(audio_path)
    print("Completed genre characteristics analysis")
    
    print("\nAll music analysis examples completed successfully!")

if __name__ == "__main__":
    main() 
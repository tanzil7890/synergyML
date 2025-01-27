"""Examples demonstrating speech analysis capabilities of SynergyML."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from synergyml.multimodal.fusion import (
    MultimodalMediaAnalyzer,
    WaveletAnalyzer,
    FusionVisualizer
)

def analyze_speech_prosody(audio_path: str):
    """Analyze speech prosody including pitch, rhythm, and intonation.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Pitch analysis
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
    
    # Energy contour
    rms = librosa.feature.rms(y=y)[0]
    
    # Rhythm and timing
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # Create visualization data
    time_points = librosa.times_like(f0)
    results = {
        'time_freq_correlation': {
            'correlation': np.vstack([f0, rms]),
            'time': time_points,
            'frequency': np.array(['Pitch', 'Energy'])
        }
    }
    
    visualizer.plot_multimodal_summary(results)
    
    return {
        'pitch': f0,
        'voiced_flag': voiced_flag,
        'energy': rms,
        'rhythm': {
            'tempo': tempo,
            'beats': beats
        },
        'time': time_points
    }

def analyze_speech_segments(audio_path: str):
    """Analyze speech segments and phonetic content.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Spectral features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Segment detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Create event detection results
    event_results = {
        'change_points': {
            'time': onset_times,
            'signal': y,
            'points': [{'time': t, 'score': 1.0} for t in onset_times]
        },
        'event_clusters': {
            'features': np.column_stack([mfcc[:, onset_frames], 
                                      delta_mfcc[:, onset_frames]]),
            'labels': np.zeros(len(onset_frames))
        }
    }
    
    visualizer.plot_event_detection(event_results)
    
    return {
        'mfcc': mfcc,
        'delta_mfcc': delta_mfcc,
        'delta2_mfcc': delta2_mfcc,
        'segments': onset_times
    }

def analyze_speaker_characteristics(audio_path: str):
    """Analyze speaker-specific characteristics.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Extract voice features
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Voice quality measures
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    
    results = {
        'voice_features': {
            'f0': f0[voiced_flag],
            'centroid': spectral_centroid,
            'contrast': spectral_contrast
        },
        'voice_quality': {
            'zcr': zero_crossing_rate,
            'harmonic_ratio': np.mean(harmonic) / (np.mean(percussive) + 1e-8)
        }
    }
    
    # Create correlation visualization
    correlation_results = {
        'dynamic_correlation': {
            'matrix': np.corrcoef(np.vstack([
                f0[voiced_flag], 
                spectral_centroid.flatten()[voiced_flag],
                spectral_contrast.mean(axis=0)[voiced_flag]
            ])),
            'variables': ['Pitch', 'Centroid', 'Contrast']
        }
    }
    
    visualizer.plot_advanced_correlation(correlation_results)
    
    return results

def analyze_emotional_content(audio_path: str):
    """Analyze emotional content in speech.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Extract emotion-relevant features
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
    rms = librosa.feature.rms(y=y)[0]
    spectral = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Compute temporal features
    f0_stats = {
        'mean': np.mean(f0[voiced_flag]),
        'std': np.std(f0[voiced_flag]),
        'range': np.ptp(f0[voiced_flag])
    }
    
    energy_stats = {
        'mean': np.mean(rms),
        'std': np.std(rms),
        'range': np.ptp(rms)
    }
    
    results = {
        'pitch_dynamics': f0_stats,
        'energy_dynamics': energy_stats,
        'spectral_features': spectral,
        'mfcc': mfcc
    }
    
    # Create visualization data
    wavelet_results = {
        'wavelet': {
            'coherence': spectral
        },
        'time_freq_correlation': {
            'correlation': np.corrcoef(mfcc),
            'time': np.arange(mfcc.shape[1]),
            'frequency': np.arange(mfcc.shape[0])
        }
    }
    
    visualizer.plot_multimodal_summary(wavelet_results)
    
    return results

def main():
    """Run speech analysis examples."""
    # Example audio file path
    audio_path = "example_speech.wav"
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"Please provide a valid audio file path. Could not find: {audio_path}")
        return
    
    print("Running Speech Analysis Examples...")
    
    # Run examples
    prosody_results = analyze_speech_prosody(audio_path)
    print("Completed prosody analysis")
    
    segment_results = analyze_speech_segments(audio_path)
    print("Completed segment analysis")
    
    speaker_results = analyze_speaker_characteristics(audio_path)
    print("Completed speaker characteristics analysis")
    
    emotion_results = analyze_emotional_content(audio_path)
    print("Completed emotional content analysis")
    
    print("\nAll speech analysis examples completed successfully!")

if __name__ == "__main__":
    main() 
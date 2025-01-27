"""Real-world examples demonstrating practical applications of SynergyML audio analysis."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from synergyml.multimodal.fusion import (
    MultimodalMediaAnalyzer,
    WaveletAnalyzer,
    FusionVisualizer
)

def analyze_podcast_audio(audio_path: str):
    """Analyze podcast audio for quality and content insights.
    
    Parameters
    ----------
    audio_path : str
        Path to podcast audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Voice clarity analysis
    harmonic, percussive = librosa.effects.hpss(y)
    voice_to_noise = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-8)
    
    # Speech segments and silence detection
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Speaker turn detection
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    speaker_changes = np.where(np.diff(np.mean(spectral_contrast, axis=0)) > np.std(spectral_contrast))[0]
    
    results = {
        'voice_quality': {
            'clarity': voice_to_noise,
            'speech_segments': onset_times,
            'speaker_turns': librosa.frames_to_time(speaker_changes)
        },
        'content_structure': {
            'mfcc': mfcc,
            'contrast': spectral_contrast
        }
    }
    
    # Visualize speaker turns and segments
    event_results = {
        'change_points': {
            'time': librosa.frames_to_time(speaker_changes),
            'signal': y,
            'points': [{'time': t, 'score': 1.0} for t in onset_times]
        }
    }
    
    visualizer.plot_event_detection(event_results)
    return results

def analyze_live_music_recording(audio_path: str):
    """Analyze live music recording for quality assessment and enhancement.
    
    Parameters
    ----------
    audio_path : str
        Path to live music recording file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Audio quality metrics
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    
    # Room acoustics analysis
    harmonic, percussive = librosa.effects.hpss(y)
    reverb_ratio = np.mean(np.abs(y - harmonic - percussive))
    
    # Audience noise detection
    noise_frames = librosa.effects.preemphasis(percussive)
    noise_envelope = librosa.feature.rms(y=noise_frames)[0]
    
    # Performance timing analysis
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_stability = np.std(np.diff(librosa.frames_to_time(beats)))
    
    results = {
        'audio_quality': {
            'spectral_rolloff': spectral_rolloff,
            'bandwidth': spectral_bandwidth,
            'flatness': spectral_flatness,
            'reverb': reverb_ratio
        },
        'performance': {
            'tempo': tempo,
            'tempo_stability': tempo_stability,
            'audience_noise': noise_envelope
        }
    }
    
    # Visualize quality metrics over time
    correlation_results = {
        'dynamic_correlation': {
            'matrix': np.corrcoef([
                spectral_rolloff.flatten(),
                spectral_bandwidth.flatten(),
                spectral_flatness,
                noise_envelope
            ]),
            'variables': ['Rolloff', 'Bandwidth', 'Flatness', 'Noise']
        }
    }
    
    visualizer.plot_advanced_correlation(correlation_results)
    return results

def analyze_environmental_audio(audio_path: str):
    """Analyze environmental audio for acoustic monitoring and event detection.
    
    Parameters
    ----------
    audio_path : str
        Path to environmental audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Background noise profile
    spectral = np.abs(librosa.stft(y))
    background = np.median(spectral, axis=1)
    
    # Event detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_frames)
    
    # Frequency band analysis
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    band_energies = librosa.power_to_db(mel_spec)
    
    # Acoustic complexity
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    delta_mfcc = librosa.feature.delta(mfcc)
    complexity = np.mean(np.abs(delta_mfcc))
    
    results = {
        'background': {
            'profile': background,
            'complexity': complexity
        },
        'events': {
            'times': onset_times,
            'strengths': onset_env[onset_frames]
        },
        'spectral': {
            'mel': band_energies,
            'mfcc': mfcc
        }
    }
    
    # Visualize acoustic events
    wavelet_results = {
        'wavelet': {
            'coherence': mel_spec
        },
        'time_freq_correlation': {
            'correlation': np.corrcoef(mfcc),
            'time': np.arange(mfcc.shape[1]),
            'frequency': np.arange(mfcc.shape[0])
        }
    }
    
    visualizer.plot_multimodal_summary(wavelet_results)
    return results

def analyze_voice_quality(audio_path: str):
    """Analyze voice quality for speech assessment or voice training.
    
    Parameters
    ----------
    audio_path : str
        Path to voice recording file
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    visualizer = FusionVisualizer()
    
    # Pitch analysis
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
    pitch_stability = np.std(f0[voiced_flag])
    
    # Voice quality measures
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    harmonic_ratio = np.mean(harmonic) / (np.mean(percussive) + 1e-8)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    # Formant analysis (approximated through MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    formant_stability = np.std(mfcc, axis=1)
    
    results = {
        'pitch': {
            'f0': f0[voiced_flag],
            'stability': pitch_stability
        },
        'voice_quality': {
            'harmonic_ratio': harmonic_ratio,
            'formant_stability': formant_stability
        },
        'spectral': {
            'centroid': spectral_centroid,
            'contrast': spectral_contrast,
            'bandwidth': spectral_bandwidth
        }
    }
    
    # Create visualization
    correlation_results = {
        'dynamic_correlation': {
            'matrix': np.corrcoef([
                f0[voiced_flag],
                spectral_centroid.flatten()[voiced_flag],
                spectral_contrast.mean(axis=0)[voiced_flag]
            ]),
            'variables': ['Pitch', 'Centroid', 'Contrast']
        }
    }
    
    visualizer.plot_advanced_correlation(correlation_results)
    return results

def main():
    """Run real-world audio analysis examples."""
    # Example audio file paths
    podcast_path = "example_podcast.wav"
    live_music_path = "example_live.wav"
    environmental_path = "example_env.wav"
    voice_path = "example_voice.wav"
    
    # Check if audio files exist
    audio_files = {
        'podcast': podcast_path,
        'live_music': live_music_path,
        'environmental': environmental_path,
        'voice': voice_path
    }
    
    print("Running Real-World Audio Analysis Examples...")
    
    for name, path in audio_files.items():
        if not Path(path).exists():
            print(f"Skipping {name} analysis - file not found: {path}")
            continue
            
        print(f"\nAnalyzing {name} audio...")
        if name == 'podcast':
            results = analyze_podcast_audio(path)
        elif name == 'live_music':
            results = analyze_live_music_recording(path)
        elif name == 'environmental':
            results = analyze_environmental_audio(path)
        elif name == 'voice':
            results = analyze_voice_quality(path)
        print(f"Completed {name} analysis")
    
    print("\nAll real-world audio analysis examples completed!")

if __name__ == "__main__":
    main() 
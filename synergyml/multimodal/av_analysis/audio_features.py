"""Advanced audio feature extraction module."""

import numpy as np
import librosa
import torch
import pywt
from scipy import signal
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from statsmodels.tsa.stattools import acf
import scipy.stats

class SpectralFeatures:
    """Advanced spectral feature extraction."""
    
    def __init__(self, sr: int = 16000):
        """Initialize spectral feature extractor.
        
        Parameters
        ----------
        sr : int
            Sample rate
        """
        self.sr = sr
        self.scaler = StandardScaler()
    
    def extract_spectral_features(
        self,
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """Extract comprehensive spectral features.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        frame_length : int
            Frame length for STFT
        hop_length : int
            Hop length for STFT
            
        Returns
        -------
        Dict[str, np.ndarray]
            Spectral features
        """
        # Compute spectrogram
        D = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            S=np.abs(D), sr=self.sr
        )
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            S=np.abs(D), sr=self.sr
        )
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            S=np.abs(D), sr=self.sr
        )
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            S=np.abs(D), sr=self.sr
        )
        
        # Spectral flatness
        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            S=np.abs(D)
        )
        
        return features
    
    def compute_wavelet_features(
        self,
        audio: np.ndarray,
        wavelet: str = 'db4',
        level: int = 5
    ) -> Dict[str, np.ndarray]:
        """Compute wavelet-based features.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        wavelet : str
            Wavelet type
        level : int
            Decomposition level
            
        Returns
        -------
        Dict[str, np.ndarray]
            Wavelet features
        """
        # Wavelet decomposition
        coeffs = pywt.wavedec(audio, wavelet, level=level)
        
        features = {}
        for i, coef in enumerate(coeffs):
            # Energy
            features[f'wavelet_energy_level_{i}'] = np.sum(coef**2)
            # Statistics
            features[f'wavelet_mean_level_{i}'] = np.mean(np.abs(coef))
            features[f'wavelet_std_level_{i}'] = np.std(coef)
            features[f'wavelet_kurtosis_level_{i}'] = scipy.stats.kurtosis(coef)
        
        return features

class EmotionFeatures:
    """Audio emotion feature extraction."""
    
    def __init__(self, sr: int = 16000):
        """Initialize emotion feature extractor.
        
        Parameters
        ----------
        sr : int
            Sample rate
        """
        self.sr = sr
    
    def extract_prosodic_features(
        self,
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """Extract prosodic features for emotion analysis.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        frame_length : int
            Frame length
        hop_length : int
            Hop length
            
        Returns
        -------
        Dict[str, np.ndarray]
            Prosodic features
        """
        features = {}
        
        # Pitch features
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        
        features['pitch_mean'] = np.mean(f0[~np.isnan(f0)])
        features['pitch_std'] = np.std(f0[~np.isnan(f0)])
        features['pitch_range'] = np.ptp(f0[~np.isnan(f0)])
        
        # Energy features
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_range'] = np.ptp(rms)
        
        # Speech rate features
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        features['speech_rate'] = tempo
        
        # Voice quality features
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_emotion_dynamics(
        self,
        audio: np.ndarray,
        window_size: int = 1024
    ) -> Dict[str, np.ndarray]:
        """Extract emotion dynamics features.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        window_size : int
            Analysis window size
            
        Returns
        -------
        Dict[str, np.ndarray]
            Emotion dynamics features
        """
        features = {}
        
        # Segment audio into windows
        windows = librosa.util.frame(audio, frame_length=window_size, hop_length=window_size//2)
        
        # Compute features for each window
        window_features = []
        for window in windows.T:
            # Extract basic features
            mfcc = librosa.feature.mfcc(y=window, sr=self.sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=window, sr=self.sr)
            
            # Combine features
            window_feat = np.concatenate([
                np.mean(mfcc, axis=1),
                np.mean(chroma, axis=1)
            ])
            window_features.append(window_feat)
        
        # Convert to array
        window_features = np.array(window_features)
        
        # Compute temporal dynamics
        features['temporal_variance'] = np.var(window_features, axis=0)
        features['temporal_derivatives'] = np.gradient(window_features, axis=0)
        
        # Compute emotional trajectory
        trajectory_dist = dtw.distance_matrix(window_features)
        features['emotion_trajectory'] = trajectory_dist
        
        return features

class AdvancedMusicFeatures:
    """Advanced music feature extraction."""
    
    def __init__(self, sr: int = 16000):
        """Initialize music feature extractor.
        
        Parameters
        ----------
        sr : int
            Sample rate
        """
        self.sr = sr
    
    def extract_rhythm_features(
        self,
        audio: np.ndarray,
        hop_length: int = 512
    ) -> Dict[str, Any]:
        """Extract advanced rhythm features.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        hop_length : int
            Hop length
            
        Returns
        -------
        Dict[str, Any]
            Rhythm features
        """
        features = {}
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        # Tempo features
        features['tempo'] = tempo
        features['beat_times'] = librosa.frames_to_time(beats, sr=self.sr)
        
        # Beat statistics
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            features['beat_regularity'] = np.std(beat_intervals)
            features['beat_histogram'] = np.histogram(beat_intervals, bins=20)[0]
        
        # Rhythm patterns
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr)
        rhythm_patterns = librosa.decompose.nn_filter(
            mel_spec,
            aggregate=np.median,
            metric='cosine'
        )
        features['rhythm_patterns'] = rhythm_patterns
        
        return features
    
    def extract_harmony_features(
        self,
        audio: np.ndarray,
        hop_length: int = 512
    ) -> Dict[str, Any]:
        """Extract advanced harmony features.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        hop_length : int
            Hop length
            
        Returns
        -------
        Dict[str, Any]
            Harmony features
        """
        features = {}
        
        # Chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, hop_length=hop_length)
        
        # Key detection
        key_krumhansl = librosa.feature.key_krumhansl(chroma=chroma)
        features['estimated_key'] = key_krumhansl
        
        # Chord recognition
        chord_features = librosa.feature.chroma_cens(y=audio, sr=self.sr)
        features['chord_features'] = chord_features
        
        # Tonal centroids
        tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sr)
        features['tonal_centroids'] = tonnetz
        
        return features
    
    def extract_timbre_features(
        self,
        audio: np.ndarray,
        n_mfcc: int = 20
    ) -> Dict[str, Any]:
        """Extract advanced timbre features.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        n_mfcc : int
            Number of MFCC coefficients
            
        Returns
        -------
        Dict[str, Any]
            Timbre features
        """
        features = {}
        
        # MFCC with deltas
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        features['mfcc'] = mfcc
        features['mfcc_delta'] = mfcc_delta
        features['mfcc_delta2'] = mfcc_delta2
        
        # Spectral features
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        features['spectral_contrast'] = spectral_contrast
        
        return features 
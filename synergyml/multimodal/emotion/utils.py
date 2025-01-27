"""Utility functions for emotion analysis."""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from dtaidistance import dtw

def align_modalities(
    audio_emotions: Dict[str, np.ndarray],
    video_emotions: Dict[str, np.ndarray],
    window_size: int
) -> Dict[str, np.ndarray]:
    """Align emotions from different modalities.
    
    Parameters
    ----------
    audio_emotions : Dict[str, np.ndarray]
        Audio emotion predictions
    video_emotions : Dict[str, np.ndarray]
        Video emotion predictions
    window_size : int
        Size of temporal window
        
    Returns
    -------
    Dict[str, np.ndarray]
        Aligned emotion predictions
    """
    # Ensure equal lengths through interpolation
    target_len = min(
        len(audio_emotions['timestamps']),
        len(video_emotions['timestamps'])
    )
    
    aligned = {}
    for modality in ['audio', 'video']:
        emotions = audio_emotions if modality == 'audio' else video_emotions
        
        # Interpolate emotion probabilities
        aligned[modality] = {
            'timestamps': np.linspace(0, 1, target_len),
            'emotions': {}
        }
        
        for emotion in emotions['emotions']:
            aligned[modality]['emotions'][emotion] = np.interp(
                aligned[modality]['timestamps'],
                emotions['timestamps'],
                emotions['emotions'][emotion]
            )
    
    return aligned

def compute_emotion_coherence(
    audio_emotions: Dict[str, np.ndarray],
    video_emotions: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Compute coherence between audio and video emotions.
    
    Parameters
    ----------
    audio_emotions : Dict[str, np.ndarray]
        Audio emotion predictions
    video_emotions : Dict[str, np.ndarray]
        Video emotion predictions
        
    Returns
    -------
    Dict[str, Any]
        Coherence analysis results
    """
    results = {
        'correlation': {},
        'mutual_information': {},
        'dtw_distance': {},
        'overall_coherence': 0.0
    }
    
    # Compute metrics for each emotion
    for emotion in audio_emotions['emotions']:
        audio_signal = audio_emotions['emotions'][emotion]
        video_signal = video_emotions['emotions'][emotion]
        
        # Pearson correlation
        correlation, p_value = stats.pearsonr(audio_signal, video_signal)
        results['correlation'][emotion] = {
            'coefficient': correlation,
            'p_value': p_value
        }
        
        # Mutual information
        results['mutual_information'][emotion] = mutual_info_score(
            audio_signal,
            video_signal
        )
        
        # Dynamic Time Warping distance
        results['dtw_distance'][emotion] = dtw.distance(
            audio_signal,
            video_signal
        )
    
    # Compute overall coherence
    results['overall_coherence'] = np.mean([
        results['correlation'][e]['coefficient']
        for e in results['correlation']
    ])
    
    return results

def detect_emotion_peaks(
    emotion_signal: np.ndarray,
    window_size: int = 5,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Detect peaks in emotion signal.
    
    Parameters
    ----------
    emotion_signal : np.ndarray
        Emotion intensity signal
    window_size : int
        Size of detection window
    threshold : float
        Detection threshold
        
    Returns
    -------
    List[Dict[str, Any]]
        Detected emotion peaks
    """
    peaks = []
    
    # Use rolling window
    for i in range(window_size, len(emotion_signal) - window_size):
        window = emotion_signal[i-window_size:i+window_size]
        center = emotion_signal[i]
        
        # Check if center is peak
        if (center > threshold and 
            center == max(window) and 
            center > np.mean(window) + np.std(window)):
            
            peaks.append({
                'index': i,
                'intensity': center,
                'prominence': center - np.mean(window)
            })
    
    return peaks

def compute_emotion_stability(
    emotion_signal: np.ndarray,
    window_size: int = 5
) -> float:
    """Compute temporal stability of emotion.
    
    Parameters
    ----------
    emotion_signal : np.ndarray
        Emotion intensity signal
    window_size : int
        Size of rolling window
        
    Returns
    -------
    float
        Stability score (0-1)
    """
    if len(emotion_signal) < window_size:
        return 1.0
    
    # Compute rolling variance
    variances = []
    for i in range(len(emotion_signal) - window_size + 1):
        window = emotion_signal[i:i + window_size]
        variances.append(np.var(window))
    
    # Return inverse of mean variance (higher means more stable)
    mean_var = np.mean(variances)
    return 1.0 / (1.0 + mean_var) 
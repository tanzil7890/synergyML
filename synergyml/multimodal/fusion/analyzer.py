"""Multimodal fusion module for SynergyML."""

from typing import Dict, Any, Optional, List, Tuple
import os
from pathlib import Path
import numpy as np
from scipy import stats, signal
from scipy.signal import correlate, correlation_lags, welch, stft
from scipy.stats import bootstrap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score
from dtaidistance import dtw
import pywt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import librosa  # Add this import for audio processing

from ..video import VideoAnalyzer
from ..audio.analyzer import AudioAnalyzer
from ..text.analyzer import TextAnalyzer

class MultimodalMediaAnalyzer:
    """Multimodal analysis class for comprehensive media understanding."""
    
    MODEL_CONFIG = {
        'audio': {
            'speech': 'openai/whisper-large-v3',
            'music': 'facebook/musicgen-large',
            'sound': 'microsoft/audio-spectrogram-transformer'
        },
        'video': {
            'scene': 'microsoft/videomae-base',
            'action': 'facebook/timesformer-base-256'
        },
        'multimodal': {
            'understanding': 'openai/gpt-4-vision-preview',
            'embedding': 'laion/CLAP-ViT-L-14'
        }
    }
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Dict[str, str]]] = None,
        use_gpu: bool = False,
        sampling_rate: int = 44100
    ):
        """Initialize multimodal analyzer.
        
        Parameters
        ----------
        model_config : Optional[Dict[str, Dict[str, str]]]
            Custom model configuration
        use_gpu : bool
            Whether to use GPU acceleration
        sampling_rate : int
            Sampling rate for audio processing
        """
        self.config = model_config or self.MODEL_CONFIG
        
        # Initialize individual analyzers
        self.video_analyzer = VideoAnalyzer()
        self.audio_analyzer = AudioAnalyzer(
            speech_model=self.config['audio']['speech'],
            music_model=self.config['audio']['music'],
            sound_model=self.config['audio']['sound']
        )
        self.text_analyzer = TextAnalyzer(use_gpu=use_gpu)
        self.sr = sampling_rate  # Store sampling rate
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file.
        
        Parameters
        ----------
        video_path : str
            Path to video file
            
        Returns
        -------
        str
            Path to extracted audio file
        """
        # Create temporary directory for audio
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate output path
        video_name = Path(video_path).stem
        audio_path = str(temp_dir / f"{video_name}_audio.wav")
        
        # Extract audio using ffmpeg
        os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path}")
        
        return audio_path
    
    def _synthesize_results(
        self,
        video_results: Dict[str, Any],
        audio_results: Dict[str, Any],
        text_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize results from different modalities.
        
        Parameters
        ----------
        video_results : Dict[str, Any]
            Video analysis results
        audio_results : Dict[str, Any]
            Audio analysis results
        text_results : Optional[Dict[str, Any]]
            Text analysis results
            
        Returns
        -------
        Dict[str, Any]
            Combined analysis results
        """
        synthesis = {
            'video': video_results,
            'audio': audio_results
        }
        
        if text_results:
            synthesis['text'] = text_results
        
        # Add cross-modal insights
        synthesis['cross_modal'] = {
            'temporal_alignment': self._align_modalities(
                video_results,
                audio_results,
                text_results
            ),
            'content_correlation': self._analyze_correlations(
                video_results,
                audio_results,
                text_results
            ),
            'event_detection': self._detect_multimodal_events(
                video_results,
                audio_results,
                text_results
            )
        }
        
        return synthesis
    
    def _align_modalities(
        self,
        video_results: Dict[str, Any],
        audio_results: Dict[str, Any],
        text_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Align events across modalities.
        
        Parameters
        ----------
        video_results : Dict[str, Any]
            Video analysis results
        audio_results : Dict[str, Any]
            Audio analysis results
        text_results : Optional[Dict[str, Any]]
            Text analysis results
            
        Returns
        -------
        Dict[str, Any]
            Temporal alignment information
        """
        alignments = {
            'video_audio': [],
            'speech_text': []
        }
        
        # Align video scenes with audio segments
        if 'scenes' in video_results and 'speech' in audio_results:
            for scene in video_results['scenes']:
                scene_start = scene['start_time']
                scene_end = scene['end_time']
                
                # Find overlapping speech segments
                overlapping_speech = []
                if 'timestamps' in audio_results['speech']:
                    for timestamp in audio_results['speech']['timestamps']:
                        if (timestamp['start'] >= scene_start and 
                            timestamp['start'] < scene_end):
                            overlapping_speech.append(timestamp)
                
                alignments['video_audio'].append({
                    'scene': scene,
                    'speech_segments': overlapping_speech
                })
        
        # Align speech with transcript if available
        if text_results and 'speech' in audio_results:
            alignments['speech_text'] = self._align_speech_text(
                audio_results['speech'],
                text_results
            )
        
        return alignments
    
    def _analyze_correlations(
        self,
        video_results: Dict[str, Any],
        audio_results: Dict[str, Any],
        text_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between modalities.
        
        Parameters
        ----------
        video_results : Dict[str, Any]
            Video analysis results
        audio_results : Dict[str, Any]
            Audio analysis results
        text_results : Optional[Dict[str, Any]]
            Text analysis results
            
        Returns
        -------
        Dict[str, Any]
            Correlation analysis results
        """
        correlations = {
            'action_sound': [],
            'emotion_speech': [],
            'scene_music': []
        }
        
        # Correlate actions with sound events
        if 'actions' in video_results and 'sound' in audio_results:
            correlations['action_sound'] = self._correlate_actions_sounds(
                video_results['actions'],
                audio_results['sound']
            )
        
        # Correlate emotional content
        if 'emotions' in video_results and 'speech' in audio_results:
            correlations['emotion_speech'] = self._correlate_emotions_speech(
                video_results['emotions'],
                audio_results['speech']
            )
        
        # Correlate scenes with music
        if 'scenes' in video_results and 'music' in audio_results:
            correlations['scene_music'] = self._correlate_scenes_music(
                video_results['scenes'],
                audio_results['music']
            )
        
        return correlations
    
    def _detect_multimodal_events(
        self,
        video_results: Dict[str, Any],
        audio_results: Dict[str, Any],
        text_results: Optional[Dict[str, Any]] = None
    ) -> list:
        """Detect events that span multiple modalities.
        
        Parameters
        ----------
        video_results : Dict[str, Any]
            Video analysis results
        audio_results : Dict[str, Any]
            Audio analysis results
        text_results : Optional[Dict[str, Any]]
            Text analysis results
            
        Returns
        -------
        list
            List of detected multimodal events
        """
        events = []
        
        # Detect significant moments where multiple modalities show high activity
        if ('scenes' in video_results and 
            'sound' in audio_results):
            
            for scene in video_results['scenes']:
                scene_start = scene['start_time']
                scene_end = scene['end_time']
                
                # Check for corresponding audio events
                audio_events = self._find_audio_events(
                    audio_results,
                    scene_start,
                    scene_end
                )
                
                if audio_events:
                    events.append({
                        'timestamp': scene_start,
                        'duration': scene_end - scene_start,
                        'video_content': scene['content'],
                        'audio_events': audio_events,
                        'confidence': self._calculate_event_confidence(
                            scene,
                            audio_events
                        )
                    })
        
        return events
    
    def analyze_media(
        self,
        video_path: str,
        transcript_path: Optional[str] = None,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Perform comprehensive media analysis.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        transcript_path : Optional[str]
            Path to transcript file
        analysis_type : str
            Type of analysis to perform
            
        Returns
        -------
        Dict[str, Any]
            Combined analysis results
        """
        # Extract audio from video
        audio_path = self._extract_audio(video_path)
        
        try:
            # Analyze each modality
            video_results = self.video_analyzer.analyze(video_path)
            audio_results = self.audio_analyzer.analyze(audio_path)
            
            text_results = None
            if transcript_path and os.path.exists(transcript_path):
                with open(transcript_path, 'r') as f:
                    transcript = f.read()
                text_results = self.text_analyzer.analyze(transcript)
            
            # Synthesize results
            results = self._synthesize_results(
                video_results,
                audio_results,
                text_results
            )
            
            return results
            
        finally:
            # Cleanup temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Remove temp directory if empty
            temp_dir = Path(audio_path).parent
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
    
    def _compute_mutual_information(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        bins: int = 20
    ) -> float:
        """Compute mutual information between two signals.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal
        signal2 : np.ndarray
            Second signal
        bins : int
            Number of bins for histogram
            
        Returns
        -------
        float
            Mutual information score
        """
        # Discretize signals for MI computation
        hist1, _ = np.histogram(signal1, bins=bins)
        hist2, _ = np.histogram(signal2, bins=bins)
        
        # Normalize histograms to get probability distributions
        p1 = hist1 / np.sum(hist1)
        p2 = hist2 / np.sum(hist2)
        
        return mutual_info_score(p1, p2)
    
    def _compute_cross_correlation(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        max_lag: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute cross-correlation between two signals.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal
        signal2 : np.ndarray
            Second signal
        max_lag : Optional[int]
            Maximum lag to consider
            
        Returns
        -------
        Dict[str, Any]
            Cross-correlation results
        """
        # Normalize signals
        s1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
        s2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
        
        # Compute cross-correlation
        cross_corr = correlate(s1_norm, s2_norm, mode='full')
        lags = correlation_lags(len(s1_norm), len(s2_norm))
        
        if max_lag is not None:
            mask = np.abs(lags) <= max_lag
            cross_corr = cross_corr[mask]
            lags = lags[mask]
        
        # Find maximum correlation and corresponding lag
        max_corr_idx = np.argmax(np.abs(cross_corr))
        
        return {
            'correlation': cross_corr,
            'lags': lags,
            'max_correlation': cross_corr[max_corr_idx],
            'optimal_lag': lags[max_corr_idx]
        }
    
    def _compute_coherence(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        fs: float = 1.0,
        nperseg: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute magnitude squared coherence between two signals.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal
        signal2 : np.ndarray
            Second signal
        fs : float
            Sampling frequency
        nperseg : Optional[int]
            Length of each segment
            
        Returns
        -------
        Dict[str, Any]
            Coherence analysis results
        """
        # Compute coherence
        f, coh = signal.coherence(
            signal1,
            signal2,
            fs=fs,
            nperseg=nperseg or min(256, len(signal1))
        )
        
        return {
            'frequencies': f,
            'coherence': coh,
            'mean_coherence': np.mean(coh),
            'max_coherence': np.max(coh),
            'coherent_freq': f[np.argmax(coh)]
        }
    
    def _compute_significance(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        correlation_type: str = 'pearson',
        n_iterations: int = 1000,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compute statistical significance of correlation.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal
        signal2 : np.ndarray
            Second signal
        correlation_type : str
            Type of correlation to test
        n_iterations : int
            Number of bootstrap iterations
        alpha : float
            Significance level
            
        Returns
        -------
        Dict[str, Any]
            Significance test results
        """
        def compute_correlation(x, y):
            if correlation_type == 'pearson':
                return stats.pearsonr(x, y)[0]
            elif correlation_type == 'spearman':
                return stats.spearmanr(x, y)[0]
            elif correlation_type == 'kendall':
                return stats.kendalltau(x, y)[0]
            else:
                raise ValueError(f"Unsupported correlation type: {correlation_type}")
        
        # Compute observed correlation
        observed_corr = compute_correlation(signal1, signal2)
        
        # Perform bootstrap test
        data = np.column_stack([signal1, signal2])
        bootstrap_results = bootstrap(
            (data,),
            lambda x: compute_correlation(x[:, 0], x[:, 1]),
            n_resamples=n_iterations
        )
        
        # Compute confidence intervals
        ci_low, ci_high = bootstrap_results.confidence_interval
        
        # Compute p-value
        null_dist = np.zeros(n_iterations)
        for i in range(n_iterations):
            perm = np.random.permutation(len(signal1))
            null_dist[i] = compute_correlation(signal1, signal2[perm])
        
        p_value = np.mean(np.abs(null_dist) >= np.abs(observed_corr))
        
        return {
            'correlation': observed_corr,
            'p_value': p_value,
            'significant': p_value < alpha,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'bootstrap_std': bootstrap_results.standard_error
        }
    
    def _compute_frequency_analysis(
        self,
        signal: np.ndarray,
        fs: float = 1.0
    ) -> Dict[str, Any]:
        """Compute comprehensive frequency domain analysis.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        fs : float
            Sampling frequency
            
        Returns
        -------
        Dict[str, Any]
            Frequency analysis results
        """
        # Compute power spectral density
        freqs, psd = welch(signal, fs=fs)
        
        # Compute Short-time Fourier Transform
        f, t, Zxx = stft(signal, fs=fs)
        
        # Compute dominant frequencies
        dominant_freqs = freqs[np.argsort(psd)[-3:]]
        
        # Compute frequency bands
        freq_bands = {
            'low': np.mean(psd[freqs < fs/4]),
            'mid': np.mean(psd[(freqs >= fs/4) & (freqs < fs/2)]),
            'high': np.mean(psd[freqs >= fs/2])
        }
        
        # Compute spectral entropy
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        return {
            'psd': {
                'frequencies': freqs,
                'power': psd,
                'dominant_frequencies': dominant_freqs
            },
            'stft': {
                'frequencies': f,
                'times': t,
                'magnitudes': np.abs(Zxx)
            },
            'spectral_features': {
                'entropy': spectral_entropy,
                'centroid': np.sum(freqs * psd) / np.sum(psd),
                'bandwidth': np.sqrt(np.sum(((freqs - np.sum(freqs * psd) / np.sum(psd))**2) * psd) / np.sum(psd)),
                'flatness': stats.gmean(psd) / np.mean(psd),
                'rolloff': freqs[np.cumsum(psd) >= 0.85 * np.sum(psd)][0]
            },
            'frequency_bands': freq_bands
        }
    
    def _test_granger_causality(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        max_lag: int = 5,
        test_type: str = 'ssr_chi2test'
    ) -> Dict[str, Any]:
        """Test for Granger causality between two signals.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal (potential cause)
        signal2 : np.ndarray
            Second signal (potential effect)
        max_lag : int
            Maximum number of lags to test
        test_type : str
            Type of test statistic to use
            
        Returns
        -------
        Dict[str, Any]
            Granger causality test results
        """
        # Ensure signals are stationary
        def make_stationary(x):
            # Take first difference if signal is non-stationary
            adf_result = adfuller(x)
            if adf_result[1] > 0.05:  # p-value > 0.05 indicates non-stationarity
                return np.diff(x)
            return x
        
        s1_stat = make_stationary(signal1)
        s2_stat = make_stationary(signal2)
        
        # Ensure equal lengths after differencing
        min_len = min(len(s1_stat), len(s2_stat))
        s1_stat = s1_stat[:min_len]
        s2_stat = s2_stat[:min_len]
        
        # Prepare data for VAR model
        data = np.column_stack([s1_stat, s2_stat])
        
        # Test Granger causality in both directions
        results = {
            'forward': {},  # signal1 -> signal2
            'backward': {}  # signal2 -> signal1
        }
        
        # Test signal1 -> signal2
        gc_forward = grangercausalitytests(
            data,
            maxlag=max_lag,
            verbose=False
        )
        
        # Test signal2 -> signal1
        data_reverse = np.column_stack([s2_stat, s1_stat])
        gc_backward = grangercausalitytests(
            data_reverse,
            maxlag=max_lag,
            verbose=False
        )
        
        # Extract test results
        for lag in range(1, max_lag + 1):
            results['forward'][lag] = {
                'test_stat': gc_forward[lag][0][test_type][0],
                'p_value': gc_forward[lag][0][test_type][1],
                'significant': gc_forward[lag][0][test_type][1] < 0.05
            }
            results['backward'][lag] = {
                'test_stat': gc_backward[lag][0][test_type][0],
                'p_value': gc_backward[lag][0][test_type][1],
                'significant': gc_backward[lag][0][test_type][1] < 0.05
            }
        
        # Determine optimal lag using AIC
        model = VAR(data)
        aic_results = []
        for lag in range(1, max_lag + 1):
            try:
                result = model.fit(lag)
                aic_results.append((lag, result.aic))
            except:
                continue
        
        if aic_results:
            optimal_lag = min(aic_results, key=lambda x: x[1])[0]
            results['optimal_lag'] = optimal_lag
            results['optimal_results'] = {
                'forward': results['forward'][optimal_lag],
                'backward': results['backward'][optimal_lag]
            }
        
        return results
    
    def _compute_wavelet_analysis(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        wavelet: str = 'cmor1.5-1.0',
        scales: Optional[np.ndarray] = None,
        freq_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Perform continuous wavelet transform analysis.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        fs : float
            Sampling frequency
        wavelet : str
            Wavelet to use (default: complex Morlet)
        scales : Optional[np.ndarray]
            Scales for wavelet transform
        freq_range : Optional[Tuple[float, float]]
            Frequency range to analyze (min_freq, max_freq)
            
        Returns
        -------
        Dict[str, Any]
            Wavelet analysis results
        """
        # Normalize signal
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        # Set up scales if not provided
        if scales is None:
            if freq_range is not None:
                min_freq, max_freq = freq_range
                scales = pywt.frequency2scale(
                    wavelet,
                    np.logspace(
                        np.log10(min_freq),
                        np.log10(max_freq),
                        num=64
                    ),
                    fs
                )
            else:
                scales = np.arange(1, min(128, len(signal)//2))
        
        # Perform continuous wavelet transform
        coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
        
        # Compute wavelet power spectrum
        power = np.abs(coef) ** 2
        
        # Compute global wavelet spectrum
        global_ws = np.mean(power, axis=1)
        
        # Find ridge and instantaneous frequency
        ridge_idx = np.argmax(power, axis=0)
        inst_freq = freqs[ridge_idx]
        
        # Compute time-averaged power in different frequency bands
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        band_powers = {}
        for band_name, (fmin, fmax) in freq_bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            if np.any(mask):
                band_powers[band_name] = np.mean(power[mask, :])
        
        return {
            'coefficients': coef,
            'frequencies': freqs,
            'power': power,
            'global_spectrum': global_ws,
            'ridge': {
                'indices': ridge_idx,
                'frequencies': inst_freq
            },
            'band_powers': band_powers,
            'scales': scales
        }
    
    def _correlate_actions_sounds(
        self,
        actions: Dict[str, List[float]],
        sounds: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Correlate video actions with sound events.
        
        Parameters
        ----------
        actions : Dict[str, List[float]]
            Video action recognition results
        sounds : Dict[str, Any]
            Sound analysis results
            
        Returns
        -------
        List[Dict[str, Any]]
            Action-sound correlations
        """
        correlations = []
        sound_features = sounds['features']['spectral']
        # Get sampling rate from sounds dictionary or use default
        fs = sounds.get('sampling_rate', 44100)  # Default to 44.1kHz if not provided
        
        for action_name, action_scores in actions.items():
            # Prepare signals
            action_array = np.array(action_scores)
            centroid_array = sound_features['centroid'].flatten()
            
            # Ensure equal lengths through interpolation
            if len(action_array) != len(centroid_array):
                target_len = min(len(action_array), len(centroid_array))
                action_array = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(action_array)),
                    action_array
                )
                centroid_array = np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(centroid_array)),
                    centroid_array
                )
            
            # Calculate advanced correlations with significance testing
            correlation = {
                'action': action_name,
                'basic_metrics': {
                    'pearson': self._compute_significance(
                        action_array,
                        centroid_array,
                        correlation_type='pearson'
                    ),
                    'spearman': self._compute_significance(
                        action_array,
                        centroid_array,
                        correlation_type='spearman'
                    ),
                    'kendall': self._compute_significance(
                        action_array,
                        centroid_array,
                        correlation_type='kendall'
                    )
                },
                'temporal_metrics': {
                    'dtw_distance': dtw.distance(action_array, centroid_array),
                    'peak_alignment': self._analyze_peak_alignment(
                        action_array,
                        centroid_array
                    )
                },
                'information_metrics': {
                    'mutual_information': self._compute_mutual_information(
                        action_array,
                        centroid_array
                    )
                },
                'frequency_metrics': {
                    'coherence': self._compute_coherence(
                        action_array,
                        centroid_array
                    ),
                    'action_spectrum': self._compute_frequency_analysis(action_array),
                    'sound_spectrum': self._compute_frequency_analysis(centroid_array)
                },
                'cross_correlation': self._compute_cross_correlation(
                    action_array,
                    centroid_array,
                    max_lag=int(len(action_array) * 0.1)
                ),
                'causality': self._test_granger_causality(
                    action_array,
                    centroid_array
                ),
                'wavelet_analysis': {
                    'action': self._compute_wavelet_analysis(
                        action_array,
                        fs=fs,  # Pass sampling rate explicitly
                        freq_range=(0.1, fs/2)
                    ),
                    'sound': self._compute_wavelet_analysis(
                        centroid_array,
                        fs=fs,  # Pass sampling rate explicitly
                        freq_range=(0.1, fs/2)
                    )
                }
            }
            
            correlations.append(correlation)
        
        return correlations
    
    def _correlate_emotions_speech(
        self,
        emotions: Dict[str, List[float]],
        speech: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Correlate visual emotions with speech characteristics.
        
        Parameters
        ----------
        emotions : Dict[str, List[float]]
            Visual emotion recognition results
        speech : Dict[str, Any]
            Speech analysis results
            
        Returns
        -------
        List[Dict[str, Any]]
            Emotion-speech correlations
        """
        correlations = []
        speech_segments = speech.get('timestamps', [])
        
        for emotion_name, emotion_scores in emotions.items():
            emotion_timeline = []
            segment_correlations = []
            
            for segment in speech_segments:
                start_idx = int(segment['start'] * len(emotion_scores))
                end_idx = int(segment['end'] * len(emotion_scores))
                segment_emotions = emotion_scores[start_idx:end_idx]
                
                # Extract speech features if available
                speech_features = segment.get('features', {})
                if speech_features:
                    segment_correlations.append({
                        'temporal_alignment': self._compute_cross_correlation(
                            np.array(segment_emotions),
                            speech_features.get('energy', np.zeros_like(segment_emotions))
                        ),
                        'spectral_analysis': {
                            'emotion_spectrum': self._compute_frequency_analysis(
                                np.array(segment_emotions)
                            ),
                            'speech_spectrum': self._compute_frequency_analysis(
                                speech_features.get('energy', np.zeros_like(segment_emotions))
                            )
                        },
                        'significance_tests': {
                            'emotion_energy': self._compute_significance(
                                np.array(segment_emotions),
                                speech_features.get('energy', np.zeros_like(segment_emotions))
                            ),
                            'emotion_pitch': self._compute_significance(
                                np.array(segment_emotions),
                                speech_features.get('pitch', np.zeros_like(segment_emotions))
                            )
                        },
                        'causality_analysis': {
                            'emotion_energy': self._test_granger_causality(
                                np.array(segment_emotions),
                                speech_features.get('energy', np.zeros_like(segment_emotions))
                            ),
                            'emotion_pitch': self._test_granger_causality(
                                np.array(segment_emotions),
                                speech_features.get('pitch', np.zeros_like(segment_emotions))
                            )
                        },
                        'time_frequency_analysis': {
                            'emotion': self._compute_wavelet_analysis(
                                np.array(segment_emotions)
                            ),
                            'speech': self._compute_wavelet_analysis(
                                speech_features.get('energy', np.zeros_like(segment_emotions))
                            )
                        }
                    })
                
                emotion_timeline.append({
                    'timestamp': segment['start'],
                    'duration': segment['end'] - segment['start'],
                    'emotion_intensity': np.mean(segment_emotions),
                    'emotion_variance': np.var(segment_emotions),
                    'emotion_entropy': stats.entropy(
                        np.histogram(segment_emotions, bins=10)[0]
                    ),
                    'speech_content': segment.get('text', '')
                })
            
            correlations.append({
                'emotion': emotion_name,
                'timeline': emotion_timeline,
                'segment_correlations': segment_correlations,
                'overall_stats': {
                    'mean_intensity': np.mean([e['emotion_intensity'] for e in emotion_timeline]),
                    'intensity_variance': np.var([e['emotion_intensity'] for e in emotion_timeline]),
                    'temporal_stability': self._compute_temporal_stability(
                        [e['emotion_intensity'] for e in emotion_timeline]
                    )
                }
            })
        
        return correlations
    
    def _compute_temporal_stability(
        self,
        signal: List[float],
        window_size: int = 3
    ) -> float:
        """Compute temporal stability of a signal.
        
        Parameters
        ----------
        signal : List[float]
            Input signal
        window_size : int
            Size of rolling window
            
        Returns
        -------
        float
            Stability score
        """
        if len(signal) < window_size:
            return 1.0
        
        # Compute rolling variance
        variances = []
        for i in range(len(signal) - window_size + 1):
            window = signal[i:i + window_size]
            variances.append(np.var(window))
        
        # Return inverse of mean variance (higher means more stable)
        mean_var = np.mean(variances)
        return 1.0 / (1.0 + mean_var)
    
    def _analyze_peak_alignment(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze alignment of peaks between two signals.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal
        signal2 : np.ndarray
            Second signal
            
        Returns
        -------
        Dict[str, Any]
            Peak alignment analysis
        """
        # Find peaks in both signals
        peaks1 = self._find_signal_peaks(signal1)
        peaks2 = self._find_signal_peaks(signal2)
        
        # Calculate temporal distances between peaks
        peak_distances = []
        for p1 in peaks1:
            min_dist = min(abs(p1 - p2) for p2 in peaks2)
            peak_distances.append(min_dist)
        
        return {
            'average_distance': np.mean(peak_distances),
            'max_distance': np.max(peak_distances),
            'aligned_peaks': sum(1 for d in peak_distances if d < len(signal1) * 0.05)
        }
    
    def _analyze_tonal_stability(self, chroma: np.ndarray) -> float:
        """Analyze tonal stability using chroma features.
        
        Parameters
        ----------
        chroma : np.ndarray
            Chromagram matrix
            
        Returns
        -------
        float
            Tonal stability score
        """
        # Calculate frame-to-frame cosine similarity
        similarities = []
        for i in range(chroma.shape[1] - 1):
            sim = cosine_similarity(
                chroma[:, i].reshape(1, -1),
                chroma[:, i + 1].reshape(1, -1)
            )[0, 0]
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def _detect_key_changes(self, chroma: np.ndarray) -> List[Dict[str, Any]]:
        """Detect key changes using chroma features.
        
        Parameters
        ----------
        chroma : np.ndarray
            Chromagram matrix
            
        Returns
        -------
        List[Dict[str, Any]]
            Detected key changes
        """
        key_changes = []
        window_size = 8  # frames
        
        for i in range(0, chroma.shape[1] - window_size, window_size):
            window1 = chroma[:, i:i + window_size]
            window2 = chroma[:, i + window_size:i + 2 * window_size]
            
            if window2.shape[1] < window_size:
                break
            
            # Compare average chroma vectors
            similarity = cosine_similarity(
                np.mean(window1, axis=1).reshape(1, -1),
                np.mean(window2, axis=1).reshape(1, -1)
            )[0, 0]
            
            if similarity < 0.85:  # Threshold for key change detection
                key_changes.append({
                    'timestamp': i * window_size,
                    'confidence': 1 - similarity
                })
        
        return key_changes
    
    def _find_signal_peaks(
        self,
        signal: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Find peaks in a signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        threshold : float
            Peak detection threshold
            
        Returns
        -------
        np.ndarray
            Indices of detected peaks
        """
        # Normalize signal
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        # Find peaks
        peaks = []
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and 
                signal[i] > signal[i+1] and 
                signal[i] > threshold):
                peaks.append(i)
        
        return np.array(peaks)

    def analyze_correlation(self, signal1: np.ndarray, signal2: np.ndarray, 
                          method: str = 'pearson') -> Dict[str, Any]:
        """Analyze correlation between two signals."""
        # Use self.sr instead of fs
        times = np.arange(len(signal1)) / self.sr
        
        if method == 'pearson':
            corr, p_value = stats.pearsonr(signal1, signal2)
        elif method == 'dtw':
            distance = dtw.distance(signal1, signal2)
            corr = 1 / (1 + distance)  # Convert distance to similarity
            p_value = None
            
        return {
            'correlation': corr,
            'p_value': p_value,
            'times': times,
            'method': method
        } 
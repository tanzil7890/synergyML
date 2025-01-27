"""Cross-modal emotion analysis module for SynergyML."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torchaudio
import torchvision
from pathlib import Path
from transformers import (
    Wav2Vec2ForSequenceClassification,
    TimesformerForVideoClassification,
    RobertaForSequenceClassification,
    Wav2Vec2Processor,
    VideoMAEImageProcessor
)
import scipy.signal
import sklearn.metrics

from ..fusion import MultimodalMediaAnalyzer
from .utils import align_modalities, compute_emotion_coherence, detect_emotion_peaks, compute_emotion_stability
from .visualization import plot_emotion_alignment

class EmotionAnalyzer(MultimodalMediaAnalyzer):
    """Analyze emotions across audio, video, and text modalities."""
    
    EMOTION_CATEGORIES = [
        'anger', 'disgust', 'fear', 'happiness', 
        'sadness', 'surprise', 'neutral'
    ]
    
    def __init__(
        self,
        model_config: Optional[Dict[str, str]] = None,
        use_gpu: bool = False
    ):
        """Initialize emotion analyzer.
        
        Parameters
        ----------
        model_config : Optional[Dict[str, str]]
            Custom model configuration
        use_gpu : bool
            Whether to use GPU acceleration
        """
        super().__init__()
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Default emotion models
        self.emotion_models = {
            'audio': 'MIT/wav2vec-emotion',
            'video': 'facebook/timesformer-emotion',
            'text': 'roberta-base-emotion'
        }
        if model_config:
            self.emotion_models.update(model_config)
            
        # Initialize models and processors
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize emotion detection models."""
        try:
            # Audio emotion model and processor
            self.audio_emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.emotion_models['audio']
            ).to(self.device)
            self.audio_processor = Wav2Vec2Processor.from_pretrained(
                self.emotion_models['audio']
            )
            
            # Video emotion model and processor
            self.video_emotion_model = TimesformerForVideoClassification.from_pretrained(
                self.emotion_models['video']
            ).to(self.device)
            self.video_processor = VideoMAEImageProcessor.from_pretrained(
                self.emotion_models['video']
            )
            
            # Text emotion model
            self.text_emotion_model = RobertaForSequenceClassification.from_pretrained(
                self.emotion_models['text']
            ).to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize emotion models: {str(e)}")
    
    def analyze_emotional_coherence(
        self,
        video_path: str,
        window_size: int = 5,
        sampling_rate: int = 16000
    ) -> Dict[str, Any]:
        """Analyze emotional coherence across modalities.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        window_size : int
            Size of temporal window for pattern analysis
        sampling_rate : int
            Audio sampling rate
            
        Returns
        -------
        Dict[str, Any]
            Emotional coherence analysis results
        """
        # Extract audio
        audio_path = self._extract_audio(video_path)
        
        try:
            # Get emotions from each modality
            audio_emotions = self._analyze_audio_emotions(audio_path, sampling_rate)
            video_emotions = self._analyze_video_emotions(video_path)
            
            # Align modalities
            aligned_emotions = align_modalities(
                audio_emotions,
                video_emotions,
                window_size
            )
            
            # Compute coherence
            coherence_results = compute_emotion_coherence(
                aligned_emotions['audio'],
                aligned_emotions['video']
            )
            
            # Analyze temporal patterns
            temporal_results = self._analyze_emotion_patterns(
                aligned_emotions,
                window_size
            )
            
            results = {
                'emotion_alignment': coherence_results,
                'temporal_patterns': temporal_results,
                'raw_emotions': aligned_emotions
            }
            
            # Generate visualizations
            plot_emotion_alignment(results)
            
            return results
            
        finally:
            # Cleanup temporary audio file
            if Path(audio_path).exists():
                Path(audio_path).unlink()
    
    def _analyze_audio_emotions(
        self,
        audio_path: str,
        sampling_rate: int
    ) -> Dict[str, np.ndarray]:
        """Analyze emotions in audio.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        sampling_rate : int
            Target sampling rate
            
        Returns
        -------
        Dict[str, np.ndarray]
            Audio emotion analysis results
        """
        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process audio in chunks
        chunk_size = sampling_rate * 5  # 5-second chunks
        hop_length = chunk_size // 2    # 50% overlap
        
        timestamps = []
        emotion_probs = {emotion: [] for emotion in self.EMOTION_CATEGORIES}
        
        for i in range(0, waveform.shape[1], hop_length):
            chunk = waveform[:, i:i + chunk_size]
            if chunk.shape[1] < chunk_size:
                # Pad last chunk if needed
                chunk = torch.nn.functional.pad(
                    chunk, (0, chunk_size - chunk.shape[1])
                )
            
            # Process audio chunk
            inputs = self.audio_processor(
                chunk.numpy(),
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.audio_emotion_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Store results
            timestamps.append(i / sampling_rate)
            for emotion_idx, emotion in enumerate(self.EMOTION_CATEGORIES):
                emotion_probs[emotion].append(probs[0, emotion_idx].item())
        
        return {
            'timestamps': np.array(timestamps),
            'emotions': {
                emotion: np.array(probs)
                for emotion, probs in emotion_probs.items()
            }
        }
    
    def _analyze_video_emotions(
        self,
        video_path: str
    ) -> Dict[str, np.ndarray]:
        """Analyze emotions in video frames.
        
        Parameters
        ----------
        video_path : str
            Path to video file
            
        Returns
        -------
        Dict[str, np.ndarray]
            Video emotion analysis results
        """
        # Load video
        video = torchvision.io.read_video(video_path)[0]  # [T, H, W, C]
        fps = torchvision.io.read_video_metadata(video_path)['video']['fps']
        
        # Process video in chunks
        chunk_size = 16  # TimeSformer default
        hop_length = chunk_size // 2
        
        timestamps = []
        emotion_probs = {emotion: [] for emotion in self.EMOTION_CATEGORIES}
        
        for i in range(0, len(video), hop_length):
            chunk = video[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk if needed
                chunk = torch.nn.functional.pad(
                    chunk,
                    (0, 0, 0, 0, 0, 0, 0, chunk_size - len(chunk))
                )
            
            # Process video chunk
            inputs = self.video_processor(
                list(chunk.numpy()),
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.video_emotion_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Store results
            timestamps.append(i / fps)
            for emotion_idx, emotion in enumerate(self.EMOTION_CATEGORIES):
                emotion_probs[emotion].append(probs[0, emotion_idx].item())
        
        return {
            'timestamps': np.array(timestamps),
            'emotions': {
                emotion: np.array(probs)
                for emotion, probs in emotion_probs.items()
            }
        }
    
    def _analyze_emotion_patterns(
        self,
        aligned_emotions: Dict[str, np.ndarray],
        window_size: int
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in emotions.
        
        Parameters
        ----------
        aligned_emotions : Dict[str, np.ndarray]
            Aligned emotion predictions
        window_size : int
            Size of analysis window
            
        Returns
        -------
        Dict[str, Any]
            Temporal pattern analysis results
        """
        results = {
            'peaks': {},
            'stability': {},
            'pattern_strength': np.zeros_like(
                aligned_emotions['audio']['timestamps']
            ),
            'timestamps': aligned_emotions['audio']['timestamps']
        }
        
        # Analyze each emotion
        for emotion in self.EMOTION_CATEGORIES:
            # Get emotion signals
            audio_signal = aligned_emotions['audio']['emotions'][emotion]
            video_signal = aligned_emotions['video']['emotions'][emotion]
            
            # Detect peaks
            audio_peaks = detect_emotion_peaks(audio_signal, window_size)
            video_peaks = detect_emotion_peaks(video_signal, window_size)
            
            results['peaks'][emotion] = {
                'audio': audio_peaks,
                'video': video_peaks
            }
            
            # Compute stability
            results['stability'][emotion] = np.mean([
                compute_emotion_stability(audio_signal, window_size),
                compute_emotion_stability(video_signal, window_size)
            ])
            
            # Update pattern strength
            pattern_contribution = np.maximum(audio_signal, video_signal)
            results['pattern_strength'] += pattern_contribution
        
        # Normalize pattern strength
        results['pattern_strength'] /= len(self.EMOTION_CATEGORIES)
        
        return results
    
    def analyze_emotion_complexity(
        self,
        aligned_emotions: Dict[str, np.ndarray],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """Analyze emotional complexity and dynamics.
        
        Parameters
        ----------
        aligned_emotions : Dict[str, np.ndarray]
            Aligned emotion predictions
        window_size : int
            Size of analysis window
            
        Returns
        -------
        Dict[str, Any]
            Complexity analysis results
        """
        results = {
            'entropy': {},
            'transitions': {},
            'dominance': {},
            'blending': {},
            'timestamps': aligned_emotions['audio']['timestamps']
        }
        
        for modality in ['audio', 'video']:
            # Compute emotion entropy over time
            emotion_probs = np.stack([
                aligned_emotions[modality]['emotions'][emotion]
                for emotion in self.EMOTION_CATEGORIES
            ], axis=1)
            
            # Rolling entropy calculation
            entropy = []
            for i in range(len(emotion_probs)):
                window = emotion_probs[
                    max(0, i - window_size):min(len(emotion_probs), i + window_size + 1)
                ]
                window_entropy = -np.sum(
                    window * np.log2(window + 1e-10), axis=1
                ).mean()
                entropy.append(window_entropy)
            
            results['entropy'][modality] = np.array(entropy)
            
            # Analyze emotion transitions
            transitions = []
            dominant_emotions = np.argmax(emotion_probs, axis=1)
            
            for i in range(1, len(dominant_emotions)):
                if dominant_emotions[i] != dominant_emotions[i-1]:
                    transitions.append({
                        'time': results['timestamps'][i],
                        'from': self.EMOTION_CATEGORIES[dominant_emotions[i-1]],
                        'to': self.EMOTION_CATEGORIES[dominant_emotions[i]]
                    })
            
            results['transitions'][modality] = transitions
            
            # Compute emotion dominance
            total_dominance = np.zeros(len(self.EMOTION_CATEGORIES))
            for i, emotion in enumerate(self.EMOTION_CATEGORIES):
                total_dominance[i] = np.mean(
                    aligned_emotions[modality]['emotions'][emotion]
                )
            
            results['dominance'][modality] = {
                emotion: score for emotion, score in zip(
                    self.EMOTION_CATEGORIES, total_dominance
                )
            }
            
            # Analyze emotion blending
            blending = []
            for probs in emotion_probs:
                sorted_probs = np.sort(probs)[::-1]
                ratio = sorted_probs[1] / (sorted_probs[0] + 1e-10)
                blending.append(ratio)
            
            results['blending'][modality] = np.array(blending)
        
        return results
    
    def analyze_emotion_synchronization(
        self,
        aligned_emotions: Dict[str, np.ndarray],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """Analyze synchronization between audio and video emotions.
        
        Parameters
        ----------
        aligned_emotions : Dict[str, np.ndarray]
            Aligned emotion predictions
        window_size : int
            Size of analysis window
            
        Returns
        -------
        Dict[str, Any]
            Synchronization analysis results
        """
        results = {
            'lag_correlation': {},
            'coherence': {},
            'mutual_information': {},
            'timestamps': aligned_emotions['audio']['timestamps']
        }
        
        for emotion in self.EMOTION_CATEGORIES:
            audio_signal = aligned_emotions['audio']['emotions'][emotion]
            video_signal = aligned_emotions['video']['emotions'][emotion]
            
            # Compute lag correlation
            max_lag = window_size
            lags = np.arange(-max_lag, max_lag + 1)
            correlations = []
            
            for lag in lags:
                if lag < 0:
                    corr = np.corrcoef(
                        audio_signal[:lag], video_signal[-lag:]
                    )[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(
                        audio_signal[lag:], video_signal[:-lag]
                    )[0, 1]
                else:
                    corr = np.corrcoef(audio_signal, video_signal)[0, 1]
                correlations.append(corr)
            
            results['lag_correlation'][emotion] = {
                'lags': lags,
                'correlations': np.array(correlations)
            }
            
            # Compute windowed coherence
            coherence = []
            for i in range(len(audio_signal)):
                window_start = max(0, i - window_size)
                window_end = min(len(audio_signal), i + window_size + 1)
                
                audio_window = audio_signal[window_start:window_end]
                video_window = video_signal[window_start:window_end]
                
                # Compute coherence using Welch's method
                f, Cxy = scipy.signal.coherence(
                    audio_window, video_window,
                    fs=1.0/np.mean(np.diff(results['timestamps']))
                )
                coherence.append(np.mean(Cxy))
            
            results['coherence'][emotion] = np.array(coherence)
            
            # Compute mutual information over time
            mi_scores = []
            for i in range(len(audio_signal)):
                window_start = max(0, i - window_size)
                window_end = min(len(audio_signal), i + window_size + 1)
                
                audio_window = audio_signal[window_start:window_end]
                video_window = video_signal[window_start:window_end]
                
                # Discretize signals for MI computation
                n_bins = min(10, len(audio_window))
                audio_bins = np.digitize(
                    audio_window,
                    bins=np.linspace(0, 1, n_bins)
                )
                video_bins = np.digitize(
                    video_window,
                    bins=np.linspace(0, 1, n_bins)
                )
                
                mi = sklearn.metrics.mutual_info_score(
                    audio_bins, video_bins
                )
                mi_scores.append(mi)
            
            results['mutual_information'][emotion] = np.array(mi_scores)
        
        return results
    
    def analyze_emotion_context(
        self,
        aligned_emotions: Dict[str, np.ndarray],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """Analyze emotional context and patterns.
        
        Parameters
        ----------
        aligned_emotions : Dict[str, np.ndarray]
            Aligned emotion predictions
        window_size : int
            Size of analysis window
            
        Returns
        -------
        Dict[str, Any]
            Context analysis results
        """
        results = {
            'emotion_sequences': {},
            'co_occurrence': np.zeros(
                (len(self.EMOTION_CATEGORIES), len(self.EMOTION_CATEGORIES))
            ),
            'temporal_patterns': {},
            'timestamps': aligned_emotions['audio']['timestamps']
        }
        
        for modality in ['audio', 'video']:
            # Extract dominant emotion sequences
            emotion_probs = np.stack([
                aligned_emotions[modality]['emotions'][emotion]
                for emotion in self.EMOTION_CATEGORIES
            ], axis=1)
            
            dominant_emotions = np.argmax(emotion_probs, axis=1)
            sequences = []
            
            current_seq = {
                'emotion': self.EMOTION_CATEGORIES[dominant_emotions[0]],
                'start': 0
            }
            
            for i in range(1, len(dominant_emotions)):
                if dominant_emotions[i] != dominant_emotions[i-1]:
                    current_seq['end'] = i
                    current_seq['duration'] = (
                        results['timestamps'][current_seq['end']] -
                        results['timestamps'][current_seq['start']]
                    )
                    sequences.append(current_seq)
                    
                    current_seq = {
                        'emotion': self.EMOTION_CATEGORIES[dominant_emotions[i]],
                        'start': i
                    }
            
            # Add final sequence
            current_seq['end'] = len(dominant_emotions)
            current_seq['duration'] = (
                results['timestamps'][current_seq['end']-1] -
                results['timestamps'][current_seq['start']]
            )
            sequences.append(current_seq)
            
            results['emotion_sequences'][modality] = sequences
            
            # Compute emotion co-occurrence
            for i in range(len(self.EMOTION_CATEGORIES)):
                for j in range(len(self.EMOTION_CATEGORIES)):
                    emotion1 = aligned_emotions[modality]['emotions'][
                        self.EMOTION_CATEGORIES[i]
                    ]
                    emotion2 = aligned_emotions[modality]['emotions'][
                        self.EMOTION_CATEGORIES[j]
                    ]
                    
                    # Consider co-occurrence when both emotions are above threshold
                    co_occur = np.mean(
                        (emotion1 > 0.3) & (emotion2 > 0.3)
                    )
                    results['co_occurrence'][i, j] += co_occur
            
            # Analyze temporal patterns
            patterns = {}
            for emotion in self.EMOTION_CATEGORIES:
                signal = aligned_emotions[modality]['emotions'][emotion]
                
                # Detect cyclic patterns using autocorrelation
                acf = np.correlate(signal, signal, mode='full')[len(signal)-1:]
                peaks, _ = scipy.signal.find_peaks(acf, distance=window_size)
                
                if len(peaks) > 1:
                    avg_period = np.mean(np.diff(peaks))
                    strength = np.mean(acf[peaks[1:]])
                else:
                    avg_period = 0
                    strength = 0
                
                patterns[emotion] = {
                    'period': avg_period,
                    'strength': strength
                }
            
            results['temporal_patterns'][modality] = patterns
        
        # Normalize co-occurrence matrix
        results['co_occurrence'] /= 2  # Average across modalities
        
        return results
    
    def analyze_emotion_changepoints(
        self,
        aligned_emotions: Dict[str, np.ndarray],
        window_size: int = 5,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Detect significant changes in emotional patterns.
        
        Parameters
        ----------
        aligned_emotions : Dict[str, np.ndarray]
            Aligned emotion predictions
        window_size : int
            Size of analysis window
        threshold : float
            Threshold for change point detection
            
        Returns
        -------
        Dict[str, Any]
            Change point analysis results
        """
        results = {
            'changepoints': {},
            'change_scores': {},
            'regime_statistics': {},
            'timestamps': aligned_emotions['audio']['timestamps']
        }
        
        for modality in ['audio', 'video']:
            # Stack emotion probabilities
            emotion_probs = np.stack([
                aligned_emotions[modality]['emotions'][emotion]
                for emotion in self.EMOTION_CATEGORIES
            ], axis=1)
            
            # Compute change scores using sliding windows
            change_scores = []
            changepoints = []
            
            for i in range(window_size, len(emotion_probs) - window_size):
                window1 = emotion_probs[i - window_size:i]
                window2 = emotion_probs[i:i + window_size]
                
                # Compute KL divergence between windows
                mean1 = np.mean(window1, axis=0)
                mean2 = np.mean(window2, axis=0)
                
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                kl_div = np.sum(mean1 * np.log((mean1 + eps) / (mean2 + eps)))
                change_scores.append(kl_div)
                
                # Detect changepoint if score exceeds threshold
                if kl_div > threshold:
                    changepoints.append({
                        'time': results['timestamps'][i],
                        'score': kl_div,
                        'before_state': {
                            emotion: float(mean1[j])
                            for j, emotion in enumerate(self.EMOTION_CATEGORIES)
                        },
                        'after_state': {
                            emotion: float(mean2[j])
                            for j, emotion in enumerate(self.EMOTION_CATEGORIES)
                        }
                    })
            
            results['changepoints'][modality] = changepoints
            results['change_scores'][modality] = np.array(change_scores)
            
            # Compute statistics for each regime between changepoints
            regime_stats = []
            start_idx = 0
            
            for cp in changepoints:
                end_idx = np.where(results['timestamps'] >= cp['time'])[0][0]
                
                regime = emotion_probs[start_idx:end_idx]
                stats = {
                    'start_time': results['timestamps'][start_idx],
                    'end_time': cp['time'],
                    'duration': cp['time'] - results['timestamps'][start_idx],
                    'dominant_emotion': self.EMOTION_CATEGORIES[
                        np.argmax(np.mean(regime, axis=0))
                    ],
                    'stability': np.mean([
                        compute_emotion_stability(regime[:, i], window_size)
                        for i in range(regime.shape[1])
                    ]),
                    'complexity': -np.sum(
                        np.mean(regime, axis=0) * 
                        np.log2(np.mean(regime, axis=0) + eps)
                    )
                }
                regime_stats.append(stats)
                start_idx = end_idx
            
            # Add final regime
            if start_idx < len(emotion_probs):
                regime = emotion_probs[start_idx:]
                stats = {
                    'start_time': results['timestamps'][start_idx],
                    'end_time': results['timestamps'][-1],
                    'duration': results['timestamps'][-1] - results['timestamps'][start_idx],
                    'dominant_emotion': self.EMOTION_CATEGORIES[
                        np.argmax(np.mean(regime, axis=0))
                    ],
                    'stability': np.mean([
                        compute_emotion_stability(regime[:, i], window_size)
                        for i in range(regime.shape[1])
                    ]),
                    'complexity': -np.sum(
                        np.mean(regime, axis=0) * 
                        np.log2(np.mean(regime, axis=0) + eps)
                    )
                }
                regime_stats.append(stats)
            
            results['regime_statistics'][modality] = regime_stats
        
        return results
    
    def analyze_emotion_trends(
        self,
        aligned_emotions: Dict[str, np.ndarray],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """Analyze emotional trends and patterns over time.
        
        Parameters
        ----------
        aligned_emotions : Dict[str, np.ndarray]
            Aligned emotion predictions
        window_size : int
            Size of analysis window
            
        Returns
        -------
        Dict[str, Any]
            Trend analysis results
        """
        results = {
            'trends': {},
            'seasonality': {},
            'momentum': {},
            'timestamps': aligned_emotions['audio']['timestamps']
        }
        
        for modality in ['audio', 'video']:
            trends = {}
            seasonality = {}
            momentum = {}
            
            for emotion in self.EMOTION_CATEGORIES:
                signal = aligned_emotions[modality]['emotions'][emotion]
                
                # Compute trend using rolling average
                trend = np.convolve(
                    signal,
                    np.ones(window_size) / window_size,
                    mode='valid'
                )
                
                # Pad trend to match original length
                pad_width = len(signal) - len(trend)
                trend = np.pad(
                    trend,
                    (pad_width - pad_width//2, pad_width//2),
                    mode='edge'
                )
                
                # Compute seasonality by subtracting trend
                seasonal = signal - trend
                
                # Detect seasonal patterns using FFT
                fft = np.fft.fft(seasonal)
                freqs = np.fft.fftfreq(len(seasonal))
                
                # Find dominant frequencies
                main_freq_idx = np.argsort(np.abs(fft))[-3:]  # Top 3 frequencies
                seasonal_patterns = [{
                    'frequency': float(freqs[idx]),
                    'amplitude': float(np.abs(fft[idx]) / len(seasonal)),
                    'phase': float(np.angle(fft[idx]))
                } for idx in main_freq_idx]
                
                # Compute momentum indicators
                momentum_indicators = {
                    'rate_of_change': np.gradient(signal),
                    'acceleration': np.gradient(np.gradient(signal)),
                    'trend_strength': np.abs(
                        scipy.stats.pearsonr(
                            np.arange(len(signal)), signal
                        )[0]
                    )
                }
                
                trends[emotion] = trend
                seasonality[emotion] = {
                    'residuals': seasonal,
                    'patterns': seasonal_patterns
                }
                momentum[emotion] = momentum_indicators
            
            results['trends'][modality] = trends
            results['seasonality'][modality] = seasonality
            results['momentum'][modality] = momentum
        
        return results 
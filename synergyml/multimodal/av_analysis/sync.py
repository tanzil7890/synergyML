"""Video-audio synchronization module."""

import numpy as np
import librosa
import decord
import cv2
from typing import Dict, Any, List, Tuple, Optional
from scipy.signal import correlate
from scipy.optimize import minimize
from .utils import ModelCache

class AVSynchronizer:
    """Video-audio synchronization tools."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize synchronizer.
        
        Parameters
        ----------
        cache_dir : Optional[str]
            Cache directory
        device : Optional[str]
            Device to use for computations
        """
        self.cache_dir = cache_dir
        if cache_dir:
            self.cache = ModelCache(cache_dir)
        else:
            self.cache = None
    
    def extract_audio_features(
        self,
        audio: np.ndarray,
        sr: int,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """Extract audio features for synchronization.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        frame_length : int
            Frame length
        hop_length : int
            Hop length
            
        Returns
        -------
        Dict[str, np.ndarray]
            Audio features
        """
        # Extract onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=hop_length
        )
        
        # Extract spectral features
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13,
            hop_length=hop_length
        )
        
        return {
            'onset_env': onset_env,
            'mel_spec': mel_spec_db,
            'mfcc': mfcc
        }
    
    def extract_video_features(
        self,
        video_path: str,
        frame_step: int = 1
    ) -> Dict[str, np.ndarray]:
        """Extract video features for synchronization.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        frame_step : int
            Frame step size
            
        Returns
        -------
        Dict[str, np.ndarray]
            Video features
        """
        # Load video
        video = decord.VideoReader(video_path)
        total_frames = len(video)
        
        # Initialize features
        motion_features = []
        intensity_features = []
        
        # Process frames
        prev_frame = None
        for i in range(0, total_frames, frame_step):
            frame = video[i].asnumpy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate intensity
            intensity_features.append(np.mean(gray))
            
            if prev_frame is not None:
                # Calculate motion
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame,
                    gray,
                    None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                motion_features.append(motion)
            
            prev_frame = gray
        
        # Add zero for first frame motion
        motion_features.insert(0, 0)
        
        return {
            'motion': np.array(motion_features),
            'intensity': np.array(intensity_features)
        }
    
    def compute_sync_offset(
        self,
        audio_features: Dict[str, np.ndarray],
        video_features: Dict[str, np.ndarray],
        max_offset_seconds: float = 5.0,
        sr: int = 16000,
        hop_length: int = 512
    ) -> Dict[str, Any]:
        """Compute synchronization offset between audio and video.
        
        Parameters
        ----------
        audio_features : Dict[str, np.ndarray]
            Audio features
        video_features : Dict[str, np.ndarray]
            Video features
        max_offset_seconds : float
            Maximum offset to consider
        sr : int
            Audio sample rate
        hop_length : int
            Hop length used for audio features
            
        Returns
        -------
        Dict[str, Any]
            Synchronization results
        """
        # Resample video features to match audio feature rate
        video_times = np.arange(len(video_features['motion']))
        audio_times = np.arange(len(audio_features['onset_env']))
        video_motion = np.interp(
            audio_times,
            video_times,
            video_features['motion']
        )
        
        # Compute correlation between onset and motion
        correlation = correlate(
            audio_features['onset_env'],
            video_motion,
            mode='full'
        )
        
        # Find best offset
        max_offset_frames = int(max_offset_seconds * sr / hop_length)
        center = len(correlation) // 2
        search_range = slice(
            center - max_offset_frames,
            center + max_offset_frames
        )
        offset_frames = np.argmax(correlation[search_range]) - max_offset_frames
        offset_seconds = offset_frames * hop_length / sr
        
        # Compute confidence
        max_corr = np.max(correlation[search_range])
        mean_corr = np.mean(correlation[search_range])
        std_corr = np.std(correlation[search_range])
        confidence = (max_corr - mean_corr) / std_corr
        
        return {
            'offset_seconds': offset_seconds,
            'offset_frames': offset_frames,
            'confidence': float(confidence),
            'correlation': correlation[search_range]
        }
    
    def compute_quality_metrics(
        self,
        audio_features: Dict[str, np.ndarray],
        video_features: Dict[str, np.ndarray],
        sync_offset: float,
        sr: int,
        hop_length: int
    ) -> Dict[str, float]:
        """Compute synchronization quality metrics.
        
        Parameters
        ----------
        audio_features : Dict[str, np.ndarray]
            Audio features
        video_features : Dict[str, np.ndarray]
            Video features
        sync_offset : float
            Synchronization offset in seconds
        sr : int
            Audio sample rate
        hop_length : int
            Hop length used for audio features
            
        Returns
        -------
        Dict[str, float]
            Quality metrics
        """
        # Align features using offset
        offset_frames = int(sync_offset * sr / hop_length)
        if offset_frames >= 0:
            audio_onset = audio_features['onset_env'][offset_frames:]
            video_motion = video_features['motion'][:len(audio_onset)]
        else:
            audio_onset = audio_features['onset_env'][:offset_frames]
            video_motion = video_features['motion'][-len(audio_onset):]
        
        # Normalize features
        audio_onset = (audio_onset - np.mean(audio_onset)) / np.std(audio_onset)
        video_motion = (video_motion - np.mean(video_motion)) / np.std(video_motion)
        
        # Compute correlation coefficient
        correlation = np.corrcoef(audio_onset, video_motion)[0, 1]
        
        # Compute mutual information
        hist_2d, _, _ = np.histogram2d(audio_onset, video_motion, bins=20)
        hist_2d_prob = hist_2d / np.sum(hist_2d)
        mutual_info = np.sum(hist_2d_prob * np.log2(hist_2d_prob + 1e-10))
        
        # Compute peak alignment score
        audio_peaks = librosa.util.peak_pick(audio_onset, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
        video_peaks = librosa.util.peak_pick(video_motion, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
        peak_alignment = len(set(audio_peaks).intersection(set(video_peaks))) / max(len(audio_peaks), len(video_peaks))
        
        return {
            'correlation': float(correlation),
            'mutual_information': float(mutual_info),
            'peak_alignment': float(peak_alignment),
            'overall_quality': float((correlation + mutual_info + peak_alignment) / 3)
        }
    
    def refine_sync_offset(
        self,
        audio_features: Dict[str, np.ndarray],
        video_features: Dict[str, np.ndarray],
        initial_offset: float,
        sr: int,
        hop_length: int
    ) -> Dict[str, Any]:
        """Refine synchronization offset using optimization.
        
        Parameters
        ----------
        audio_features : Dict[str, np.ndarray]
            Audio features
        video_features : Dict[str, np.ndarray]
            Video features
        initial_offset : float
            Initial offset estimate in seconds
        sr : int
            Audio sample rate
        hop_length : int
            Hop length used for audio features
            
        Returns
        -------
        Dict[str, Any]
            Refined synchronization results
        """
        def objective(offset):
            # Convert offset to frames
            offset_frames = int(offset * sr / hop_length)
            
            # Align features
            if offset_frames >= 0:
                audio_onset = audio_features['onset_env'][offset_frames:]
                video_motion = video_features['motion'][:len(audio_onset)]
            else:
                audio_onset = audio_features['onset_env'][:offset_frames]
                video_motion = video_features['motion'][-len(audio_onset):]
            
            if len(audio_onset) < 10 or len(video_motion) < 10:
                return 1.0
            
            # Compute correlation
            correlation = np.corrcoef(audio_onset, video_motion)[0, 1]
            return -correlation  # Minimize negative correlation
        
        # Optimize around initial offset
        result = minimize(
            objective,
            x0=initial_offset,
            method='Nelder-Mead',
            options={'maxiter': 100}
        )
        
        refined_offset = float(result.x[0])
        
        # Compute quality metrics for refined offset
        quality_metrics = self.compute_quality_metrics(
            audio_features,
            video_features,
            refined_offset,
            sr,
            hop_length
        )
        
        return {
            'offset_seconds': refined_offset,
            'offset_frames': int(refined_offset * sr / hop_length),
            'optimization_success': bool(result.success),
            'optimization_iterations': int(result.nit),
            'quality_metrics': quality_metrics
        }
    
    def synchronize(
        self,
        video_path: str,
        audio_path: str,
        max_offset_seconds: float = 5.0
    ) -> Dict[str, Any]:
        """Synchronize video and audio.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        audio_path : str
            Path to audio file
        max_offset_seconds : float
            Maximum offset to consider
            
        Returns
        -------
        Dict[str, Any]
            Synchronization results
        """
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                'av_sync',
                f"{video_path}_{audio_path}",
                {'max_offset': max_offset_seconds}
            )
            cached_results = self.cache.load(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        audio_features = self.extract_audio_features(audio, sr)
        video_features = self.extract_video_features(video_path)
        
        # Compute sync offset
        sync_results = self.compute_sync_offset(
            audio_features,
            video_features,
            max_offset_seconds,
            sr=sr
        )
        
        # Add metadata
        sync_results['video_path'] = video_path
        sync_results['audio_path'] = audio_path
        sync_results['sample_rate'] = sr
        
        # Cache results
        if self.cache:
            self.cache.save(cache_key, sync_results)
        
        return sync_results
    
    def synchronize_with_quality(
        self,
        video_path: str,
        audio_path: str,
        max_offset_seconds: float = 5.0,
        refine_sync: bool = True
    ) -> Dict[str, Any]:
        """Synchronize video and audio with quality assessment.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        audio_path : str
            Path to audio file
        max_offset_seconds : float
            Maximum offset to consider
        refine_sync : bool
            Whether to refine synchronization using optimization
            
        Returns
        -------
        Dict[str, Any]
            Synchronization results with quality metrics
        """
        # Get initial sync results
        initial_results = self.synchronize(video_path, audio_path, max_offset_seconds)
        
        # Load features
        audio, sr = librosa.load(audio_path, sr=None)
        audio_features = self.extract_audio_features(audio, sr)
        video_features = self.extract_video_features(video_path)
        
        if refine_sync:
            # Refine synchronization
            refined_results = self.refine_sync_offset(
                audio_features,
                video_features,
                initial_results['offset_seconds'],
                sr,
                512  # Default hop length
            )
            
            # Update results
            results = {**initial_results, **refined_results}
        else:
            # Compute quality metrics for initial sync
            quality_metrics = self.compute_quality_metrics(
                audio_features,
                video_features,
                initial_results['offset_seconds'],
                sr,
                512
            )
            results = {
                **initial_results,
                'quality_metrics': quality_metrics
            }
        
        return results
    
    def apply_sync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        sync_results: Dict[str, Any]
    ) -> None:
        """Apply synchronization to create synchronized video.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        audio_path : str
            Path to audio file
        output_path : str
            Path to output synchronized video
        sync_results : Dict[str, Any]
            Synchronization results from synchronize()
        """
        import av
        
        # Open input files
        video = av.open(video_path)
        audio = av.open(audio_path)
        
        # Create output container
        output = av.open(output_path, mode='w')
        
        # Copy stream settings
        video_stream = output.add_stream(template=video.streams.video[0])
        audio_stream = output.add_stream(template=audio.streams.audio[0])
        
        # Calculate offset in audio samples
        offset_samples = int(sync_results['offset_seconds'] * sync_results['sample_rate'])
        
        # Process video frames
        for frame in video.decode(video=0):
            # Encode frame
            for packet in video_stream.encode(frame):
                output.mux(packet)
        
        # Flush video encoder
        for packet in video_stream.encode():
            output.mux(packet)
        
        # Process audio frames with offset
        if offset_samples > 0:
            # Video is ahead, skip some audio
            for i, frame in enumerate(audio.decode(audio=0)):
                if i * frame.samples >= offset_samples:
                    for packet in audio_stream.encode(frame):
                        output.mux(packet)
        else:
            # Audio is ahead, add silence
            silence = np.zeros(abs(offset_samples), dtype=np.float32)
            silence_frame = av.AudioFrame.from_ndarray(
                silence,
                layout='mono'
            )
            silence_frame.rate = sync_results['sample_rate']
            
            # Add silence
            for packet in audio_stream.encode(silence_frame):
                output.mux(packet)
            
            # Add actual audio
            for frame in audio.decode(audio=0):
                for packet in audio_stream.encode(frame):
                    output.mux(packet)
        
        # Flush audio encoder
        for packet in audio_stream.encode():
            output.mux(packet)
        
        # Close files
        output.close()
        video.close()
        audio.close() 
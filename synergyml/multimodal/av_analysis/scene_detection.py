"""Advanced scene detection methods."""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
import torch
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import decord
from .utils import ModelCache

class SceneDetector:
    """Advanced scene detection methods."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize scene detector.
        
        Parameters
        ----------
        cache_dir : Optional[str]
            Cache directory
        device : Optional[str]
            Device to use for computations
        """
        self.cache_dir = cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if cache_dir:
            self.cache = ModelCache(cache_dir)
        else:
            self.cache = None
    
    def detect_scenes_histogram(
        self,
        video_path: str,
        threshold: float = 0.3,
        window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Detect scenes using color histogram differences.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        threshold : float
            Difference threshold for scene change
        window_size : int
            Window size for smoothing
            
        Returns
        -------
        List[Dict[str, Any]]
            Detected scene boundaries
        """
        # Load video
        video = decord.VideoReader(video_path)
        total_frames = len(video)
        
        # Initialize variables
        prev_hist = None
        scene_boundaries = []
        
        # Process frames
        for i in range(0, total_frames):
            frame = video[i].asnumpy()
            
            # Calculate color histogram
            hist = cv2.calcHist(
                [frame],
                [0, 1, 2],
                None,
                [8, 8, 8],
                [0, 256, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Calculate histogram difference
                diff = cv2.compareHist(
                    prev_hist,
                    hist,
                    cv2.HISTCMP_CORREL
                )
                
                if abs(1 - diff) > threshold:
                    scene_boundaries.append({
                        'frame_idx': i,
                        'timestamp': i / video.get_avg_fps(),
                        'confidence': abs(1 - diff),
                        'method': 'histogram'
                    })
            
            prev_hist = hist
        
        return self._smooth_boundaries(scene_boundaries, window_size)
    
    def detect_scenes_optical_flow(
        self,
        video_path: str,
        threshold: float = 0.5,
        window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Detect scenes using optical flow analysis.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        threshold : float
            Motion threshold for scene change
        window_size : int
            Window size for smoothing
            
        Returns
        -------
        List[Dict[str, Any]]
            Detected scene boundaries
        """
        # Load video
        video = decord.VideoReader(video_path)
        total_frames = len(video)
        
        # Initialize variables
        prev_frame_gray = None
        scene_boundaries = []
        flow_history = []
        
        # Process frames
        for i in range(0, total_frames):
            frame = video[i].asnumpy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame_gray is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame_gray,
                    frame_gray,
                    None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate flow magnitude
                magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                flow_history.append(magnitude)
                
                if len(flow_history) > window_size:
                    # Detect sudden changes in motion
                    avg_flow = np.mean(flow_history[-window_size:])
                    if magnitude > avg_flow * (1 + threshold):
                        scene_boundaries.append({
                            'frame_idx': i,
                            'timestamp': i / video.get_avg_fps(),
                            'confidence': magnitude / avg_flow,
                            'method': 'optical_flow'
                        })
            
            prev_frame_gray = frame_gray
        
        return self._smooth_boundaries(scene_boundaries, window_size)
    
    def detect_scenes_deep(
        self,
        video_path: str,
        chunk_size: int = 30,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect scenes using deep features.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        chunk_size : int
            Number of frames to process at once
        threshold : float
            Similarity threshold for scene change
            
        Returns
        -------
        List[Dict[str, Any]]
            Detected scene boundaries
        """
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                'scene_detection_deep',
                video_path,
                {'chunk_size': chunk_size}
            )
            cached_results = self.cache.load(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Load video
        video = decord.VideoReader(video_path)
        total_frames = len(video)
        
        # Load feature extractor
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = model.to(self.device)
        model.eval()
        
        # Process frames
        features = []
        timestamps = []
        
        for i in range(0, total_frames, chunk_size):
            # Load chunk
            end_idx = min(i + chunk_size, total_frames)
            frames = video.get_batch(list(range(i, end_idx))).asnumpy()
            
            # Extract features
            with torch.no_grad():
                batch = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(self.device)
                feat = model.conv1(batch)
                feat = model.bn1(feat)
                feat = model.relu(feat)
                feat = model.maxpool(feat)
                feat = model.layer1(feat)
                feat = torch.mean(feat, dim=[2, 3])
                features.append(feat.cpu().numpy())
                timestamps.extend([j / video.get_avg_fps() for j in range(i, end_idx)])
        
        # Combine features
        features = np.concatenate(features, axis=0)
        
        # Detect scene changes using feature similarity
        scene_boundaries = []
        for i in range(1, len(features)):
            similarity = np.dot(features[i], features[i-1]) / (
                np.linalg.norm(features[i]) * np.linalg.norm(features[i-1])
            )
            if similarity < threshold:
                scene_boundaries.append({
                    'frame_idx': i * chunk_size,
                    'timestamp': timestamps[i],
                    'confidence': 1 - similarity,
                    'method': 'deep'
                })
        
        # Cache results
        if self.cache:
            self.cache.save(cache_key, scene_boundaries)
        
        return scene_boundaries
    
    def detect_scenes_combined(
        self,
        video_path: str,
        methods: List[str] = ['histogram', 'optical_flow', 'deep'],
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Detect scenes using multiple methods.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        methods : List[str]
            Methods to use
        weights : Optional[List[float]]
            Weights for each method
            
        Returns
        -------
        List[Dict[str, Any]]
            Detected scene boundaries
        """
        if weights is None:
            weights = [1.0] * len(methods)
        
        all_boundaries = []
        
        # Collect boundaries from all methods
        for method, weight in zip(methods, weights):
            if method == 'histogram':
                boundaries = self.detect_scenes_histogram(video_path)
            elif method == 'optical_flow':
                boundaries = self.detect_scenes_optical_flow(video_path)
            elif method == 'deep':
                boundaries = self.detect_scenes_deep(video_path)
            
            # Apply weight
            for boundary in boundaries:
                boundary['confidence'] *= weight
            
            all_boundaries.extend(boundaries)
        
        # Sort by timestamp
        all_boundaries.sort(key=lambda x: x['timestamp'])
        
        # Merge close boundaries
        merged = []
        i = 0
        while i < len(all_boundaries):
            current = all_boundaries[i]
            j = i + 1
            while j < len(all_boundaries) and \
                  abs(all_boundaries[j]['timestamp'] - current['timestamp']) < 0.5:
                current['confidence'] = max(
                    current['confidence'],
                    all_boundaries[j]['confidence']
                )
                j += 1
            merged.append(current)
            i = j
        
        return merged
    
    def _smooth_boundaries(
        self,
        boundaries: List[Dict[str, Any]],
        window_size: int
    ) -> List[Dict[str, Any]]:
        """Smooth scene boundaries by removing close ones.
        
        Parameters
        ----------
        boundaries : List[Dict[str, Any]]
            Scene boundaries
        window_size : int
            Window size for smoothing
            
        Returns
        -------
        List[Dict[str, Any]]
            Smoothed boundaries
        """
        if not boundaries:
            return boundaries
        
        # Sort by frame index
        boundaries.sort(key=lambda x: x['frame_idx'])
        
        # Remove boundaries that are too close
        smoothed = [boundaries[0]]
        for boundary in boundaries[1:]:
            if boundary['frame_idx'] - smoothed[-1]['frame_idx'] > window_size:
                smoothed.append(boundary)
        
        return smoothed
    
    def analyze_transition_type(
        self,
        video_path: str,
        boundary: Dict[str, Any],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze the type of scene transition.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        boundary : Dict[str, Any]
            Scene boundary information
        window_size : int
            Number of frames to analyze before/after boundary
            
        Returns
        -------
        Dict[str, Any]
            Transition analysis results
        """
        # Load video
        video = decord.VideoReader(video_path)
        fps = video.get_avg_fps()
        frame_idx = boundary['frame_idx']
        
        # Get frames around boundary
        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(video), frame_idx + window_size)
        frames = video.get_batch(list(range(start_idx, end_idx))).asnumpy()
        
        # Calculate frame differences
        diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            diffs.append(diff)
        
        # Analyze transition pattern
        diffs = np.array(diffs)
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        pattern = diffs / max_diff
        
        # Classify transition type
        if max_diff < 30:  # Low difference threshold
            trans_type = 'match_cut'
        elif np.all(pattern[:window_size//2] < 0.3) and np.all(pattern[window_size//2:] < 0.3):
            trans_type = 'cut'
        elif np.all(np.diff(pattern[:window_size//2]) > 0):
            trans_type = 'fade_out'
        elif np.all(np.diff(pattern[window_size//2:]) < 0):
            trans_type = 'fade_in'
        elif np.mean(pattern[:window_size//2]) > np.mean(pattern[window_size//2:]):
            trans_type = 'dissolve'
        else:
            trans_type = 'other'
        
        return {
            'transition_type': trans_type,
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'difference_pattern': pattern.tolist(),
            'duration_frames': window_size * 2,
            'duration_seconds': window_size * 2 / fps
        }
    
    def analyze_scene_content(
        self,
        video_path: str,
        boundary: Dict[str, Any],
        frames_per_scene: int = 5
    ) -> Dict[str, Any]:
        """Analyze the content of scenes around boundary.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        boundary : Dict[str, Any]
            Scene boundary information
        frames_per_scene : int
            Number of frames to analyze per scene
            
        Returns
        -------
        Dict[str, Any]
            Scene content analysis results
        """
        # Load video
        video = decord.VideoReader(video_path)
        frame_idx = boundary['frame_idx']
        
        # Get frames from both scenes
        prev_indices = np.linspace(
            max(0, frame_idx - frames_per_scene),
            frame_idx - 1,
            frames_per_scene,
            dtype=int
        )
        next_indices = np.linspace(
            frame_idx,
            min(len(video) - 1, frame_idx + frames_per_scene - 1),
            frames_per_scene,
            dtype=int
        )
        
        prev_frames = video.get_batch(prev_indices).asnumpy()
        next_frames = video.get_batch(next_indices).asnumpy()
        
        # Analyze scene characteristics
        def analyze_frames(frames):
            # Color analysis
            hsv_frames = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames])
            color_mean = np.mean(hsv_frames, axis=(0,1,2))
            color_std = np.std(hsv_frames, axis=(0,1,2))
            
            # Motion analysis
            motion = 0
            for i in range(1, len(frames)):
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion += np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            motion /= len(frames) - 1
            
            # Brightness and contrast
            gray_frames = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
            brightness = np.mean(gray_frames)
            contrast = np.std(gray_frames)
            
            return {
                'color_mean': color_mean.tolist(),
                'color_std': color_std.tolist(),
                'motion': float(motion),
                'brightness': float(brightness),
                'contrast': float(contrast)
            }
        
        prev_analysis = analyze_frames(prev_frames)
        next_analysis = analyze_frames(next_frames)
        
        # Compute scene difference metrics
        color_diff = np.mean(np.abs(
            np.array(prev_analysis['color_mean']) - 
            np.array(next_analysis['color_mean'])
        ))
        motion_diff = abs(prev_analysis['motion'] - next_analysis['motion'])
        brightness_diff = abs(prev_analysis['brightness'] - next_analysis['brightness'])
        
        return {
            'previous_scene': prev_analysis,
            'next_scene': next_analysis,
            'differences': {
                'color': float(color_diff),
                'motion': float(motion_diff),
                'brightness': float(brightness_diff)
            }
        }

    def detect_scenes_with_analysis(
        self,
        video_path: str,
        methods: List[str] = ['histogram', 'optical_flow', 'deep'],
        analyze_transitions: bool = True,
        analyze_content: bool = True
    ) -> List[Dict[str, Any]]:
        """Detect scenes with detailed analysis.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        methods : List[str]
            Methods to use for detection
        analyze_transitions : bool
            Whether to analyze transition types
        analyze_content : bool
            Whether to analyze scene content
            
        Returns
        -------
        List[Dict[str, Any]]
            Detected scenes with analysis
        """
        # Detect scene boundaries
        boundaries = self.detect_scenes_combined(video_path, methods)
        
        # Add analysis if requested
        for boundary in boundaries:
            if analyze_transitions:
                boundary['transition_analysis'] = self.analyze_transition_type(
                    video_path,
                    boundary
                )
            
            if analyze_content:
                boundary['content_analysis'] = self.analyze_scene_content(
                    video_path,
                    boundary
                )
        
        return boundaries 
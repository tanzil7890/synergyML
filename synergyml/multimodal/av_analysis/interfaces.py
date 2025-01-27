"""Interface definitions for audio-visual analysis components."""

from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class AudioAnalysisResult:
    """Container for audio analysis results."""
    speech_text: Optional[str]
    music_segments: List[Dict[str, Any]]
    sound_events: List[Dict[str, Any]]
    timestamps: np.ndarray
    features: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

@dataclass
class VideoAnalysisResult:
    """Container for video analysis results."""
    scenes: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    keyframes: List[Dict[str, Any]]
    timestamps: np.ndarray
    features: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

@dataclass
class MultimodalAnalysisResult:
    """Container for combined audio-visual analysis results."""
    audio: AudioAnalysisResult
    video: VideoAnalysisResult
    alignment_info: Dict[str, Any]
    cross_modal_features: Dict[str, np.ndarray]
    understanding: Dict[str, Any]

class AudioAnalyzer(Protocol):
    """Protocol for audio analysis components."""
    
    def analyze_speech(
        self,
        audio_path: str,
        chunk_size: int = 30
    ) -> Dict[str, Any]:
        """Analyze speech content."""
        ...
    
    def analyze_music(
        self,
        audio_path: str,
        chunk_size: int = 30
    ) -> Dict[str, Any]:
        """Analyze musical elements."""
        ...
    
    def analyze_sound_events(
        self,
        audio_path: str,
        chunk_size: int = 30
    ) -> Dict[str, Any]:
        """Analyze sound events."""
        ...

class VideoAnalyzer(Protocol):
    """Protocol for video analysis components."""
    
    def analyze_scenes(
        self,
        video_path: str,
        frame_chunk: int = 90
    ) -> Dict[str, Any]:
        """Analyze scene content."""
        ...
    
    def analyze_actions(
        self,
        video_path: str,
        frame_chunk: int = 90
    ) -> Dict[str, Any]:
        """Analyze actions."""
        ...
    
    def extract_keyframes(
        self,
        video_path: str,
        method: str = 'uniform'
    ) -> Dict[str, Any]:
        """Extract key frames."""
        ...

class MultimodalAnalyzer(Protocol):
    """Protocol for multimodal analysis."""
    
    def align_modalities(
        self,
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult
    ) -> Dict[str, Any]:
        """Align audio and video analysis results."""
        ...
    
    def extract_cross_modal_features(
        self,
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult
    ) -> Dict[str, np.ndarray]:
        """Extract cross-modal features."""
        ...
    
    def generate_understanding(
        self,
        audio_result: AudioAnalysisResult,
        video_result: VideoAnalysisResult,
        cross_modal_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Generate multimodal understanding."""
        ... 
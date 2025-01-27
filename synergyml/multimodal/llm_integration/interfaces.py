"""Interface definitions for emotion analysis and LLM integration."""

from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass
import numpy as np

@dataclass
class EmotionAnalysisResult:
    """Container for emotion analysis results."""
    raw_emotions: Dict[str, Dict[str, np.ndarray]]  # Modality -> Emotion -> Probabilities
    timestamps: np.ndarray
    transitions: List[Dict[str, Any]]
    coherence: float
    modality_specific: Dict[str, Any]  # Additional modality-specific metrics

@dataclass
class LLMAnalysisRequest:
    """Container for LLM analysis request."""
    emotion_data: EmotionAnalysisResult
    prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    similar_analyses: Optional[List[Dict[str, Any]]] = None

@dataclass
class LLMAnalysisResponse:
    """Container for LLM analysis response."""
    insights: str
    quality_metrics: Dict[str, float]
    execution_metrics: Dict[str, float]
    model_config: Dict[str, Any]

class EmotionAnalyzer(Protocol):
    """Protocol defining emotion analyzer interface."""
    
    def analyze_emotional_coherence(
        self,
        video_path: str,
        window_size: int = 5,
        sampling_rate: int = 16000
    ) -> EmotionAnalysisResult:
        """Analyze emotional coherence in video."""
        ...
    
    def analyze_emotion_complexity(
        self,
        aligned_emotions: Dict[str, np.ndarray],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """Analyze emotion complexity."""
        ...

class LLMAnalyzer(Protocol):
    """Protocol defining LLM analyzer interface."""
    
    def analyze_with_llm_understanding(
        self,
        request: LLMAnalysisRequest,
        time_budget: Optional[float] = None,
        quality_requirement: Optional[float] = None
    ) -> LLMAnalysisResponse:
        """Analyze emotions using LLM understanding."""
        ...
    
    def enhance_prompt(
        self,
        base_prompt: str,
        similar_analyses: List[Dict[str, Any]]
    ) -> str:
        """Enhance analysis prompt with similar analyses."""
        ...

class AnalysisCache(Protocol):
    """Protocol defining analysis cache interface."""
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results."""
        ...
    
    def put(self, key: str, value: Dict[str, Any]) -> None:
        """Cache analysis results."""
        ...
    
    def generate_key(
        self,
        video_path: str,
        prompt: Optional[str],
        model: str
    ) -> str:
        """Generate cache key."""
        ...

class MemoryManager(Protocol):
    """Protocol defining memory manager interface."""
    
    def add_memory(
        self,
        features: np.ndarray,
        data: Dict[str, Any],
        importance_score: Optional[float] = None
    ) -> int:
        """Add new memory."""
        ...
    
    def get_similar_memories(
        self,
        features: np.ndarray,
        k: int = 3,
        min_importance: Optional[float] = None,
        max_age: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get similar memories."""
        ...
    
    def forget_old_memories(self) -> int:
        """Remove old memories."""
        ...

class PipelineOptimizer(Protocol):
    """Protocol defining pipeline optimizer interface."""
    
    def optimize_pipeline(
        self,
        analysis_context: Dict[str, Any]
    ) -> tuple[Any, Dict[str, Any]]:
        """Optimize pipeline configuration."""
        ...
    
    def compute_quality_metrics(
        self,
        llm_insights: str,
        context: Dict[str, Any]
    ) -> Any:
        """Compute quality metrics."""
        ...
    
    def update_metrics(
        self,
        model_config: Any,
        context: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> None:
        """Update pipeline metrics."""
        ... 
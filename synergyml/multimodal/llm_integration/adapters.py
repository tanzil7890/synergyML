"""Adapter classes for connecting emotion analysis and LLM components."""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import hashlib
import json
import os
from datetime import datetime

from .interfaces import (
    EmotionAnalysisResult,
    LLMAnalysisRequest,
    LLMAnalysisResponse,
    EmotionAnalyzer,
    LLMAnalyzer,
    AnalysisCache,
    MemoryManager,
    PipelineOptimizer
)

class EmotionAnalyzerAdapter:
    """Adapter for emotion analyzer component."""
    
    def __init__(self, analyzer: EmotionAnalyzer):
        """Initialize adapter.
        
        Parameters
        ----------
        analyzer : EmotionAnalyzer
            Emotion analyzer instance
        """
        self.analyzer = analyzer
    
    def convert_to_analysis_result(
        self,
        raw_results: Dict[str, Any]
    ) -> EmotionAnalysisResult:
        """Convert raw analysis results to EmotionAnalysisResult.
        
        Parameters
        ----------
        raw_results : Dict[str, Any]
            Raw analysis results
            
        Returns
        -------
        EmotionAnalysisResult
            Converted results
        """
        return EmotionAnalysisResult(
            raw_emotions=raw_results['raw_emotions'],
            timestamps=raw_results['timestamps'],
            transitions=raw_results.get('transitions', []),
            coherence=raw_results.get('coherence', 0.0),
            modality_specific=raw_results.get('modality_specific', {})
        )
    
    def analyze_video(
        self,
        video_path: str,
        window_size: int = 5
    ) -> EmotionAnalysisResult:
        """Analyze video emotions.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        window_size : int
            Analysis window size
            
        Returns
        -------
        EmotionAnalysisResult
            Analysis results
        """
        raw_results = self.analyzer.analyze_emotional_coherence(
            video_path,
            window_size=window_size
        )
        return self.convert_to_analysis_result(raw_results)

class LLMAnalyzerAdapter:
    """Adapter for LLM analyzer component."""
    
    def __init__(
        self,
        analyzer: LLMAnalyzer,
        cache: Optional[AnalysisCache] = None,
        memory: Optional[MemoryManager] = None,
        optimizer: Optional[PipelineOptimizer] = None
    ):
        """Initialize adapter.
        
        Parameters
        ----------
        analyzer : LLMAnalyzer
            LLM analyzer instance
        cache : Optional[AnalysisCache]
            Analysis cache instance
        memory : Optional[MemoryManager]
            Memory manager instance
        optimizer : Optional[PipelineOptimizer]
            Pipeline optimizer instance
        """
        self.analyzer = analyzer
        self.cache = cache
        self.memory = memory
        self.optimizer = optimizer
    
    def prepare_analysis_request(
        self,
        emotion_result: EmotionAnalysisResult,
        prompt: Optional[str] = None
    ) -> LLMAnalysisRequest:
        """Prepare LLM analysis request.
        
        Parameters
        ----------
        emotion_result : EmotionAnalysisResult
            Emotion analysis results
        prompt : Optional[str]
            Analysis prompt
            
        Returns
        -------
        LLMAnalysisRequest
            Prepared request
        """
        # Get similar analyses if memory manager available
        similar_analyses = None
        if self.memory is not None:
            features = self._emotion_results_to_features(emotion_result)
            similar_analyses = self.memory.get_similar_memories(features)
        
        # Prepare context
        context = {
            'content_length': len(str(emotion_result.raw_emotions)),
            'complexity': self._estimate_complexity(emotion_result),
            'modalities': list(emotion_result.raw_emotions.keys())
        }
        
        return LLMAnalysisRequest(
            emotion_data=emotion_result,
            prompt=prompt,
            context=context,
            similar_analyses=similar_analyses
        )
    
    def analyze_emotions(
        self,
        emotion_result: EmotionAnalysisResult,
        prompt: Optional[str] = None,
        time_budget: Optional[float] = None,
        quality_requirement: Optional[float] = None
    ) -> LLMAnalysisResponse:
        """Analyze emotions using LLM.
        
        Parameters
        ----------
        emotion_result : EmotionAnalysisResult
            Emotion analysis results
        prompt : Optional[str]
            Analysis prompt
        time_budget : Optional[float]
            Time budget for analysis
        quality_requirement : Optional[float]
            Required quality score
            
        Returns
        -------
        LLMAnalysisResponse
            LLM analysis response
        """
        # Prepare request
        request = self.prepare_analysis_request(emotion_result, prompt)
        
        # Get analysis from LLM
        response = self.analyzer.analyze_with_llm_understanding(
            request,
            time_budget=time_budget,
            quality_requirement=quality_requirement
        )
        
        # Update memory if available
        if self.memory is not None:
            features = self._emotion_results_to_features(emotion_result)
            self.memory.add_memory(
                features,
                {
                    'emotion_analysis': emotion_result,
                    'llm_insights': response.insights,
                    'quality_metrics': response.quality_metrics
                }
            )
        
        return response
    
    def _emotion_results_to_features(
        self,
        results: EmotionAnalysisResult
    ) -> np.ndarray:
        """Convert emotion results to feature vector.
        
        Parameters
        ----------
        results : EmotionAnalysisResult
            Emotion analysis results
            
        Returns
        -------
        np.ndarray
            Feature vector
        """
        features = []
        
        for modality in results.raw_emotions:
            emotions = results.raw_emotions[modality]
            # Add mean and std for each emotion
            for emotion_probs in emotions.values():
                features.extend([
                    np.mean(emotion_probs),
                    np.std(emotion_probs)
                ])
        
        return np.array(features)
    
    def _estimate_complexity(
        self,
        results: EmotionAnalysisResult
    ) -> float:
        """Estimate analysis complexity.
        
        Parameters
        ----------
        results : EmotionAnalysisResult
            Emotion analysis results
            
        Returns
        -------
        float
            Complexity score
        """
        # Consider factors like:
        # 1. Emotion variance
        variances = []
        for modality in results.raw_emotions:
            for emotion_probs in results.raw_emotions[modality].values():
                variances.append(np.var(emotion_probs))
        variance_score = np.mean(variances)
        
        # 2. Number of transitions
        transition_score = min(1.0, len(results.transitions) / 20)
        
        # 3. Coherence (lower coherence = higher complexity)
        coherence_score = 1 - results.coherence
        
        # Combine scores
        complexity = np.mean([
            variance_score,
            transition_score,
            coherence_score
        ])
        
        return min(1.0, max(0.0, complexity))

class AnalysisCacheAdapter(AnalysisCache):
    """Adapter for analysis cache."""
    
    def __init__(self, cache_dir: str):
        """Initialize adapter.
        
        Parameters
        ----------
        cache_dir : str
            Cache directory path
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def put(self, key: str, value: Dict[str, Any]) -> None:
        """Cache analysis results."""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_file, 'w') as f:
            json.dump(value, f)
    
    def generate_key(
        self,
        video_path: str,
        prompt: Optional[str],
        model: str
    ) -> str:
        """Generate cache key."""
        key_components = [
            video_path,
            str(prompt),
            model
        ]
        return hashlib.md5("".join(key_components).encode()).hexdigest() 
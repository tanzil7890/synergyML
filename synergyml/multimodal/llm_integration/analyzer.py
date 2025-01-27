"""LLM-enhanced multimodal analysis module."""

from typing import Dict, Any, Optional, List, Union
from synergyml.multimodal.emotion import EmotionAnalyzer
from synergyml.llm.base import BaseTextCompletionMixin
from synergyml.llm.gpt.mixin import GPTTextCompletionMixin
from synergyml.llm.vertex.mixin import VertexTextCompletionMixin
from synergyml.config import SynergyMLConfig
from .memory_manager import AdvancedMemoryManager
from .visualization import LLMVisualizerMixin
from .pipeline_optimizer import PipelineOptimizer, ModelTier
from .interfaces import (
    EmotionAnalysisResult,
    LLMAnalysisRequest,
    LLMAnalysisResponse,
    LLMAnalyzer
)
from .adapters import (
    EmotionAnalyzerAdapter,
    LLMAnalyzerAdapter,
    AnalysisCacheAdapter
)
import numpy as np
import json
import hashlib
from datetime import datetime
import os
import time

class LLMEnhancedAnalyzer(LLMVisualizerMixin, LLMAnalyzer):
    """Analyzer that combines multimodal analysis with LLM understanding."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        use_gpu: bool = False,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        memory_backend: str = "annoy",
        cache_dir: Optional[str] = None,
        llm_backend: str = "gpt",
        max_memories: int = 1000,
        memory_lifetime: int = 30,
        importance_threshold: float = 0.3,
        default_model_tier: ModelTier = ModelTier.BALANCED
    ):
        """Initialize analyzer.
        
        Parameters
        ----------
        model : str
            LLM model identifier
        use_gpu : bool
            Whether to use GPU acceleration
        openai_key : Optional[str]
            OpenAI API key
        openai_org : Optional[str]
            OpenAI organization ID
        memory_backend : str
            Memory backend type
        cache_dir : Optional[str]
            Cache directory path
        llm_backend : str
            LLM backend type
        max_memories : int
            Maximum number of memories
        memory_lifetime : int
            Memory lifetime in days
        importance_threshold : float
            Memory importance threshold
        default_model_tier : ModelTier
            Default model tier
        """
        # Initialize LLM backend
        self.llm_backend = llm_backend
        if llm_backend == "gpt":
            self.llm_mixin = GPTTextCompletionMixin()
            self.llm_mixin._set_keys(openai_key, openai_org)
        elif llm_backend == "vertex":
            self.llm_mixin = VertexTextCompletionMixin()
        elif llm_backend == "local":
            # For local models like llama.cpp
            if not model.startswith("gguf::"):
                model = f"gguf::{model}"
            self.llm_mixin = GPTTextCompletionMixin()
            
        self.model = model
        
        # Initialize components
        self.emotion_analyzer = EmotionAnalyzer(use_gpu=use_gpu)
        self.emotion_adapter = EmotionAnalyzerAdapter(self.emotion_analyzer)
        
        self.cache_dir = cache_dir or SynergyMLConfig.get_cache_dir()
        self.cache_adapter = AnalysisCacheAdapter(self.cache_dir)
        
        # Initialize memory manager
        self.memory_manager = AdvancedMemoryManager(
            memory_backend=memory_backend,
            max_memories=max_memories,
            memory_lifetime=memory_lifetime,
            importance_threshold=importance_threshold
        )
        
        # Initialize pipeline optimizer
        self.pipeline_optimizer = PipelineOptimizer(
            cache_dir=self.cache_dir,
            default_tier=default_model_tier
        )
        
        # Initialize LLM adapter
        self.llm_adapter = LLMAnalyzerAdapter(
            analyzer=self,
            cache=self.cache_adapter,
            memory=self.memory_manager,
            optimizer=self.pipeline_optimizer
        )
        
        # Load memory metadata if exists
        metadata_path = os.path.join(self.cache_dir, "memory_metadata.json")
        if os.path.exists(metadata_path):
            self.memory_manager.load_metadata(metadata_path)
    
    def analyze_with_llm_understanding(
        self,
        request: LLMAnalysisRequest,
        time_budget: Optional[float] = None,
        quality_requirement: Optional[float] = None
    ) -> LLMAnalysisResponse:
        """Analyze emotions using LLM understanding.
        
        Parameters
        ----------
        request : LLMAnalysisRequest
            Analysis request
        time_budget : Optional[float]
            Time budget for analysis
        quality_requirement : Optional[float]
            Required quality score
            
        Returns
        -------
        LLMAnalysisResponse
            Analysis response
        """
        start_time = time.time()
        
        # Get optimized model configuration
        model_config, params = self.pipeline_optimizer.optimize_pipeline(
            request.context or {}
        )
        
        # Prepare prompt
        if request.prompt is None:
            request.prompt = self._get_default_analysis_prompt(request.emotion_data)
            
        if request.similar_analyses:
            request.prompt = self.enhance_prompt(
                request.prompt,
                request.similar_analyses
            )
            
        # Get LLM insights
        llm_insights = self.llm_mixin._get_chat_completion(
            model=model_config.model_name,
            messages=request.prompt,
            system_message="You are an expert in analyzing emotional patterns and human behavior.",
            **params
        )
        
        # Convert insights to string
        insights_str = self.llm_mixin._convert_completion_to_str(llm_insights)
        
        # Compute quality metrics
        quality_metrics = self.pipeline_optimizer.compute_quality_metrics(
            insights_str,
            request.context or {}
        )
        
        # Prepare execution metrics
        execution_metrics = {
            'execution_time': time.time() - start_time,
            'token_count': len(insights_str) / 4  # Rough estimate
        }
        
        # Update pipeline metrics
        self.pipeline_optimizer.update_metrics(
            model_config,
            request.context or {},
            {
                **execution_metrics,
                'quality_metrics': quality_metrics
            }
        )
        
        return LLMAnalysisResponse(
            insights=insights_str,
            quality_metrics=quality_metrics.__dict__,
            execution_metrics=execution_metrics,
            model_config={
                'model': model_config.model_name,
                'parameters': params
            }
        )
    
    def enhance_prompt(
        self,
        base_prompt: str,
        similar_analyses: List[Dict[str, Any]]
    ) -> str:
        """Enhance analysis prompt with similar analyses.
        
        Parameters
        ----------
        base_prompt : str
            Base analysis prompt
        similar_analyses : List[Dict[str, Any]]
            Similar analyses
            
        Returns
        -------
        str
            Enhanced prompt
        """
        prompt = base_prompt + "\n\nBased on similar analyses, consider these patterns:\n"
        
        for idx, analysis in enumerate(similar_analyses, 1):
            prompt += f"\nPattern {idx} (Importance: {analysis.get('importance', 'N/A')}):\n"
            prompt += f"{analysis['llm_insights']}\n"
            
        prompt += "\nUse these patterns to inform your analysis while focusing on the unique aspects of the current video."
        
        return prompt
    
    def _get_default_analysis_prompt(
        self,
        emotion_data: EmotionAnalysisResult
    ) -> str:
        """Get default prompt for LLM analysis.
        
        Parameters
        ----------
        emotion_data : EmotionAnalysisResult
            Emotion analysis results
            
        Returns
        -------
        str
            Default analysis prompt
        """
        prompt = f"""Analyze the following emotional patterns detected in the video:

Audio Emotions:
{emotion_data.raw_emotions.get('audio', {})}

Video Emotions:
{emotion_data.raw_emotions.get('video', {})}

Please provide insights on:
1. The overall emotional narrative
2. Key emotional transitions and their significance
3. The relationship between audio and visual emotions
4. Potential underlying psychological patterns
5. Recommendations for emotional engagement

Format your response in a clear, structured way."""
        
        return prompt 
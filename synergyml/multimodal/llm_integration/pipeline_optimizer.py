"""Pipeline optimization for LLM-enhanced analysis."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from datetime import datetime
import json
import os
from dataclasses import dataclass
from enum import Enum
from .pipeline_visualization import PipelineVisualizerMixin

class ModelTier(Enum):
    """Model tiers for different analysis needs."""
    FAST = "fast"      # Quick, basic analysis
    BALANCED = "balanced"  # Good balance of speed/quality
    QUALITY = "quality"    # Highest quality analysis

@dataclass
class ModelConfig:
    """Configuration for model pipeline."""
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    
@dataclass
class QualityMetrics:
    """Detailed quality metrics for analysis results."""
    comprehensiveness: float  # Coverage of analysis aspects
    coherence: float         # Logical flow and consistency
    insight_depth: float     # Depth of analysis
    relevance: float         # Relevance to context
    clarity: float           # Clarity of expression
    
    @property
    def overall_score(self) -> float:
        """Compute overall quality score."""
        weights = {
            'comprehensiveness': 0.25,
            'coherence': 0.2,
            'insight_depth': 0.25,
            'relevance': 0.2,
            'clarity': 0.1
        }
        
        return sum(
            getattr(self, metric) * weight
            for metric, weight in weights.items()
        )
    
class PipelineOptimizer(PipelineVisualizerMixin):
    """Optimizer for LLM analysis pipeline."""
    
    # Model configurations for different tiers
    MODEL_CONFIGS = {
        ModelTier.FAST: ModelConfig(
            model_name="gpt-3.5-turbo",
            max_tokens=500,
            temperature=0.3,
            top_p=0.9,
            presence_penalty=0.0,
            frequency_penalty=0.0
        ),
        ModelTier.BALANCED: ModelConfig(
            model_name="gpt-4",
            max_tokens=1000,
            temperature=0.5,
            top_p=0.95,
            presence_penalty=0.1,
            frequency_penalty=0.1
        ),
        ModelTier.QUALITY: ModelConfig(
            model_name="gpt-4-turbo",
            max_tokens=2000,
            temperature=0.7,
            top_p=1.0,
            presence_penalty=0.2,
            frequency_penalty=0.2
        )
    }
    
    # Quality assessment criteria
    QUALITY_INDICATORS = {
        'comprehensiveness': [
            'overall', 'comprehensive', 'complete', 'thorough', 'detailed',
            'extensive', 'broad', 'wide-ranging', 'in-depth'
        ],
        'coherence': [
            'therefore', 'because', 'consequently', 'thus', 'hence',
            'as a result', 'due to', 'leads to', 'follows from'
        ],
        'insight_depth': [
            'suggests', 'indicates', 'implies', 'reveals', 'demonstrates',
            'signifies', 'points to', 'highlights', 'underscores'
        ],
        'relevance': [
            'context', 'specifically', 'particularly', 'notably',
            'importantly', 'significantly', 'relevant', 'pertinent'
        ],
        'clarity': [
            'clear', 'precise', 'specific', 'explicit', 'defined',
            'outlined', 'structured', 'organized', 'systematic'
        ]
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_tier: ModelTier = ModelTier.BALANCED
    ):
        """Initialize pipeline optimizer.
        
        Parameters
        ----------
        cache_dir : Optional[str]
            Directory for caching optimization results
        default_tier : ModelTier
            Default model tier to use
        """
        self.cache_dir = cache_dir
        self.default_tier = default_tier
        self.performance_metrics: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load cached metrics if available
        if cache_dir:
            metrics_path = os.path.join(cache_dir, "pipeline_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.performance_metrics = json.load(f)
                    
    def optimize_pipeline(
        self,
        analysis_context: Dict[str, Any]
    ) -> Tuple[ModelConfig, Dict[str, Any]]:
        """Optimize pipeline based on analysis context.
        
        Parameters
        ----------
        analysis_context : Dict[str, Any]
            Context information for optimization including:
            - content_length: Length of content to analyze
            - complexity: Estimated analysis complexity
            - time_budget: Available time for analysis
            - quality_requirement: Required quality level
            
        Returns
        -------
        Tuple[ModelConfig, Dict[str, Any]]
            Optimized model configuration and additional parameters
        """
        # Determine appropriate model tier
        model_tier = self._select_model_tier(analysis_context)
        model_config = self.MODEL_CONFIGS[model_tier]
        
        # Optimize parameters based on context
        params = self._optimize_parameters(model_config, analysis_context)
        
        return model_config, params
    
    def compute_quality_metrics(
        self,
        llm_insights: str,
        context: Dict[str, Any]
    ) -> QualityMetrics:
        """Compute detailed quality metrics for analysis results.
        
        Parameters
        ----------
        llm_insights : str
            LLM analysis results
        context : Dict[str, Any]
            Analysis context
            
        Returns
        -------
        QualityMetrics
            Detailed quality metrics
        """
        # Normalize text for analysis
        text = llm_insights.lower()
        
        # Compute metrics
        metrics = {}
        for aspect, indicators in self.QUALITY_INDICATORS.items():
            # Count occurrences of quality indicators
            indicator_counts = [text.count(ind.lower()) for ind in indicators]
            
            # Normalize score
            max_expected = len(indicators)  # Adjust based on text length if needed
            metrics[aspect] = min(1.0, sum(indicator_counts) / max_expected)
            
        # Adjust metrics based on context
        self._adjust_metrics_by_context(metrics, context)
        
        return QualityMetrics(**metrics)
    
    def _adjust_metrics_by_context(
        self,
        metrics: Dict[str, float],
        context: Dict[str, Any]
    ) -> None:
        """Adjust quality metrics based on analysis context."""
        # Adjust based on content complexity
        complexity = context.get('complexity', 0.5)
        if complexity > 0.7:
            # For complex content, we expect more depth and coherence
            metrics['insight_depth'] *= 1.2
            metrics['coherence'] *= 1.1
        elif complexity < 0.3:
            # For simple content, we emphasize clarity and relevance
            metrics['clarity'] *= 1.2
            metrics['relevance'] *= 1.1
            
        # Adjust based on quality requirement
        quality_req = context.get('quality_requirement', 0.5)
        if quality_req > 0.8:
            # For high quality requirements, boost comprehensiveness
            metrics['comprehensiveness'] *= 1.2
            
        # Normalize all metrics to [0, 1]
        for key in metrics:
            metrics[key] = min(1.0, metrics[key])
    
    def update_metrics(
        self,
        model_config: ModelConfig,
        context: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> None:
        """Update performance metrics after analysis.
        
        Parameters
        ----------
        model_config : ModelConfig
            Used model configuration
        context : Dict[str, Any]
            Analysis context
        metrics : Dict[str, Any]
            Performance metrics including:
            - execution_time: Analysis execution time
            - token_count: Number of tokens used
            - quality_metrics: QualityMetrics instance
        """
        if model_config.model_name not in self.performance_metrics:
            self.performance_metrics[model_config.model_name] = []
            
        # Add new metrics
        self.performance_metrics[model_config.model_name].append({
            'context': context,
            'metrics': {
                **metrics,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        # Save updated metrics
        if self.cache_dir:
            metrics_path = os.path.join(self.cache_dir, "pipeline_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f)
                
    def _select_model_tier(self, context: Dict[str, Any]) -> ModelTier:
        """Select appropriate model tier based on context."""
        # Get context parameters with defaults
        content_length = context.get('content_length', 1000)
        complexity = context.get('complexity', 0.5)
        time_budget = context.get('time_budget', float('inf'))
        quality_req = context.get('quality_requirement', 0.5)
        
        # Score different aspects
        speed_score = self._compute_speed_score(content_length, time_budget)
        quality_score = self._compute_quality_score(complexity, quality_req)
        
        # Select tier based on scores
        if quality_score > 0.8:
            return ModelTier.QUALITY
        elif speed_score > 0.8:
            return ModelTier.FAST
        return ModelTier.BALANCED
    
    def _optimize_parameters(
        self,
        model_config: ModelConfig,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize model parameters based on context and history."""
        params = {
            'max_tokens': model_config.max_tokens,
            'temperature': model_config.temperature,
            'top_p': model_config.top_p,
            'presence_penalty': model_config.presence_penalty,
            'frequency_penalty': model_config.frequency_penalty
        }
        
        # Adjust based on content complexity
        complexity = context.get('complexity', 0.5)
        if complexity > 0.7:
            params['temperature'] = min(params['temperature'] * 1.2, 1.0)
            params['top_p'] = min(params['top_p'] * 1.1, 1.0)
        elif complexity < 0.3:
            params['temperature'] *= 0.8
            
        # Adjust based on historical performance
        if model_config.model_name in self.performance_metrics:
            recent_metrics = self.performance_metrics[model_config.model_name][-10:]
            avg_quality = np.mean([
                m['metrics']['quality_metrics'].overall_score 
                for m in recent_metrics
            ])
            
            if avg_quality < 0.6:
                params['temperature'] *= 0.9
                params['presence_penalty'] += 0.1
                
        return params
    
    def _compute_speed_score(
        self,
        content_length: int,
        time_budget: float
    ) -> float:
        """Compute speed requirement score."""
        if time_budget == float('inf'):
            return 0.5
            
        # Estimate tokens per second needed
        tokens_per_second = content_length / time_budget
        
        # Normalize score (adjust thresholds as needed)
        return min(1.0, max(0.0, tokens_per_second / 100))
    
    def _compute_quality_score(
        self,
        complexity: float,
        quality_requirement: float
    ) -> float:
        """Compute quality requirement score."""
        # Combine complexity and quality requirement
        return (complexity + quality_requirement) / 2 
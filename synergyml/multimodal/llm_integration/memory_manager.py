"""Advanced memory management for LLM-enhanced analysis."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from synergyml.memory.base import BaseMemory
from synergyml.memory._annoy import AnnoyMemory
from synergyml.memory._sklearn_nn import SklearnMemory
import json
import os

class AdvancedMemoryManager:
    """Advanced memory management with forgetting and importance weighting."""
    
    def __init__(
        self,
        memory_backend: str = "annoy",
        dimension: int = 1536,
        max_memories: int = 1000,
        memory_lifetime: int = 30,  # days
        importance_threshold: float = 0.3
    ):
        """Initialize memory manager.
        
        Parameters
        ----------
        memory_backend : str
            Type of memory backend ('annoy' or 'sklearn')
        dimension : int
            Dimension of feature vectors
        max_memories : int
            Maximum number of memories to store
        memory_lifetime : int
            Number of days to keep memories
        importance_threshold : float
            Minimum importance score to keep memory
        """
        self.memory_backend = memory_backend
        self.max_memories = max_memories
        self.memory_lifetime = timedelta(days=memory_lifetime)
        self.importance_threshold = importance_threshold
        
        # Initialize memory backend
        if memory_backend == "annoy":
            self.memory = AnnoyMemory(dimension=dimension)
        else:
            self.memory = SklearnMemory()
            
        # Additional metadata storage
        self.metadata: Dict[int, Dict[str, Any]] = {}
        
    def add_memory(
        self,
        features: np.ndarray,
        data: Dict[str, Any],
        importance_score: Optional[float] = None
    ) -> int:
        """Add new memory with metadata.
        
        Parameters
        ----------
        features : np.ndarray
            Feature vector
        data : Dict[str, Any]
            Associated data
        importance_score : Optional[float]
            Manual importance score
            
        Returns
        -------
        int
            Memory index
        """
        # Compute importance if not provided
        if importance_score is None:
            importance_score = self._compute_importance(features, data)
            
        # Add timestamp and importance
        timestamp = datetime.now()
        metadata = {
            'timestamp': timestamp.isoformat(),
            'importance': importance_score,
            'access_count': 0,
            'last_access': timestamp.isoformat()
        }
        
        # Add to memory
        idx = self.memory.add(features, {**data, **metadata})
        self.metadata[idx] = metadata
        
        # Manage memory size
        self._manage_memory_size()
        
        return idx
        
    def get_similar_memories(
        self,
        features: np.ndarray,
        k: int = 3,
        min_importance: Optional[float] = None,
        max_age: Optional[int] = None  # days
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Get similar memories with filtering.
        
        Parameters
        ----------
        features : np.ndarray
            Query feature vector
        k : int
            Number of neighbors
        min_importance : Optional[float]
            Minimum importance score
        max_age : Optional[int]
            Maximum age in days
            
        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            Similar memories with similarity scores
        """
        # Get initial neighbors
        indices = self.memory.get_nearest_neighbors(features, k=k*2)  # Get more for filtering
        
        # Filter and sort memories
        filtered_memories = []
        for idx in indices:
            metadata = self.metadata[idx]
            memory_data = self.memory.get_data(idx)
            
            # Check filters
            if min_importance and metadata['importance'] < min_importance:
                continue
                
            if max_age:
                age = (datetime.now() - datetime.fromisoformat(metadata['timestamp'])).days
                if age > max_age:
                    continue
            
            # Compute similarity
            similarity = self._compute_similarity(features, self.memory.get_vector(idx))
            filtered_memories.append((memory_data, similarity))
            
            # Update access metadata
            self._update_access_metadata(idx)
            
        # Sort by similarity and return top k
        filtered_memories.sort(key=lambda x: x[1], reverse=True)
        return filtered_memories[:k]
    
    def forget_old_memories(self) -> int:
        """Remove old and unimportant memories.
        
        Returns
        -------
        int
            Number of memories removed
        """
        current_time = datetime.now()
        removed_count = 0
        
        # Identify memories to remove
        to_remove = []
        for idx, metadata in self.metadata.items():
            # Check age
            age = current_time - datetime.fromisoformat(metadata['timestamp'])
            if age > self.memory_lifetime:
                to_remove.append(idx)
                continue
                
            # Check importance and access
            if (metadata['importance'] < self.importance_threshold and
                metadata['access_count'] < 3):
                to_remove.append(idx)
                
        # Remove identified memories
        for idx in to_remove:
            self.memory.remove(idx)
            del self.metadata[idx]
            removed_count += 1
            
        return removed_count
    
    def _compute_importance(
        self,
        features: np.ndarray,
        data: Dict[str, Any]
    ) -> float:
        """Compute importance score for memory.
        
        Uses factors like:
        - Emotional intensity
        - Pattern uniqueness
        - Information content
        """
        importance_factors = []
        
        # Emotional intensity
        if 'emotion_analysis' in data:
            emotions = data['emotion_analysis']['raw_emotions']
            audio_intensity = np.max(np.abs(emotions['audio']))
            video_intensity = np.max(np.abs(emotions['video']))
            importance_factors.append((audio_intensity + video_intensity) / 2)
            
        # Pattern uniqueness (compare with existing memories)
        if len(self.memory) > 0:
            similarities = []
            for idx in range(len(self.memory)):
                sim = self._compute_similarity(features, self.memory.get_vector(idx))
                similarities.append(sim)
            uniqueness = 1 - np.mean(similarities)
            importance_factors.append(uniqueness)
            
        # Information content (based on feature variance)
        feature_variance = np.var(features)
        importance_factors.append(min(feature_variance, 1.0))
        
        # Combine factors
        if importance_factors:
            return np.mean(importance_factors)
        return 0.5
    
    def _compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """Compute similarity between feature vectors."""
        return 1 - np.mean(np.abs(features1 - features2))
    
    def _update_access_metadata(self, idx: int) -> None:
        """Update memory access metadata."""
        metadata = self.metadata[idx]
        metadata['access_count'] += 1
        metadata['last_access'] = datetime.now().isoformat()
        
        # Update importance based on access pattern
        time_factor = 1.0
        if metadata['access_count'] > 1:
            days_since_first = (datetime.now() - 
                              datetime.fromisoformat(metadata['timestamp'])).days
            time_factor = np.exp(-days_since_first / 30)  # Decay over time
            
        metadata['importance'] *= (1 + 0.1 * time_factor)  # Increase importance with use
        
    def _manage_memory_size(self) -> None:
        """Manage memory size by removing least important memories."""
        if len(self.memory) > self.max_memories:
            # Sort memories by importance
            sorted_memories = sorted(
                self.metadata.items(),
                key=lambda x: x[1]['importance']
            )
            
            # Remove least important memories
            to_remove = len(self.memory) - self.max_memories
            for idx, _ in sorted_memories[:to_remove]:
                self.memory.remove(idx)
                del self.metadata[idx]
                
    def save_metadata(self, path: str) -> None:
        """Save memory metadata to file."""
        with open(path, 'w') as f:
            json.dump(self.metadata, f)
            
    def load_metadata(self, path: str) -> None:
        """Load memory metadata from file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.metadata = json.load(f) 
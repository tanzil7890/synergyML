"""Utility functions for audio-visual analysis."""

import os
import json
import hashlib
from typing import Dict, Any, Optional
import torch
import numpy as np

class ModelCache:
    """Cache manager for model outputs."""
    
    def __init__(self, cache_dir: str):
        """Initialize cache manager.
        
        Parameters
        ----------
        cache_dir : str
            Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.model_cache_dir = os.path.join(cache_dir, 'model_outputs')
        os.makedirs(self.model_cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str) -> str:
        """Get cache file path for key."""
        return os.path.join(self.model_cache_dir, f"{key}.npz")
    
    def generate_key(
        self,
        model_name: str,
        audio_path: str,
        chunk_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key from parameters.
        
        Parameters
        ----------
        model_name : str
            Name/ID of the model
        audio_path : str
            Path to audio file
        chunk_params : Optional[Dict[str, Any]]
            Chunking parameters
            
        Returns
        -------
        str
            Cache key
        """
        # Get audio file hash
        with open(audio_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Combine parameters
        key_parts = [
            model_name,
            file_hash
        ]
        
        if chunk_params:
            key_parts.append(json.dumps(chunk_params, sort_keys=True))
        
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()
    
    def save(
        self,
        key: str,
        data: Dict[str, Any]
    ) -> None:
        """Save data to cache.
        
        Parameters
        ----------
        key : str
            Cache key
        data : Dict[str, Any]
            Data to cache
        """
        cache_path = self.get_cache_path(key)
        
        # Convert torch tensors to numpy
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                processed_data[k] = v.cpu().numpy()
            elif isinstance(v, np.ndarray):
                processed_data[k] = v
            else:
                processed_data[k] = v
        
        np.savez_compressed(cache_path, **processed_data)
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Cached data if exists, None otherwise
        """
        cache_path = self.get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                return dict(data)
        except Exception:
            return None

def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_size: int,
    overlap: float = 0.5
) -> tuple[list, np.ndarray]:
    """Split audio into overlapping chunks.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sr : int
        Sample rate
    chunk_size : int
        Chunk size in seconds
    overlap : float
        Overlap between chunks (0-1)
        
    Returns
    -------
    tuple[list, np.ndarray]
        List of chunks and array of timestamps
    """
    chunk_samples = chunk_size * sr
    overlap_samples = int(chunk_samples * overlap)
    
    chunks = []
    timestamps = []
    
    for i in range(0, len(audio), chunk_samples - overlap_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
        timestamps.append(i / sr)
    
    return chunks, np.array(timestamps) 
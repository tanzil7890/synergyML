"""Tests for audio analysis components."""

import os
import pytest
import numpy as np
import soundfile as sf
from ..audio import EnhancedAudioAnalyzer
from ..utils import ModelCache, chunk_audio
import torch

@pytest.fixture
def test_audio_path(tmp_path):
    """Create a test audio file."""
    duration = 5  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a simple sine wave
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Save to temporary file
    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, audio, sample_rate)
    
    return str(audio_path)

@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory."""
    return str(tmp_path / "cache")

@pytest.fixture
def analyzer(cache_dir):
    """Create an audio analyzer instance."""
    return EnhancedAudioAnalyzer(cache_dir=cache_dir)

def test_audio_chunking():
    """Test audio chunking utility."""
    # Create sample audio
    sr = 16000
    duration = 10
    audio = np.random.randn(sr * duration)
    
    # Test chunking
    chunk_size = 2
    overlap = 0.5
    chunks, timestamps = chunk_audio(audio, sr, chunk_size, overlap)
    
    # Check results
    assert len(chunks) > 0
    assert len(timestamps) == len(chunks)
    assert chunks[0].shape[0] == chunk_size * sr

def test_model_cache(test_audio_path, cache_dir):
    """Test model cache functionality."""
    cache = ModelCache(cache_dir)
    
    # Test key generation
    key = cache.generate_key(
        "test_model",
        test_audio_path,
        {"chunk_size": 30}
    )
    assert isinstance(key, str)
    
    # Test saving and loading
    test_data = {
        "array": np.array([1, 2, 3]),
        "tensor": torch.tensor([4, 5, 6]),
        "metadata": {"test": "value"}
    }
    
    cache.save(key, test_data)
    loaded_data = cache.load(key)
    
    assert loaded_data is not None
    assert "array" in loaded_data
    assert "metadata" in loaded_data
    np.testing.assert_array_equal(loaded_data["array"], test_data["array"])

def test_speech_analysis(analyzer, test_audio_path):
    """Test speech analysis functionality."""
    results = analyzer.analyze_speech(test_audio_path)
    
    assert "segments" in results
    assert "timestamps" in results
    assert "sample_rate" in results
    assert isinstance(results["timestamps"], np.ndarray)
    assert len(results["segments"]) > 0

def test_sound_events_analysis(analyzer, test_audio_path):
    """Test sound event analysis functionality."""
    results = analyzer.analyze_sound_events(test_audio_path)
    
    assert "events" in results
    assert "timestamps" in results
    assert "features" in results
    assert isinstance(results["events"], list)
    assert len(results["events"]) > 0
    assert isinstance(results["features"], np.ndarray)

def test_music_analysis(analyzer, test_audio_path):
    """Test music analysis functionality."""
    results = analyzer.analyze_music(test_audio_path)
    
    assert "segments" in results
    assert "timestamps" in results
    assert "features" in results
    assert isinstance(results["segments"], list)
    assert len(results["segments"]) > 0
    assert "chroma" in results["features"]
    assert "mfcc" in results["features"]

def test_complete_analysis(analyzer, test_audio_path):
    """Test complete audio analysis pipeline."""
    results = analyzer.analyze(test_audio_path)
    
    assert results.speech_text is not None
    assert isinstance(results.music_segments, list)
    assert isinstance(results.sound_events, list)
    assert isinstance(results.timestamps, np.ndarray)
    assert isinstance(results.features, dict)
    assert "music" in results.features
    assert "sound" in results.features
    assert isinstance(results.metadata, dict)
    assert "sample_rate" in results.metadata
    assert "duration" in results.metadata 
# Audio-Visual Analysis Components

This module provides comprehensive audio and video analysis capabilities for the SynergyML package.

## Installation

Ensure you have all required dependencies:

```bash
pip install -r requirements.txt
```

## Audio Analysis

The `EnhancedAudioAnalyzer` provides three main types of analysis:

1. Speech Recognition (using Whisper)
2. Sound Event Detection (using AST)
3. Music Analysis (using librosa)

### Basic Usage

```python
from synergyml.multimodal.av_analysis.audio import EnhancedAudioAnalyzer

# Initialize analyzer
analyzer = EnhancedAudioAnalyzer(
    device='cuda',  # or 'cpu'
    cache_dir='path/to/cache'  # optional
)

# Analyze a single audio file
results = analyzer.analyze('path/to/audio.wav')

# Access results
print(f"Transcribed text: {results.speech_text}")
print(f"Detected sound events: {results.sound_events}")
print(f"Music segments: {results.music_segments}")
```

### Individual Analysis

You can also perform individual analyses:

```python
# Speech recognition only
speech_results = analyzer.analyze_speech('path/to/audio.wav')

# Sound event detection
sound_results = analyzer.analyze_sound_events('path/to/audio.wav')

# Music analysis
music_results = analyzer.analyze_music('path/to/audio.wav')
```

### Using Model Cache

The analyzer includes built-in model output caching:

```python
from synergyml.multimodal.av_analysis.utils import ModelCache

# Initialize cache
cache = ModelCache('path/to/cache')

# Generate cache key
key = cache.generate_key(
    model_name='whisper-large-v3',
    audio_path='path/to/audio.wav',
    chunk_params={'chunk_size': 30}
)

# Try to load from cache
cached_results = cache.load(key)
if cached_results is None:
    # Perform analysis
    results = analyzer.analyze_speech('path/to/audio.wav')
    # Save to cache
    cache.save(key, results)
```

## Output Formats

### Speech Analysis Results
```python
{
    'segments': [
        {
            'text': str,          # Transcribed text
            'start_time': float,  # Start time in seconds
            'end_time': float     # End time in seconds
        }
    ],
    'timestamps': np.ndarray,     # Segment timestamps
    'sample_rate': int           # Audio sample rate
}
```

### Sound Event Results
```python
{
    'events': [
        {
            'events': [
                {
                    'label': str,        # Event label
                    'probability': float  # Confidence score
                }
            ],
            'start_time': float,
            'end_time': float
        }
    ],
    'timestamps': np.ndarray,
    'features': np.ndarray       # Event detection features
}
```

### Music Analysis Results
```python
{
    'segments': [
        {
            'tempo': float,           # Tempo in BPM
            'beat_frames': List[int],  # Beat positions
            'start_time': float,
            'end_time': float
        }
    ],
    'timestamps': np.ndarray,
    'features': {
        'chroma': np.ndarray,    # Chromagram
        'mfcc': np.ndarray       # MFCCs
    }
}
```

## Testing

Run the test suite:

```bash
pytest synergyml/multimodal/av_analysis/tests/
```

## Configuration

Model and pipeline configurations can be modified in `config.py`:

```python
MODEL_CONFIG = {
    'audio': {
        'speech': 'openai/whisper-large-v3',
        'sound': 'microsoft/audio-spectrogram-transformer'
    }
}

PIPELINE_CONFIG = {
    'audio': {
        'sample_rate': 16000,
        'chunk_size': 30,
        'overlap': 0.5
    }
}
``` 
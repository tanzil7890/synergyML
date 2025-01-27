"""Configuration for audio-visual analysis components."""

from typing import Dict, Any

# Model configurations for different analysis components
MODEL_CONFIG = {
    'audio': {
        'speech': 'openai/whisper-large-v3',
        'music': 'facebook/musicgen-large',
        'sound': 'microsoft/audio-spectrogram-transformer'
    },
    'video': {
        'scene': 'microsoft/videomae-base',
        'action': 'facebook/timesformer-base-256'
    },
    'multimodal': {
        'understanding': 'openai/gpt-4-vision-preview',
        'embedding': 'laion/CLAP-ViT-L-14'
    }
}

# Package requirements for audio-visual analysis
REQUIREMENTS = {
    'audio': [
        'librosa>=0.10.0',
        'soundfile>=0.12.1',
        'torchaudio>=2.0.0',
        'whisper>=1.0.0',
        'asteroid>=0.6.1'
    ],
    'video': [
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0'
    ],
    'common': [
        'numpy>=1.21.0',
        'scipy>=1.7.0'
    ]
}

# Analysis pipeline configurations
PIPELINE_CONFIG = {
    'audio': {
        'sample_rate': 16000,
        'chunk_size': 30,  # seconds
        'overlap': 0.5     # 50% overlap between chunks
    },
    'video': {
        'fps': 30,
        'frame_chunk': 90,  # frames per chunk
        'overlap_frames': 15
    }
} 
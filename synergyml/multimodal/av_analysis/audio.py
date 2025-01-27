"""Audio analysis implementation using Whisper, MusicGen, and AST."""

import os
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoFeatureExtractor, 
    AutoModelForAudioClassification
)
from .interfaces import AudioAnalyzer, AudioAnalysisResult
from .config import MODEL_CONFIG, PIPELINE_CONFIG
from .audio_features import (
    SpectralFeatures,
    EmotionFeatures,
    AdvancedMusicFeatures
)
from .utils import ModelCache, chunk_audio

class EnhancedAudioAnalyzer(AudioAnalyzer):
    """Enhanced audio analyzer implementation."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize analyzer.
        
        Parameters
        ----------
        device : Optional[str]
            Device to run models on ('cuda' or 'cpu')
        cache_dir : Optional[str]
            Directory to cache models
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # Initialize models
        self._init_speech_model()
        self._init_sound_model()
        
        # Initialize feature extractors
        self.spectral_features = SpectralFeatures()
        self.emotion_features = EmotionFeatures()
        self.music_features = AdvancedMusicFeatures()
        
        # Initialize cache
        if cache_dir:
            self.cache = ModelCache(cache_dir)
        else:
            self.cache = None
        
        # Pipeline config
        self.sample_rate = PIPELINE_CONFIG['audio']['sample_rate']
        self.chunk_size = PIPELINE_CONFIG['audio']['chunk_size']
        self.overlap = PIPELINE_CONFIG['audio']['overlap']
    
    def _init_speech_model(self):
        """Initialize Whisper model for speech recognition."""
        model_id = MODEL_CONFIG['audio']['speech']
        self.whisper_processor = WhisperProcessor.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        )
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        ).to(self.device)
    
    def _init_sound_model(self):
        """Initialize AST model for sound event detection."""
        model_id = MODEL_CONFIG['audio']['sound']
        self.sound_processor = AutoFeatureExtractor.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        )
        self.sound_model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            cache_dir=self.cache_dir
        ).to(self.device)
    
    def analyze_speech(
        self,
        audio_path: str,
        chunk_size: int = 30
    ) -> Dict[str, Any]:
        """Analyze speech content using Whisper.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        chunk_size : int
            Size of audio chunks in seconds
            
        Returns
        -------
        Dict[str, Any]
            Speech analysis results
        """
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                MODEL_CONFIG['audio']['speech'],
                audio_path,
                {'chunk_size': chunk_size}
            )
            cached_results = self.cache.load(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Get chunks
        chunks, timestamps = chunk_audio(
            audio,
            sr,
            chunk_size,
            self.overlap
        )
        
        results = []
        for chunk in chunks:
            # Process with Whisper
            inputs = self.whisper_processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.whisper_model.generate(**inputs)
            
            text = self.whisper_processor.decode(outputs[0])
            
            results.append({
                'text': text,
                'start_time': timestamps[len(results)],
                'end_time': timestamps[len(results)] + chunk_size
            })
        
        # Extract prosodic features
        prosodic_features = self.emotion_features.extract_prosodic_features(audio)
        
        output = {
            'segments': results,
            'timestamps': timestamps,
            'sample_rate': sr,
            'prosodic_features': prosodic_features
        }
        
        # Cache results
        if self.cache:
            self.cache.save(cache_key, output)
        
        return output
    
    def analyze_sound_events(
        self,
        audio_path: str,
        chunk_size: int = 30
    ) -> Dict[str, Any]:
        """Analyze sound events using AST.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        chunk_size : int
            Size of audio chunks in seconds
            
        Returns
        -------
        Dict[str, Any]
            Sound event analysis results
        """
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                MODEL_CONFIG['audio']['sound'],
                audio_path,
                {'chunk_size': chunk_size}
            )
            cached_results = self.cache.load(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Get chunks
        chunks, timestamps = chunk_audio(
            audio,
            sr,
            chunk_size,
            self.overlap
        )
        
        events = []
        features = []
        
        for chunk in chunks:
            # Process with AST
            inputs = self.sound_processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.sound_model(**inputs)
            
            # Get top events
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = probs.topk(5)
            
            events.append({
                'events': [
                    {
                        'label': self.sound_model.config.id2label[idx.item()],
                        'probability': prob.item()
                    }
                    for prob, idx in zip(top_probs[0], top_indices[0])
                ],
                'start_time': timestamps[len(events)],
                'end_time': timestamps[len(events)] + chunk_size
            })
            features.append(outputs.logits.cpu().numpy())
        
        # Extract spectral features
        spectral_features = self.spectral_features.extract_spectral_features(audio)
        wavelet_features = self.spectral_features.compute_wavelet_features(audio)
        
        output = {
            'events': events,
            'timestamps': timestamps,
            'features': np.array(features),
            'sample_rate': sr,
            'spectral_features': spectral_features,
            'wavelet_features': wavelet_features
        }
        
        # Cache results
        if self.cache:
            self.cache.save(cache_key, output)
        
        return output
    
    def analyze_music(
        self,
        audio_path: str,
        chunk_size: int = 30
    ) -> Dict[str, Any]:
        """Analyze musical elements.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        chunk_size : int
            Size of audio chunks in seconds
            
        Returns
        -------
        Dict[str, Any]
            Music analysis results
        """
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                'music_analysis',
                audio_path,
                {'chunk_size': chunk_size}
            )
            cached_results = self.cache.load(cache_key)
            if cached_results is not None:
                return cached_results
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract advanced features
        rhythm_features = self.music_features.extract_rhythm_features(audio)
        harmony_features = self.music_features.extract_harmony_features(audio)
        timbre_features = self.music_features.extract_timbre_features(audio)
        
        # Get chunks for temporal analysis
        chunks, timestamps = chunk_audio(
            audio,
            sr,
            chunk_size,
            self.overlap
        )
        
        segments = []
        for i, chunk in enumerate(chunks):
            # Analyze chunk
            chunk_rhythm = self.music_features.extract_rhythm_features(chunk)
            chunk_harmony = self.music_features.extract_harmony_features(chunk)
            
            segments.append({
                'tempo': chunk_rhythm['tempo'],
                'beat_frames': chunk_rhythm['beat_times'].tolist(),
                'harmony': {
                    'key': chunk_harmony['estimated_key'],
                    'chords': chunk_harmony['chord_features']
                },
                'start_time': timestamps[i],
                'end_time': timestamps[i] + chunk_size
            })
        
        output = {
            'segments': segments,
            'timestamps': timestamps,
            'features': {
                'rhythm': rhythm_features,
                'harmony': harmony_features,
                'timbre': timbre_features
            },
            'sample_rate': sr
        }
        
        # Cache results
        if self.cache:
            self.cache.save(cache_key, output)
        
        return output
    
    def analyze(self, audio_path: str) -> AudioAnalysisResult:
        """Perform complete audio analysis.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        AudioAnalysisResult
            Complete analysis results
        """
        # Perform all analyses
        speech_results = self.analyze_speech(audio_path)
        music_results = self.analyze_music(audio_path)
        sound_results = self.analyze_sound_events(audio_path)
        
        # Load audio for emotion analysis
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        emotion_dynamics = self.emotion_features.extract_emotion_dynamics(audio)
        
        # Combine results
        return AudioAnalysisResult(
            speech_text=" ".join(seg['text'] for seg in speech_results['segments']),
            music_segments=music_results['segments'],
            sound_events=sound_results['events'],
            timestamps=speech_results['timestamps'],
            features={
                'music': music_results['features'],
                'sound': sound_results['features'],
                'spectral': sound_results['spectral_features'],
                'wavelet': sound_results['wavelet_features'],
                'prosodic': speech_results['prosodic_features'],
                'emotion_dynamics': emotion_dynamics
            },
            metadata={
                'sample_rate': speech_results['sample_rate'],
                'duration': speech_results['timestamps'][-1]
            }
        ) 
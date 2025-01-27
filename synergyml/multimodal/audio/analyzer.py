"""Audio analysis module for SynergyML."""

import os
from typing import Dict, Any, Optional
import numpy as np
import librosa
import torch
import torchaudio
from transformers import pipeline, AutoModelForAudioClassification

class AudioAnalyzer:
    """Audio analysis class for processing and analyzing audio content."""
    
    def __init__(
        self,
        speech_model: str = "openai/whisper-large-v3",
        music_model: str = "facebook/musicgen-large",
        sound_model: str = "microsoft/audio-spectrogram-transformer"
    ):
        """Initialize audio analyzer with specified models.
        
        Parameters
        ----------
        speech_model : str
            Model for speech recognition
        music_model : str
            Model for music analysis
        sound_model : str
            Model for general sound analysis
        """
        self.models = {
            'speech': speech_model,
            'music': music_model,
            'sound': sound_model
        }
        
        # Initialize pipelines lazily
        self._speech_pipeline = None
        self._music_pipeline = None
        self._sound_pipeline = None
    
    def _load_audio(self, audio_path: str) -> Dict[str, Any]:
        """Load and preprocess audio file.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        Dict[str, Any]
            Preprocessed audio data
        """
        # Load audio with librosa
        y, sr = librosa.load(audio_path)
        
        # Extract basic features
        features = {
            'waveform': y,
            'sample_rate': sr,
            'duration': librosa.get_duration(y=y, sr=sr),
            'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
            'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        }
        
        return features
    
    def _get_speech_pipeline(self):
        """Lazy loading of speech recognition pipeline."""
        if self._speech_pipeline is None:
            self._speech_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.models['speech']
            )
        return self._speech_pipeline
    
    def _get_music_pipeline(self):
        """Lazy loading of music analysis pipeline."""
        if self._music_pipeline is None:
            self._music_pipeline = pipeline(
                "audio-classification",
                model=self.models['music']
            )
        return self._music_pipeline
    
    def _get_sound_pipeline(self):
        """Lazy loading of sound analysis pipeline."""
        if self._sound_pipeline is None:
            self._sound_pipeline = pipeline(
                "audio-classification",
                model=self.models['sound']
            )
        return self._sound_pipeline
    
    def analyze_speech(self, audio_path: str) -> Dict[str, Any]:
        """Perform speech recognition and analysis.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        Dict[str, Any]
            Speech analysis results
        """
        pipeline = self._get_speech_pipeline()
        
        # Transcribe audio
        result = pipeline(audio_path)
        
        return {
            'transcript': result['text'],
            'confidence': result.get('confidence', None),
            'timestamps': result.get('timestamps', None)
        }
    
    def analyze_music(self, audio_path: str) -> Dict[str, Any]:
        """Analyze musical content.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        Dict[str, Any]
            Music analysis results
        """
        features = self._load_audio(audio_path)
        pipeline = self._get_music_pipeline()
        
        # Extract music-specific features
        music_features = {
            'tempo': features['tempo'],
            'chroma': librosa.feature.chroma_stft(
                y=features['waveform'],
                sr=features['sample_rate']
            ),
            'onset_env': librosa.onset.onset_strength(
                y=features['waveform'],
                sr=features['sample_rate']
            )
        }
        
        # Classify music genre/style
        classification = pipeline(audio_path)
        
        return {
            'features': music_features,
            'classification': classification,
            'duration': features['duration']
        }
    
    def analyze_sound(self, audio_path: str) -> Dict[str, Any]:
        """Analyze general sound content.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        Dict[str, Any]
            Sound analysis results
        """
        features = self._load_audio(audio_path)
        pipeline = self._get_sound_pipeline()
        
        # Extract sound-specific features
        sound_features = {
            'spectral': {
                'centroid': features['spectral_centroid'],
                'bandwidth': librosa.feature.spectral_bandwidth(
                    y=features['waveform'],
                    sr=features['sample_rate']
                ),
                'rolloff': librosa.feature.spectral_rolloff(
                    y=features['waveform'],
                    sr=features['sample_rate']
                )
            },
            'mfcc': features['mfcc']
        }
        
        # Classify sound type
        classification = pipeline(audio_path)
        
        return {
            'features': sound_features,
            'classification': classification,
            'duration': features['duration']
        }
    
    def analyze(
        self,
        audio_path: str,
        analysis_types: Optional[list] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive audio analysis.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        analysis_types : Optional[list]
            List of analysis types to perform
            Default is ['speech', 'music', 'sound']
            
        Returns
        -------
        Dict[str, Any]
            Combined analysis results
        """
        if analysis_types is None:
            analysis_types = ['speech', 'music', 'sound']
        
        results = {}
        
        if 'speech' in analysis_types:
            results['speech'] = self.analyze_speech(audio_path)
        
        if 'music' in analysis_types:
            results['music'] = self.analyze_music(audio_path)
        
        if 'sound' in analysis_types:
            results['sound'] = self.analyze_sound(audio_path)
        
        return results 
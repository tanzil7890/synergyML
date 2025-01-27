# SynergyML Video Analysis

This document provides detailed information about the video analysis capabilities in SynergyML.

## Features

### 1. Basic Video Analysis
- General video content analysis
- Custom prompts for specific analysis needs
- Integration with GPT-4 Vision

### 2. Video Classification
- Zero-shot and few-shot classification
- Custom category support
- Confidence scores for predictions

### 3. Action Recognition
- Built-in support for 400+ action classes
- Multiple model backends (I3D, ViViT)
- Confidence thresholds and top-k predictions

### 4. Video Captioning
- Multiple detail levels (basic, detailed, analysis)
- Timestamp inclusion
- Scene-aware descriptions

### 5. Scene Segmentation
- Automatic scene boundary detection
- Content analysis for each scene
- Configurable thresholds and scene lengths

### 6. Temporal Analysis
- Motion pattern detection
- Transition point identification
- Temporal complexity analysis

### 7. Object Tracking
- Multi-object tracking
- Motion pattern analysis
- Object interaction detection
- Support for YOLOv8 and Detic models

### 8. Emotion Analysis
- Facial emotion recognition
- Scene emotion understanding
- Temporal emotion patterns
- Emotional arc detection

## Installation

1. Install SynergyML:
```bash
pip install synergyml
```

2. Install video analysis dependencies:
```bash
pip install -r requirements-video.txt
```

## Quick Start

```python
from synergyml.config import SynergyMLConfig
from synergyml.multimodal import VideoAnalyzer

# Set up your OpenAI API key
SynergyMLConfig.set_openai_key("your-openai-key")

# Create analyzer
analyzer = VideoAnalyzer()

# Analyze video
result = analyzer.analyze(
    video_path="video.mp4",
    prompt="Describe what's happening in this video."
)
print(result)
```

## Example Usage

See `video_analysis_examples.py` for comprehensive examples of all features.

### Basic Analysis
```python
from synergyml.multimodal import VideoAnalyzer

analyzer = VideoAnalyzer()
analysis = analyzer.analyze(
    video_path="video.mp4",
    prompt="Describe the main events in this video."
)
```

### Action Recognition
```python
from synergyml.multimodal import VideoActionRecognizer

recognizer = VideoActionRecognizer()
actions = recognizer.predict_action(
    video_path="video.mp4",
    top_k=3,
    threshold=0.3
)
```

### Object Tracking
```python
from synergyml.multimodal import VideoObjectTracker

tracker = VideoObjectTracker()
results = tracker.track_objects(
    video_path="video.mp4",
    objects_of_interest=["person", "car"],
    track_motion=True,
    analyze_interactions=True
)
```

### Emotion Analysis
```python
from synergyml.multimodal import VideoEmotionAnalyzer

analyzer = VideoEmotionAnalyzer()
emotions = analyzer.analyze_emotions(
    video_path="video.mp4",
    analyze_faces=True,
    analyze_scene=True
)
```

## Configuration

### Model Selection
Each analyzer class supports different model backends:
- VideoAnalyzer: GPT-4-Vision
- VideoActionRecognizer: I3D, ViViT
- VideoObjectTracker: YOLOv8, Detic
- VideoEmotionAnalyzer: DeepFace, CLIP

Example:
```python
analyzer = VideoActionRecognizer(
    model="gpt-4-vision-preview",
    vision_model="i3d"
)
```

### Performance Tuning
Adjust thresholds and parameters for better performance:
```python
segmenter = VideoSceneSegmenter(
    threshold=0.7,  # Scene change sensitivity
    min_scene_length=24  # Minimum frames per scene
)

tracker = VideoObjectTracker(
    confidence_threshold=0.5  # Object detection confidence
)
```

## Best Practices

1. **Video Loading**
   - Use appropriate video formats (MP4, AVI, MOV)
   - Consider video resolution and length
   - Ensure sufficient memory for processing

2. **Model Selection**
   - Choose models based on your specific needs
   - Consider speed vs. accuracy tradeoffs
   - Use appropriate confidence thresholds

3. **Error Handling**
   - Always handle potential exceptions
   - Provide fallback options
   - Validate video files before processing

4. **Performance Optimization**
   - Use appropriate batch sizes
   - Consider GPU acceleration when available
   - Optimize frame sampling for long videos

## Common Issues and Solutions

1. **Memory Issues**
   - Reduce video resolution
   - Use frame sampling
   - Process in smaller segments

2. **Speed Issues**
   - Use lighter models
   - Reduce frame rate
   - Enable GPU acceleration

3. **Accuracy Issues**
   - Try different models
   - Adjust confidence thresholds
   - Use ensemble approaches

## Contributing

We welcome contributions! Please see our main README for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
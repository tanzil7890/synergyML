# SynergyML Audio Analysis Examples

This directory contains comprehensive examples demonstrating the audio analysis capabilities of SynergyML. The examples cover various aspects of audio analysis, from basic feature extraction to advanced multimodal analysis.

## Contents

### 1. Basic Audio Analysis (`audio_analysis_examples.py`)
- Basic feature extraction
- Advanced spectral analysis
- Temporal pattern analysis
- Event detection
- Correlation analysis

### 2. Music Analysis (`music_analysis_examples.py`)
- Musical structure analysis
  - Harmony and chord detection
  - Rhythm analysis
  - Form analysis
- Performance dynamics analysis
  - Tempo and timing variations
  - Dynamic range analysis
  - Expressive features
- Genre characteristics analysis
  - Spectral features
  - Rhythm patterns
  - Timbral analysis

### 3. Speech Analysis (`speech_analysis_examples.py`)
- Prosody analysis
  - Pitch tracking
  - Rhythm and timing
  - Intonation patterns
- Speech segmentation
  - Phonetic content
  - Segment boundaries
  - Feature extraction
- Speaker characteristics
  - Voice quality
  - Speaker-specific features
- Emotional content analysis
  - Emotional indicators
  - Speech dynamics
  - Temporal patterns

### 4. Real-World Applications (`real_world_examples.py`)
- Podcast Audio Analysis
  - Voice clarity assessment
  - Speaker turn detection
  - Content structure analysis
- Live Music Recording Analysis
  - Audio quality metrics
  - Room acoustics analysis
  - Audience noise detection
  - Performance timing analysis
- Environmental Audio Analysis
  - Background noise profiling
  - Event detection
  - Acoustic complexity analysis
- Voice Quality Assessment
  - Pitch stability analysis
  - Voice quality metrics
  - Formant analysis
  - Spectral characteristics

## Features

### Visualization Capabilities
- Time-frequency representations
- Correlation matrices
- Event detection plots
- Structural analysis visualizations
- Interactive plots with Plotly
- Real-time visualization updates
- Custom plot configurations

### Analysis Methods
- Wavelet analysis
- Bispectral analysis
- Event detection
- Correlation analysis
- Feature extraction
- Pattern recognition
- Room acoustics analysis
- Voice quality assessment
- Environmental sound analysis

## Usage

### Prerequisites
```bash
pip install synergyml librosa numpy scipy matplotlib plotly soundfile
```

### Running Examples
1. Basic audio analysis:
```bash
python audio_analysis_examples.py
```

2. Music-specific analysis:
```bash
python music_analysis_examples.py
```

3. Speech analysis:
```bash
python speech_analysis_examples.py
```

4. Real-world applications:
```bash
python real_world_examples.py
```

### Example Code
```python
from synergyml.multimodal.fusion import MultimodalMediaAnalyzer, FusionVisualizer

# Initialize analyzer and visualizer
analyzer = MultimodalMediaAnalyzer()
visualizer = FusionVisualizer()

# Basic audio analysis
results = analyzer.analyze_audio("example.wav")

# Visualize results with custom configuration
visualizer.plot_multimodal_summary(
    results,
    title="Audio Analysis Results",
    downsample_factor=2,
    show_colorbar=True
)

# Real-world example: Analyze podcast
podcast_results = analyzer.analyze_podcast("podcast.wav")
visualizer.plot_event_detection(
    podcast_results,
    title="Podcast Speaker Turns",
    min_event_duration=0.5
)
```

## Advanced Features

### 1. Multimodal Analysis
- Integration of multiple audio features
- Cross-modal correlation analysis
- Event synchronization
- Pattern discovery
- Real-time analysis capabilities

### 2. Interactive Visualization
- Dynamic plots
- Real-time parameter adjustment
- Custom visualization options
- Export capabilities
- Multiple visualization layouts
- Interactive event markers

### 3. Advanced Analytics
- Wavelet-based analysis
- Statistical significance testing
- Pattern recognition
- Event detection
- Room acoustics analysis
- Voice quality metrics
- Environmental sound classification

## Best Practices

1. **Data Preparation**
   - Use appropriate sample rates
   - Normalize audio inputs
   - Handle missing data
   - Check audio quality before analysis
   - Remove DC offset if present

2. **Feature Selection**
   - Choose relevant features
   - Consider computational cost
   - Balance accuracy and efficiency
   - Use domain-specific features
   - Validate feature importance

3. **Visualization**
   - Use appropriate plot types
   - Include relevant context
   - Maintain clear labeling
   - Consider interactive elements
   - Optimize for performance

4. **Analysis Pipeline**
   - Modular approach
   - Error handling
   - Result validation
   - Performance monitoring
   - Caching for large files

## Common Issues and Solutions

### Audio Loading
```python
# Recommended way to load audio
y, sr = librosa.load(audio_path, sr=None)  # Use original sample rate

# For partial loading of large files
y, sr = librosa.load(audio_path, offset=30.0, duration=10.0)
```

### Memory Management
```python
# For large files, use frame-wise processing
for frame in librosa.stream(audio_path, block_length=256, frame_length=4096):
    process_frame(frame)

# For visualization of long audio
visualizer.plot_multimodal_summary(results, downsample_factor=2)
```

### Real-time Processing
```python
# For real-time analysis, use smaller buffer sizes
buffer_size = 1024
hop_length = buffer_size // 4

def process_realtime(audio_buffer):
    features = analyzer.extract_features(
        audio_buffer,
        buffer_size=buffer_size,
        hop_length=hop_length
    )
    return features
```

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Add your examples
4. Submit a pull request

### Adding New Examples
1. Create a new Python file in the examples directory
2. Follow the existing code structure
3. Include comprehensive documentation
4. Add tests for new functionality
5. Update the README

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
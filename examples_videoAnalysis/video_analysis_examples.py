"""Examples demonstrating the video analysis capabilities of SynergyML."""

import os
from pathlib import Path
from typing import Dict, Any

from synergyml.multimodal import (
    VideoAnalyzer,
    VideoClassifier,
    VideoActionRecognizer,
    VideoCaptioner,
    VideoSceneSegmenter,
    VideoTemporalAnalyzer,
    VideoObjectTracker,
    VideoEmotionAnalyzer
)
from synergyml.config import SynergyMLConfig

from utils import (
    validate_video_file,
    validate_output_directory,
    handle_api_error,
    validate_model_config,
    get_video_info
)
from visualizations import (
    plot_emotion_timeline,
    plot_scene_segmentation,
    plot_action_recognition,
    plot_object_tracking,
    plot_temporal_patterns,
    create_analysis_dashboard
)

@handle_api_error
def basic_video_analysis(
    video_path: str,
    output_dir: str,
    custom_prompt: str = "Analyze the content and key events in this video."
) -> Dict[str, Any]:
    """Perform basic video analysis with custom prompts.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
    custom_prompt : str, optional
        Custom analysis prompt
        
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    # Validate inputs
    validate_video_file(video_path)
    output_dir = validate_output_directory(output_dir)
    
    # Initialize analyzer
    analyzer = VideoAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze(
        video_path=video_path,
        prompt=custom_prompt
    )
    
    return results

@handle_api_error
def video_classification(
    video_path: str,
    output_dir: str,
    categories: list = None
) -> Dict[str, Any]:
    """Perform video classification.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
    categories : list, optional
        Custom categories for classification
        
    Returns
    -------
    Dict[str, Any]
        Classification results
    """
    # Validate inputs
    validate_video_file(video_path)
    output_dir = validate_output_directory(output_dir)
    
    # Initialize classifier
    classifier = VideoClassifier()
    
    # Perform classification
    results = classifier.classify(
        video_path=video_path,
        categories=categories
    )
    
    return results

@handle_api_error
def action_recognition(
    video_path: str,
    output_dir: str,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """Perform action recognition.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
    confidence_threshold : float, optional
        Confidence threshold for action detection
        
    Returns
    -------
    Dict[str, Any]
        Action recognition results
    """
    # Validate inputs
    validate_video_file(video_path)
    output_dir = validate_output_directory(output_dir)
    
    # Initialize recognizer
    recognizer = VideoActionRecognizer()
    
    # Perform recognition
    results = recognizer.recognize(
        video_path=video_path,
        confidence_threshold=confidence_threshold
    )
    
    # Visualize results
    plot_action_recognition(
        results['action_scores'],
        os.path.join(output_dir, 'action_recognition.png')
    )
    
    return results

@handle_api_error
def scene_segmentation(
    video_path: str,
    output_dir: str,
    min_scene_length: int = 30
) -> Dict[str, Any]:
    """Perform scene segmentation.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
    min_scene_length : int, optional
        Minimum scene length in frames
        
    Returns
    -------
    Dict[str, Any]
        Scene segmentation results
    """
    # Validate inputs
    validate_video_file(video_path)
    output_dir = validate_output_directory(output_dir)
    
    # Initialize segmenter
    segmenter = VideoSceneSegmenter()
    
    # Perform segmentation
    results = segmenter.segment(
        video_path=video_path,
        min_scene_length=min_scene_length
    )
    
    # Visualize results
    plot_scene_segmentation(
        results['scenes'],
        os.path.join(output_dir, 'scene_segmentation.html')
    )
    
    return results

@handle_api_error
def temporal_analysis(
    video_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Perform temporal pattern analysis.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
        
    Returns
    -------
    Dict[str, Any]
        Temporal analysis results
    """
    # Validate inputs
    validate_video_file(video_path)
    output_dir = validate_output_directory(output_dir)
    
    # Initialize analyzer
    analyzer = VideoTemporalAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze(video_path=video_path)
    
    # Visualize results
    plot_temporal_patterns(
        results,
        os.path.join(output_dir, 'temporal_analysis.html')
    )
    
    return results

@handle_api_error
def object_tracking(
    video_path: str,
    output_dir: str,
    target_objects: list = None
) -> Dict[str, Any]:
    """Perform object tracking.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
    target_objects : list, optional
        List of objects to track
        
    Returns
    -------
    Dict[str, Any]
        Object tracking results
    """
    # Validate inputs
    validate_video_file(video_path)
    output_dir = validate_output_directory(output_dir)
    
    # Initialize tracker
    tracker = VideoObjectTracker()
    
    # Perform tracking
    results = tracker.track(
        video_path=video_path,
        target_objects=target_objects
    )
    
    # Get video info for visualization
    video_info = get_video_info(video_path)
    
    # Visualize results for key frames
    for frame_idx in range(0, video_info['frame_count'], video_info['fps']):
        plot_object_tracking(
            results['tracking_data'],
            results['frames'][frame_idx],
            frame_idx,
            os.path.join(output_dir, f'tracking_frame_{frame_idx}.png')
        )
    
    return results

@handle_api_error
def emotion_analysis(
    video_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Perform emotion analysis.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
        
    Returns
    -------
    Dict[str, Any]
        Emotion analysis results
    """
    # Validate inputs
    validate_video_file(video_path)
    output_dir = validate_output_directory(output_dir)
    
    # Initialize analyzer
    analyzer = VideoEmotionAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze(video_path=video_path)
    
    # Visualize results
    plot_emotion_timeline(
        results['emotion_scores'],
        os.path.join(output_dir, 'emotion_analysis.png')
    )
    
    return results

def run_comprehensive_analysis(
    video_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Run comprehensive video analysis with all features.
    
    Parameters
    ----------
    video_path : str
        Path to input video file
    output_dir : str
        Path to output directory
        
    Returns
    -------
    Dict[str, Any]
        Combined analysis results
    """
    # Create output directory
    output_dir = validate_output_directory(output_dir)
    
    # Run all analyses
    results = {
        'basic': basic_video_analysis(video_path, output_dir),
        'classification': video_classification(video_path, output_dir),
        'actions': action_recognition(video_path, output_dir),
        'scenes': scene_segmentation(video_path, output_dir),
        'temporal': temporal_analysis(video_path, output_dir),
        'objects': object_tracking(video_path, output_dir),
        'emotions': emotion_analysis(video_path, output_dir)
    }
    
    # Create dashboard
    create_analysis_dashboard(results, output_dir)
    
    return results

def main():
    """Run example analysis on a sample video."""
    # Set up configuration
    SynergyMLConfig.set_openai_key(os.getenv('OPENAI_API_KEY'))
    
    # Define paths
    video_path = "path/to/sample_video.mp4"
    output_dir = "analysis_results"
    
    try:
        # Validate video file
        validate_video_file(video_path)
        
        # Run comprehensive analysis
        results = run_comprehensive_analysis(video_path, output_dir)
        
        print("Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 
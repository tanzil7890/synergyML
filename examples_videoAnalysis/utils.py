"""Utility functions for video analysis examples."""

import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from pathlib import Path

def validate_video_file(video_path: str) -> bool:
    """Validate video file exists and can be opened.
    
    Parameters
    ----------
    video_path : str
        Path to video file
        
    Returns
    -------
    bool
        True if video is valid
        
    Raises
    ------
    FileNotFoundError
        If video file doesn't exist
    ValueError
        If video file can't be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    cap.release()
    return True

def validate_output_directory(output_dir: str) -> str:
    """Create output directory if it doesn't exist.
    
    Parameters
    ----------
    output_dir : str
        Path to output directory
        
    Returns
    -------
    str
        Absolute path to output directory
    """
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)

def safe_float_conversion(value: str) -> Optional[float]:
    """Safely convert string to float.
    
    Parameters
    ----------
    value : str
        String to convert
        
    Returns
    -------
    Optional[float]
        Converted float or None if conversion fails
    """
    try:
        # Remove 's' from time strings
        if value.endswith('s'):
            value = value[:-1]
        return float(value)
    except (ValueError, TypeError):
        return None

def validate_analysis_results(results: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate analysis results contain required keys.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Analysis results
    required_keys : List[str]
        Required keys in results
        
    Returns
    -------
    bool
        True if results are valid
        
    Raises
    ------
    ValueError
        If results are missing required keys
    """
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValueError(f"Analysis results missing required keys: {missing_keys}")
    return True

def handle_api_error(func):
    """Decorator to handle API errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

def validate_model_config(
    model: str,
    supported_models: Dict[str, str],
    model_type: str = "vision"
) -> bool:
    """Validate model configuration.
    
    Parameters
    ----------
    model : str
        Model identifier
    supported_models : Dict[str, str]
        Dictionary of supported models
    model_type : str, optional
        Type of model, by default "vision"
        
    Returns
    -------
    bool
        True if model config is valid
        
    Raises
    ------
    ValueError
        If model is not supported
    """
    if model not in supported_models:
        raise ValueError(
            f"Unsupported {model_type} model: {model}. "
            f"Supported models: {list(supported_models.keys())}"
        )
    return True

def validate_frame_indices(
    frame_indices: List[int],
    total_frames: int
) -> List[int]:
    """Validate frame indices are within bounds.
    
    Parameters
    ----------
    frame_indices : List[int]
        List of frame indices
    total_frames : int
        Total number of frames
        
    Returns
    -------
    List[int]
        Validated frame indices
        
    Raises
    ------
    ValueError
        If indices are out of bounds
    """
    if not frame_indices:
        raise ValueError("No frame indices provided")
    
    if max(frame_indices) >= total_frames:
        raise ValueError(
            f"Frame index {max(frame_indices)} out of bounds "
            f"for video with {total_frames} frames"
        )
    
    return frame_indices

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video file information.
    
    Parameters
    ----------
    video_path : str
        Path to video file
        
    Returns
    -------
    Dict[str, Any]
        Video information
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()
    return info 
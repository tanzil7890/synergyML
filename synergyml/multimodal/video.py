"""Video analysis capabilities for SynergyML."""

import os
import base64
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin

# Optional dependencies with helpful error messages
try:
    import cv2
except ImportError:
    raise ImportError(
        "OpenCV (cv2) is required for video processing. "
        "Please install it with: pip install opencv-python>=4.8.0"
    )

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "NumPy is required for video processing. "
        "Please install it with: pip install numpy>=1.24.0"
    )

try:
    import torch
    from transformers import AutoProcessor, AutoModel
except ImportError:
    raise ImportError(
        "PyTorch and Transformers are required for video analysis. "
        "Please install them with: pip install torch>=2.0.0 transformers>=4.30.0"
    )

try:
    import av
except ImportError:
    raise ImportError(
        "PyAV is required for efficient video loading. "
        "Please install it with: pip install av>=10.0.0"
    )

from synergyml.llm.gpt.mixin import GPTClassifierMixin
from synergyml.models._base.classifier import SingleLabelMixin

class VideoAnalyzer(BaseEstimator, GPTClassifierMixin):
    """Base class for video analysis using multimodal LLMs and video models."""
    
    SUPPORTED_MODELS = {
        "videomae": "MCG-NJU/videomae-base",
        "vivit": "google/vivit-b-16-224",
        "i3d": "damo-vilab/i3d-kinetics-400",
        "mvit": "facebook/mvit-base-16x4",
        "clip": "openai/clip-vit-large-patch14",
    }

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        vision_model: str = "videomae",
        max_tokens: int = 500,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the video analyzer.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        vision_model : str, optional
            The vision model to use for feature extraction, by default "videomae"
        max_tokens : int, optional
            Maximum tokens in response, by default 500
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.vision_model = vision_model
        self.max_tokens = max_tokens
        self._set_keys(key, org)
        
        if vision_model in self.SUPPORTED_MODELS:
            self.processor = AutoProcessor.from_pretrained(self.SUPPORTED_MODELS[vision_model])
            self.vision_encoder = AutoModel.from_pretrained(self.SUPPORTED_MODELS[vision_model])
        
    def _load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video using PyAV.
        
        Parameters
        ----------
        video_path : str
            Path to video file
            
        Returns
        -------
        List[np.ndarray]
            List of video frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
        return frames
        
    def analyze(
        self, 
        video_path: str,
        prompt: str,
        num_frames: int = 8
    ) -> str:
        """Analyze video with custom prompt.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        prompt : str
            Custom prompt for analysis
        num_frames : int, optional
            Number of frames to sample, by default 8
            
        Returns
        -------
        str
            Analysis result
        """
        # Load video and sample frames
        frames = self._load_video(video_path)
        
        # Sample frames evenly
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        
        # Process frames through vision model
        inputs = self.processor(sampled_frames, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vision_encoder(**inputs)
        
        # Get LLM analysis
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{self._frames_to_base64(sampled_frames)}"}
                    }
                ]
            }
        ]
        
        completion = self._get_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens
        )
        return self._convert_completion_to_str(completion)
    
    def _frames_to_base64(self, frames) -> str:
        """Convert video frames to base64 string.
        
        Parameters
        ----------
        frames : np.ndarray
            Video frames
            
        Returns
        -------
        str
            Base64 encoded frames as a grid
        """
        # Convert frames to grid layout
        n = int(np.ceil(np.sqrt(len(frames))))
        grid = np.zeros((n * frames[0].shape[0], n * frames[0].shape[1], 3), dtype=np.uint8)
        
        for idx, frame in enumerate(frames):
            i = idx // n
            j = idx % n
            grid[i*frames[0].shape[0]:(i+1)*frames[0].shape[0],
                 j*frames[0].shape[1]:(j+1)*frames[0].shape[1]] = frame
                 
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', grid)
        return base64.b64encode(buffer).decode()

class VideoClassifier(BaseEstimator, ClassifierMixin, GPTClassifierMixin, SingleLabelMixin):
    """Video classifier using multimodal LLMs and video models."""
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        vision_model: str = "videomae",
        default_label: str = "unknown",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the video classifier.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        vision_model : str, optional
            The vision model to use for feature extraction, by default "videomae"
        default_label : str, optional
            Default label for failed predictions, by default "unknown"
        prompt_template : Optional[str], optional
            Custom prompt template, by default None
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.vision_model = vision_model
        self.default_label = default_label
        self.prompt_template = prompt_template
        self._set_keys(key, org)
        
    def fit(self, X, y):
        """Fit the classifier.
        
        Parameters
        ----------
        X : array-like
            List of video paths
        y : array-like
            Target labels
            
        Returns
        -------
        self
            Fitted classifier
        """
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X) -> np.ndarray:
        """Predict labels for videos.
        
        Parameters
        ----------
        X : array-like
            List of video paths
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        predictions = []
        analyzer = VideoAnalyzer(
            model=self.model,
            vision_model=self.vision_model,
            key=self._get_openai_key(),
            org=self._get_openai_org()
        )
        
        prompt = self.prompt_template or f"""
        Classify this video into one of the following categories: {', '.join(self.classes_)}
        Respond with ONLY the category label and nothing else.
        """
        
        for x in X:
            try:
                label = analyzer.analyze(x, prompt)
                predictions.append(self.validate_prediction(label))
            except Exception as e:
                predictions.append(self._get_default_label())
                
        return np.array(predictions)
        
    def _get_default_label(self) -> str:
        """Returns the default label to use when prediction fails.
        
        Returns
        -------
        str
            The default label specified during initialization
        """
        return self.default_label 

class VideoActionRecognizer(BaseEstimator, ClassifierMixin, GPTClassifierMixin, SingleLabelMixin):
    """Action recognition in videos using multimodal LLMs and specialized video models."""
    
    KINETICS_CLASSES = None  # Will be loaded from model config
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        vision_model: str = "i3d",  # Default to I3D for action recognition
        default_label: str = "unknown",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the action recognizer.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        vision_model : str, optional
            The vision model to use for action recognition, by default "i3d"
        default_label : str, optional
            Default label for failed predictions, by default "unknown"
        prompt_template : Optional[str], optional
            Custom prompt template, by default None
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.vision_model = vision_model
        self.default_label = default_label
        self.prompt_template = prompt_template
        self._set_keys(key, org)
        
        # Load action recognition model
        if vision_model == "i3d":
            self.processor = AutoProcessor.from_pretrained("damo-vilab/i3d-kinetics-400")
            self.vision_encoder = AutoModel.from_pretrained("damo-vilab/i3d-kinetics-400")
            # Load Kinetics classes if not already loaded
            if self.KINETICS_CLASSES is None:
                self.KINETICS_CLASSES = self._load_kinetics_classes()
    
    def _load_kinetics_classes(self) -> List[str]:
        """Load Kinetics-400 action classes."""
        return self.processor.id2label.values()
    
    def predict_action(
        self, 
        video_path: str,
        top_k: int = 3,
        threshold: float = 0.3
    ) -> List[Dict[str, float]]:
        """Predict actions in video.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        top_k : int, optional
            Number of top actions to return, by default 3
        threshold : float, optional
            Confidence threshold for predictions, by default 0.3
            
        Returns
        -------
        List[Dict[str, float]]
            List of {action: confidence} dictionaries
        """
        # Load and preprocess video
        frames = self._load_video(video_path)
        
        # Get model predictions
        inputs = self.processor(frames, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vision_encoder(**inputs)
            
        # Get top-k predictions
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(), dim=0)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        
        # Format results
        results = []
        for prob, idx in zip(top_probs, top_indices):
            if prob >= threshold:
                results.append({
                    "action": self.KINETICS_CLASSES[idx],
                    "confidence": float(prob)
                })
        
        return results

class VideoCaptioner(BaseEstimator, GPTClassifierMixin):
    """Generate detailed captions and descriptions for videos."""
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        vision_model: str = "clip",  # CLIP works well for general video understanding
        max_tokens: int = 500,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the video captioner.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        vision_model : str, optional
            The vision model for feature extraction, by default "clip"
        max_tokens : int, optional
            Maximum tokens in response, by default 500
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.vision_model = vision_model
        self.max_tokens = max_tokens
        self._set_keys(key, org)
        
        # Initialize vision model
        if vision_model == "clip":
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.vision_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
    
    def generate_caption(
        self, 
        video_path: str,
        detail_level: str = "basic",
        num_frames: int = 8,
        include_timestamps: bool = False
    ) -> Union[str, Dict[str, str]]:
        """Generate video caption.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        detail_level : str, optional
            Level of detail - "basic", "detailed", or "analysis", by default "basic"
        num_frames : int, optional
            Number of frames to sample, by default 8
        include_timestamps : bool, optional
            Whether to include timestamps in the caption, by default False
            
        Returns
        -------
        Union[str, Dict[str, str]]
            Generated caption or dictionary with timestamps
        """
        # Load video frames
        frames = self._load_video(video_path)
        
        # Sample frames evenly
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        
        # Process frames
        inputs = self.processor(sampled_frames, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vision_encoder(**inputs)
        
        # Prepare prompt based on detail level
        prompts = {
            "basic": "Provide a concise caption describing the main action and content of this video.",
            "detailed": "Provide a detailed description of this video, including main actions, subjects, setting, and notable events.",
            "analysis": """Analyze this video in detail, including:
                1. Main actions and events
                2. Key subjects and their interactions
                3. Setting and context
                4. Notable visual elements
                5. Changes or transitions
                6. Overall mood and atmosphere"""
        }
        
        prompt = prompts.get(detail_level, prompts["basic"])
        if include_timestamps:
            prompt += "\nInclude approximate timestamps for key events."
        
        # Get LLM caption
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{self._frames_to_base64(sampled_frames)}"}
                    }
                ]
            }
        ]
        
        completion = self._get_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens
        )
        return self._convert_completion_to_str(completion) 

class VideoSceneSegmenter(BaseEstimator, GPTClassifierMixin):
    """Scene segmentation and analysis for videos."""
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        vision_model: str = "clip",
        threshold: float = 0.7,
        min_scene_length: int = 24,  # 1 second at 24fps
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the scene segmenter.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        vision_model : str, optional
            The vision model for feature extraction, by default "clip"
        threshold : float, optional
            Similarity threshold for scene boundaries, by default 0.7
        min_scene_length : int, optional
            Minimum number of frames per scene, by default 24
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.vision_model = vision_model
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self._set_keys(key, org)
        
        # Initialize vision model
        if vision_model == "clip":
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.vision_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
    
    def segment_scenes(
        self,
        video_path: str,
        analyze_content: bool = True
    ) -> List[Dict[str, Any]]:
        """Segment video into scenes and optionally analyze each scene.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        analyze_content : bool, optional
            Whether to analyze scene content, by default True
            
        Returns
        -------
        List[Dict[str, Any]]
            List of scene information dictionaries
        """
        # Load video frames
        frames = self._load_video(video_path)
        
        # Extract features for all frames
        features = []
        for frame in frames:
            inputs = self.processor([frame], return_tensors="pt")
            with torch.no_grad():
                output = self.vision_encoder(**inputs)
                features.append(output.pooler_output.squeeze().numpy())
        
        # Detect scene boundaries using cosine similarity
        scene_boundaries = [0]
        prev_features = features[0]
        
        for i, curr_features in enumerate(features[1:], 1):
            similarity = np.dot(prev_features, curr_features) / (
                np.linalg.norm(prev_features) * np.linalg.norm(curr_features)
            )
            
            if similarity < self.threshold and (i - scene_boundaries[-1]) >= self.min_scene_length:
                scene_boundaries.append(i)
            prev_features = curr_features
        
        scene_boundaries.append(len(frames))
        
        # Analyze scenes
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            start_idx = scene_boundaries[i]
            end_idx = scene_boundaries[i + 1]
            scene_frames = frames[start_idx:end_idx]
            
            scene_info = {
                "start_frame": start_idx,
                "end_frame": end_idx,
                "duration_frames": end_idx - start_idx,
                "start_time": f"{start_idx/24:.2f}s",  # Assuming 24fps
                "end_time": f"{end_idx/24:.2f}s"
            }
            
            if analyze_content:
                # Sample frames for analysis
                num_samples = min(8, len(scene_frames))
                indices = np.linspace(0, len(scene_frames)-1, num_samples, dtype=int)
                sampled_frames = [scene_frames[i] for i in indices]
                
                # Get scene description
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this scene's content, setting, and any notable events."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpg;base64,{self._frames_to_base64(sampled_frames)}"}
                            }
                        ]
                    }
                ]
                
                completion = self._get_chat_completion(messages=messages)
                scene_info["description"] = self._convert_completion_to_str(completion)
            
            scenes.append(scene_info)
        
        return scenes

class VideoTemporalAnalyzer(BaseEstimator, GPTClassifierMixin):
    """Temporal analysis and pattern detection in videos."""
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        vision_model: str = "videomae",
        max_tokens: int = 500,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the temporal analyzer.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        vision_model : str, optional
            The vision model for temporal analysis, by default "videomae"
        max_tokens : int, optional
            Maximum tokens in response, by default 500
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.vision_model = vision_model
        self.max_tokens = max_tokens
        self._set_keys(key, org)
        
        # Initialize vision model
        if vision_model == "videomae":
            self.processor = AutoProcessor.from_pretrained("MCG-NJU/videomae-base")
            self.vision_encoder = AutoModel.from_pretrained("MCG-NJU/videomae-base")
    
    def analyze_temporal_patterns(
        self,
        video_path: str,
        analysis_types: List[str] = ["motion", "transitions", "patterns"],
        window_size: int = 16
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in video.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        analysis_types : List[str], optional
            Types of analysis to perform, by default ["motion", "transitions", "patterns"]
        window_size : int, optional
            Size of temporal window for analysis, by default 16
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing temporal analysis results
        """
        # Load video frames
        frames = self._load_video(video_path)
        
        # Process video in temporal windows
        results = {}
        num_windows = len(frames) // window_size
        
        for window_idx in range(num_windows):
            start_idx = window_idx * window_size
            end_idx = start_idx + window_size
            window_frames = frames[start_idx:end_idx]
            
            # Get features for temporal window
            inputs = self.processor(window_frames, return_tensors="pt")
            with torch.no_grad():
                outputs = self.vision_encoder(**inputs)
            
            # Analyze temporal patterns
            if "motion" in analysis_types:
                results.setdefault("motion_analysis", []).append(
                    self._analyze_motion(window_frames, outputs)
                )
            
            if "transitions" in analysis_types:
                results.setdefault("transition_points", []).extend(
                    self._detect_transitions(window_frames, outputs, start_idx)
                )
            
            if "patterns" in analysis_types:
                results.setdefault("temporal_patterns", []).append(
                    self._analyze_patterns(window_frames, outputs)
                )
        
        # Get overall temporal analysis from LLM
        sampled_frames = self._sample_frames_for_analysis(frames)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze the temporal aspects of this video, including:
                            1. Overall flow and pacing
                            2. Key temporal patterns
                            3. Notable changes or transitions
                            4. Motion characteristics
                            5. Temporal consistency"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{self._frames_to_base64(sampled_frames)}"}
                    }
                ]
            }
        ]
        
        completion = self._get_chat_completion(messages=messages, max_tokens=self.max_tokens)
        results["overall_analysis"] = self._convert_completion_to_str(completion)
        
        return results
    
    def _analyze_motion(
        self,
        frames: List[np.ndarray],
        features: Any
    ) -> Dict[str, Any]:
        """Analyze motion patterns in a sequence of frames."""
        # Calculate frame differences
        diffs = np.array([
            np.mean(np.abs(frames[i+1] - frames[i]))
            for i in range(len(frames)-1)
        ])
        
        return {
            "motion_magnitude": float(np.mean(diffs)),
            "motion_variance": float(np.var(diffs)),
            "peak_motion_frames": np.where(diffs > np.mean(diffs) + np.std(diffs))[0].tolist()
        }
    
    def _detect_transitions(
        self,
        frames: List[np.ndarray],
        features: Any,
        offset: int
    ) -> List[Dict[str, Any]]:
        """Detect transition points in a sequence of frames."""
        transitions = []
        prev_frame = frames[0]
        
        for i, frame in enumerate(frames[1:], 1):
            # Calculate frame difference
            diff = np.mean(np.abs(frame - prev_frame))
            
            # Detect significant changes
            if diff > np.mean([np.mean(np.abs(frames[j+1] - frames[j]))
                             for j in range(len(frames)-1)]) + np.std(diff):
                transitions.append({
                    "frame_idx": offset + i,
                    "time": f"{(offset + i)/24:.2f}s",  # Assuming 24fps
                    "transition_score": float(diff)
                })
            
            prev_frame = frame
        
        return transitions
    
    def _analyze_patterns(
        self,
        frames: List[np.ndarray],
        features: Any
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in a sequence of frames."""
        # Extract temporal features
        temporal_features = features.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Analyze feature patterns
        return {
            "temporal_complexity": float(np.var(temporal_features)),
            "feature_stability": float(np.mean([
                np.corrcoef(temporal_features[i], temporal_features[i+1])[0,1]
                for i in range(len(temporal_features)-1)
            ])),
            "pattern_strength": float(np.max(np.abs(np.fft.fft(temporal_features))))
        }
    
    def _sample_frames_for_analysis(
        self,
        frames: List[np.ndarray],
        num_samples: int = 8
    ) -> List[np.ndarray]:
        """Sample frames evenly for analysis."""
        indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
        return [frames[i] for i in indices] 

class VideoObjectTracker(BaseEstimator, GPTClassifierMixin):
    """Object tracking and analysis in videos."""
    
    SUPPORTED_MODELS = {
        "yolov8": "ultralytics/yolov8x",  # For object detection
        "detic": "facebookresearch/detic-vit-base",  # For open-vocabulary detection
    }
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        vision_model: str = "yolov8",
        confidence_threshold: float = 0.5,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the object tracker.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        vision_model : str, optional
            The vision model for object detection, by default "yolov8"
        confidence_threshold : float, optional
            Confidence threshold for detections, by default 0.5
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.vision_model = vision_model
        self.confidence_threshold = confidence_threshold
        self._set_keys(key, org)
        
        # Initialize object detection model
        if vision_model == "yolov8":
            from ultralytics import YOLO
            self.detector = YOLO(self.SUPPORTED_MODELS[vision_model])
        elif vision_model == "detic":
            self.processor = AutoProcessor.from_pretrained(self.SUPPORTED_MODELS[vision_model])
            self.detector = AutoModel.from_pretrained(self.SUPPORTED_MODELS[vision_model])
    
    def track_objects(
        self,
        video_path: str,
        objects_of_interest: Optional[List[str]] = None,
        track_motion: bool = True,
        analyze_interactions: bool = True
    ) -> Dict[str, Any]:
        """Track objects throughout the video.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        objects_of_interest : Optional[List[str]], optional
            Specific objects to track, by default None (track all)
        track_motion : bool, optional
            Whether to track object motion patterns, by default True
        analyze_interactions : bool, optional
            Whether to analyze object interactions, by default True
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing tracking results
        """
        # Load video frames
        frames = self._load_video(video_path)
        
        # Initialize tracking results
        tracking_results = {
            "objects": {},  # Object trajectories
            "frame_detections": [],  # Per-frame detections
            "interactions": [],  # Object interactions
            "motion_patterns": {},  # Motion analysis
        }
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Get detections
            if self.vision_model == "yolov8":
                results = self.detector(frame, verbose=False)[0]
                detections = self._process_yolo_detections(results, frame_idx)
            else:
                detections = self._process_detic_detections(frame, frame_idx)
            
            # Filter by confidence and objects of interest
            detections = [
                d for d in detections
                if d["confidence"] >= self.confidence_threshold
                and (objects_of_interest is None or d["class"] in objects_of_interest)
            ]
            
            # Update tracking
            self._update_object_tracking(tracking_results, detections, frame_idx)
            
            # Store frame detections
            tracking_results["frame_detections"].append(detections)
        
        # Analyze motion patterns if requested
        if track_motion:
            tracking_results["motion_patterns"] = self._analyze_motion_patterns(
                tracking_results["objects"]
            )
        
        # Analyze object interactions if requested
        if analyze_interactions:
            tracking_results["interactions"] = self._analyze_object_interactions(
                tracking_results["objects"],
                tracking_results["frame_detections"]
            )
        
        # Get LLM analysis of tracking results
        tracking_results["analysis"] = self._get_tracking_analysis(
            frames, tracking_results
        )
        
        return tracking_results
    
    def _process_yolo_detections(
        self,
        results,
        frame_idx: int
    ) -> List[Dict[str, Any]]:
        """Process YOLOv8 detection results."""
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            detections.append({
                "frame_idx": frame_idx,
                "box": box.tolist(),
                "confidence": float(conf),
                "class": results.names[int(cls)],
                "center": [(box[0] + box[2])/2, (box[1] + box[3])/2]
            })
        return detections
    
    def _process_detic_detections(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> List[Dict[str, Any]]:
        """Process Detic detection results."""
        inputs = self.processor(frame, return_tensors="pt")
        with torch.no_grad():
            outputs = self.detector(**inputs)
        
        # Process outputs (implementation depends on Detic output format)
        detections = []
        # ... process Detic outputs ...
        return detections
    
    def _update_object_tracking(
        self,
        tracking_results: Dict[str, Any],
        detections: List[Dict[str, Any]],
        frame_idx: int
    ):
        """Update object tracking with new detections."""
        for detection in detections:
            obj_class = detection["class"]
            if obj_class not in tracking_results["objects"]:
                tracking_results["objects"][obj_class] = []
            
            # Add detection to trajectory
            tracking_results["objects"][obj_class].append({
                "frame_idx": frame_idx,
                "position": detection["center"],
                "box": detection["box"],
                "confidence": detection["confidence"]
            })
    
    def _analyze_motion_patterns(
        self,
        object_trajectories: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze motion patterns of tracked objects."""
        motion_patterns = {}
        
        for obj_class, trajectory in object_trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Extract positions and calculate velocities
            positions = np.array([t["position"] for t in trajectory])
            velocities = positions[1:] - positions[:-1]
            
            # Analyze motion
            motion_patterns[obj_class] = {
                "avg_velocity": float(np.mean(np.linalg.norm(velocities, axis=1))),
                "max_velocity": float(np.max(np.linalg.norm(velocities, axis=1))),
                "direction_changes": len(self._detect_direction_changes(velocities)),
                "movement_pattern": self._classify_movement_pattern(velocities)
            }
        
        return motion_patterns
    
    def _analyze_object_interactions(
        self,
        object_trajectories: Dict[str, List[Dict[str, Any]]],
        frame_detections: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Analyze interactions between tracked objects."""
        interactions = []
        
        # Define interaction threshold (e.g., distance threshold)
        interaction_threshold = 50  # pixels
        
        # Analyze each frame's detections
        for frame_idx, detections in enumerate(frame_detections):
            # Check pairs of objects
            for i, obj1 in enumerate(detections):
                for obj2 in detections[i+1:]:
                    # Calculate distance between objects
                    distance = np.linalg.norm(
                        np.array(obj1["center"]) - np.array(obj2["center"])
                    )
                    
                    # Record interaction if objects are close
                    if distance < interaction_threshold:
                        interactions.append({
                            "frame_idx": frame_idx,
                            "time": f"{frame_idx/24:.2f}s",  # Assuming 24fps
                            "object1": obj1["class"],
                            "object2": obj2["class"],
                            "distance": float(distance),
                            "type": self._classify_interaction(obj1, obj2, distance)
                        })
        
        return interactions
    
    def _detect_direction_changes(self, velocities: np.ndarray) -> List[int]:
        """Detect significant changes in movement direction."""
        # Calculate angle changes between consecutive velocity vectors
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        angle_changes = np.diff(angles)
        
        # Detect significant direction changes (e.g., > 45 degrees)
        direction_changes = np.where(np.abs(angle_changes) > np.pi/4)[0]
        return direction_changes.tolist()
    
    def _classify_movement_pattern(self, velocities: np.ndarray) -> str:
        """Classify the type of movement pattern."""
        # Calculate various motion metrics
        speed = np.linalg.norm(velocities, axis=1)
        direction_changes = self._detect_direction_changes(velocities)
        
        # Classify based on metrics
        if len(direction_changes) == 0:
            return "linear"
        elif len(direction_changes) > len(velocities) * 0.3:
            return "erratic"
        elif np.std(speed) < 0.2 * np.mean(speed):
            return "steady"
        else:
            return "variable"
    
    def _classify_interaction(
        self,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any],
        distance: float
    ) -> str:
        """Classify the type of interaction between objects."""
        if distance < 20:
            return "contact"
        elif distance < 50:
            return "proximity"
        else:
            return "distant"
    
    def _get_tracking_analysis(
        self,
        frames: List[np.ndarray],
        tracking_results: Dict[str, Any]
    ) -> str:
        """Get LLM analysis of tracking results."""
        # Sample frames for visualization
        sampled_frames = self._sample_frames_for_analysis(frames)
        
        # Prepare tracking summary
        tracking_summary = {
            "objects_tracked": list(tracking_results["objects"].keys()),
            "total_frames": len(frames),
            "interactions": len(tracking_results["interactions"]),
            "motion_patterns": tracking_results["motion_patterns"]
        }
        
        # Get LLM analysis
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze the object tracking results:
                            1. Objects tracked: {', '.join(tracking_summary['objects_tracked'])}
                            2. Notable motion patterns and behaviors
                            3. Key interactions between objects
                            4. Overall scene dynamics"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{self._frames_to_base64(sampled_frames)}"}
                    }
                ]
            }
        ]
        
        completion = self._get_chat_completion(messages=messages)
        return self._convert_completion_to_str(completion)

class VideoEmotionAnalyzer(BaseEstimator, GPTClassifierMixin):
    """Emotion analysis in videos using facial expressions and scene context."""
    
    SUPPORTED_MODELS = {
        "face": "deepface/facial-emotion-recognition",
        "scene": "openai/clip-vit-large-patch14",
    }
    
    EMOTION_CATEGORIES = [
        "angry", "disgust", "fear", "happy", 
        "sad", "surprise", "neutral"
    ]
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        face_model: str = "face",
        scene_model: str = "scene",
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the emotion analyzer.
        
        Parameters
        ----------
        model : str, optional
            The LLM model to use, by default "gpt-4-vision-preview"
        face_model : str, optional
            The model for facial emotion recognition, by default "face"
        scene_model : str, optional
            The model for scene emotion analysis, by default "scene"
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.face_model = face_model
        self.scene_model = scene_model
        self._set_keys(key, org)
        
        # Initialize face detection and emotion recognition
        from deepface import DeepFace
        self.emotion_analyzer = DeepFace
        
        # Initialize scene understanding model
        self.scene_processor = AutoProcessor.from_pretrained(self.SUPPORTED_MODELS["scene"])
        self.scene_encoder = AutoModel.from_pretrained(self.SUPPORTED_MODELS["scene"])
    
    def analyze_emotions(
        self,
        video_path: str,
        analyze_faces: bool = True,
        analyze_scene: bool = True,
        temporal_analysis: bool = True
    ) -> Dict[str, Any]:
        """Analyze emotions in the video.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        analyze_faces : bool, optional
            Whether to analyze facial emotions, by default True
        analyze_scene : bool, optional
            Whether to analyze scene emotions, by default True
        temporal_analysis : bool, optional
            Whether to perform temporal emotion analysis, by default True
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing emotion analysis results
        """
        # Load video frames
        frames = self._load_video(video_path)
        
        results = {
            "facial_emotions": [],
            "scene_emotions": [],
            "temporal_patterns": {},
            "overall_analysis": ""
        }
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            frame_results = {}
            
            # Analyze facial emotions
            if analyze_faces:
                facial_emotions = self._analyze_facial_emotions(frame)
                results["facial_emotions"].append({
                    "frame_idx": frame_idx,
                    "time": f"{frame_idx/24:.2f}s",
                    "emotions": facial_emotions
                })
            
            # Analyze scene emotions
            if analyze_scene:
                scene_emotions = self._analyze_scene_emotions(frame)
                results["scene_emotions"].append({
                    "frame_idx": frame_idx,
                    "time": f"{frame_idx/24:.2f}s",
                    "emotions": scene_emotions
                })
        
        # Perform temporal analysis
        if temporal_analysis:
            results["temporal_patterns"] = self._analyze_emotion_patterns(
                results["facial_emotions"],
                results["scene_emotions"]
            )
        
        # Get overall analysis from LLM
        results["overall_analysis"] = self._get_emotion_analysis(frames, results)
        
        return results
    
    def _analyze_facial_emotions(
        self,
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Analyze emotions in detected faces."""
        try:
            # Analyze emotions using DeepFace
            analysis = self.emotion_analyzer.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )
            
            # Process results
            facial_emotions = []
            for face_analysis in analysis:
                emotions = face_analysis["emotion"]
                facial_emotions.append({
                    "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[0],
                    "emotion_scores": emotions,
                    "face_location": face_analysis.get("region", {})
                })
            
            return facial_emotions
        except Exception as e:
            return []  # Return empty list if no faces detected
    
    def _analyze_scene_emotions(
        self,
        frame: np.ndarray
    ) -> Dict[str, float]:
        """Analyze emotional content of the scene."""
        # Process frame through CLIP
        inputs = self.scene_processor([frame], return_tensors="pt")
        with torch.no_grad():
            features = self.scene_encoder(**inputs).pooler_output
        
        # Prepare emotion prompts
        emotion_texts = [f"This scene looks {emotion}" for emotion in self.EMOTION_CATEGORIES]
        text_inputs = self.scene_processor(
            emotion_texts,
            padding=True,
            return_tensors="pt"
        )
        
        # Get emotion scores
        with torch.no_grad():
            text_features = self.scene_encoder(**text_inputs).pooler_output
            similarity = torch.nn.functional.cosine_similarity(
                features.unsqueeze(1),
                text_features.unsqueeze(0),
                dim=-1
            )
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(similarity.squeeze(), dim=0)
        
        return {
            emotion: float(prob)
            for emotion, prob in zip(self.EMOTION_CATEGORIES, probs)
        }
    
    def _analyze_emotion_patterns(
        self,
        facial_emotions: List[Dict[str, Any]],
        scene_emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in emotions."""
        patterns = {
            "facial": self._analyze_facial_patterns(facial_emotions),
            "scene": self._analyze_scene_patterns(scene_emotions),
            "emotional_arcs": self._detect_emotional_arcs(facial_emotions, scene_emotions)
        }
        return patterns
    
    def _analyze_facial_patterns(
        self,
        facial_emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns in facial emotions."""
        if not facial_emotions:
            return {}
        
        # Track emotion frequencies and transitions
        emotion_counts = {emotion: 0 for emotion in self.EMOTION_CATEGORIES}
        emotion_transitions = []
        
        for frame_data in facial_emotions:
            for face in frame_data["emotions"]:
                emotion = face["dominant_emotion"]
                emotion_counts[emotion] += 1
                
                if emotion_transitions and emotion_transitions[-1] != emotion:
                    emotion_transitions.append(emotion)
        
        return {
            "dominant_emotions": sorted(
                emotion_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            "emotion_transitions": emotion_transitions,
            "emotional_stability": len(emotion_transitions) / len(facial_emotions)
        }
    
    def _analyze_scene_patterns(
        self,
        scene_emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns in scene emotions."""
        if not scene_emotions:
            return {}
        
        # Track emotion intensities over time
        emotion_intensities = {
            emotion: [
                frame["emotions"][emotion]
                for frame in scene_emotions
            ]
            for emotion in self.EMOTION_CATEGORIES
        }
        
        return {
            "average_intensities": {
                emotion: float(np.mean(intensities))
                for emotion, intensities in emotion_intensities.items()
            },
            "emotional_variance": {
                emotion: float(np.var(intensities))
                for emotion, intensities in emotion_intensities.items()
            }
        }
    
    def _detect_emotional_arcs(
        self,
        facial_emotions: List[Dict[str, Any]],
        scene_emotions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect significant emotional arcs in the video."""
        arcs = []
        
        # Combine facial and scene emotions
        combined_emotions = []
        for facial, scene in zip(facial_emotions, scene_emotions):
            frame_emotions = {
                "frame_idx": facial["frame_idx"],
                "time": facial["time"]
            }
            
            # Get dominant emotions
            if facial["emotions"]:
                frame_emotions["facial"] = max(
                    facial["emotions"][0]["emotion_scores"].items(),
                    key=lambda x: x[1]
                )[0]
            
            if scene["emotions"]:
                frame_emotions["scene"] = max(
                    scene["emotions"].items(),
                    key=lambda x: x[1]
                )[0]
            
            combined_emotions.append(frame_emotions)
        
        # Detect emotional transitions
        current_arc = None
        for i, emotions in enumerate(combined_emotions):
            if i == 0:
                current_arc = {
                    "start_time": emotions["time"],
                    "start_emotions": {
                        k: v for k, v in emotions.items()
                        if k not in ["frame_idx", "time"]
                    }
                }
                continue
            
            # Check for significant changes
            changes = False
            for source in ["facial", "scene"]:
                if (source in emotions and source in current_arc["start_emotions"] and
                    emotions[source] != current_arc["start_emotions"][source]):
                    changes = True
            
            if changes:
                current_arc["end_time"] = emotions["time"]
                current_arc["end_emotions"] = {
                    k: v for k, v in emotions.items()
                    if k not in ["frame_idx", "time"]
                }
                current_arc["duration"] = float(emotions["time"][:-1]) - float(current_arc["start_time"][:-1])
                
                arcs.append(current_arc)
                current_arc = {
                    "start_time": emotions["time"],
                    "start_emotions": {
                        k: v for k, v in emotions.items()
                        if k not in ["frame_idx", "time"]
                    }
                }
        
        return arcs
    
    def _get_emotion_analysis(
        self,
        frames: List[np.ndarray],
        results: Dict[str, Any]
    ) -> str:
        """Get LLM analysis of emotion results."""
        # Sample frames for visualization
        sampled_frames = self._sample_frames_for_analysis(frames)
        
        # Prepare emotion summary
        if results["facial_emotions"]:
            facial_patterns = results["temporal_patterns"]["facial"]
            dominant_emotions = facial_patterns["dominant_emotions"][:3]
            emotion_summary = f"Top emotions: {', '.join(f'{e[0]} ({e[1]})' for e in dominant_emotions)}"
        else:
            emotion_summary = "No facial emotions detected"
        
        # Get LLM analysis
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze the emotional content of this video:
                            1. {emotion_summary}
                            2. Analyze the emotional progression and key moments
                            3. Describe how facial emotions align with scene context
                            4. Identify significant emotional transitions
                            5. Comment on the overall emotional tone and impact"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{self._frames_to_base64(sampled_frames)}"}
                    }
                ]
            }
        ]
        
        completion = self._get_chat_completion(messages=messages)
        return self._convert_completion_to_str(completion) 
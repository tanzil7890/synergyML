"""Vision and multimodal capabilities for SynergyML."""

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import os

from synergyml.llm.gpt.mixin import GPTClassifierMixin
from synergyml.models._base.classifier import SingleLabelMixin, MultiLabelMixin


class ImageAnalyzer(BaseEstimator, GPTClassifierMixin):
    """Base class for image analysis using multimodal LLMs."""
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 300,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the image analyzer.
        
        Parameters
        ----------
        model : str, optional
            The model to use, by default "gpt-4-vision-preview"
        max_tokens : int, optional
            Maximum tokens in response, by default 300
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.max_tokens = max_tokens
        self._set_keys(key, org)
    
    def _encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """Encode image to base64 string.
        
        Parameters
        ----------
        image : Union[str, Image.Image, np.ndarray]
            Image to encode - can be file path, PIL Image or numpy array
            
        Returns
        -------
        str
            Base64 encoded image
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def analyze(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        prompt: str
    ) -> str:
        """Analyze image with custom prompt.
        
        Parameters
        ----------
        image : Union[str, Image.Image, np.ndarray]
            Image to analyze
        prompt : str
            Custom prompt for analysis
            
        Returns
        -------
        str
            Analysis result
        """
        encoded_image = self._encode_image(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                    }
                ]
            }
        ]
        
        completion = self._get_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens
        )
        return self._convert_completion_to_str(completion)


class ImageClassifier(BaseEstimator, ClassifierMixin, GPTClassifierMixin, SingleLabelMixin):
    """Image classifier using multimodal LLMs."""
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the image classifier.
        
        Parameters
        ----------
        model : str, optional
            The model to use, by default "gpt-4-vision-preview"
        default_label : str, optional
            Default label for failed predictions, by default "Random"
        prompt_template : Optional[str], optional
            Custom prompt template, by default None
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.default_label = default_label
        self.prompt_template = prompt_template
        self._set_keys(key, org)
        
    def fit(self, X, y):
        """Fit the classifier.
        
        Parameters
        ----------
        X : array-like
            List of image paths or PIL Images
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
        """Predict labels for images.
        
        Parameters
        ----------
        X : array-like
            List of image paths or PIL Images
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        predictions = []
        analyzer = ImageAnalyzer(
            model=self.model,
            key=self._get_openai_key(),
            org=self._get_openai_org()
        )
        
        prompt = self.prompt_template or f"""
        Classify this image into one of the following categories: {', '.join(self.classes_)}
        Respond with ONLY the category label and nothing else.
        """
        
        for x in X:
            try:
                label = analyzer.analyze(x, prompt)
                predictions.append(self.validate_prediction(label))
            except Exception as e:
                predictions.append(self._get_default_label())
                
        return np.array(predictions)


class MultimodalClassifier(BaseEstimator, ClassifierMixin, GPTClassifierMixin, MultiLabelMixin):
    """Multimodal classifier combining text and image analysis."""
    
    def __init__(
        self,
        model="gpt-4o-mini",
        max_labels=3,
        default_label="unknown",
        prompt_template=None,
        key=None,
        org=None
    ):
        self.model = model
        self.max_labels = max_labels
        self.default_label = default_label
        self.prompt_template = prompt_template or """
            Analyze this product image and review text.
            Review: {text}
            
            Based on both the image and text, provide up to {max_labels} labels.
            Available labels: {labels}
            
            Format your response as a comma-separated list of labels.
            If unsure, use 'unknown'.
            """
        self.key = key
        self.org = org
        self.classes_ = None

    def fit(self, X, y):
        """Fit the classifier with training data.
        
        Parameters
        ----------
        X : dict
            Dictionary containing 'text' and 'image' keys with corresponding lists
        y : list
            List of label lists for each sample
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Store unique labels
        self.classes_ = sorted(list(set(label for labels in y for label in labels)))
        return self

    def predict(self, X):
        """Predict labels for the input data.
        
        Parameters
        ----------
        X : dict
            Dictionary containing 'text' and 'image' keys
        
        Returns
        -------
        list
            Predicted labels for each sample
        """
        if not isinstance(X, dict) or 'text' not in X or 'image' not in X:
            raise ValueError("X must be a dictionary with 'text' and 'image' keys")

        predictions = []
        for text, image_path in zip(X['text'], X['image']):
            try:
                # Format prompt with current sample
                prompt = self.prompt_template.format(
                    text=text,
                    max_labels=self.max_labels,
                    labels=", ".join(self.classes_)
                )

                # Get prediction using GPT-4 Vision
                response = self._get_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                        ]
                    }],
                    model=self.model,
                    key=self.key,
                    org=self.org
                )

                # Extract labels from response
                labels = self._extract_labels(response)
                if not labels:
                    labels = [self.default_label] * self.max_labels
                predictions.append(labels)

            except Exception as e:
                print(f"Prediction failed: {e}")
                predictions.append([self.default_label] * self.max_labels)

        return predictions

    def _extract_labels(self, response):
        """Extract labels from the model response."""
        try:
            text = response.choices[0].message.content.strip()
            labels = [label.strip() for label in text.split(',')]
            return labels[:self.max_labels]
        except Exception as e:
            print(f"Label extraction failed: {e}")
            return []

    def _get_default_label(self) -> str:
        """Returns the default label to use when prediction fails.
        
        Returns
        -------
        str
            The default label specified during initialization
        """
        return self.default_label


class ImageCaptioner(BaseEstimator, GPTClassifierMixin):
    """Generate detailed captions and descriptions for images."""
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 500,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """Initialize the image captioner.
        
        Parameters
        ----------
        model : str, optional
            The model to use, by default "gpt-4-vision-preview"
        max_tokens : int, optional
            Maximum tokens in response, by default 500
        key : Optional[str], optional
            OpenAI API key, by default None
        org : Optional[str], optional
            OpenAI organization ID, by default None
        """
        self.model = model
        self.max_tokens = max_tokens
        self._set_keys(key, org)
        
    def generate_caption(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        detail_level: str = "basic"
    ) -> str:
        """Generate image caption.
        
        Parameters
        ----------
        image : Union[str, Image.Image, np.ndarray]
            Image to caption
        detail_level : str, optional
            Level of detail - "basic", "detailed", or "analysis", by default "basic"
            
        Returns
        -------
        str
            Generated caption
        """
        analyzer = ImageAnalyzer(
            model=self.model,
            max_tokens=self.max_tokens,
            key=self._get_openai_key(),
            org=self._get_openai_org()
        )
        
        prompts = {
            "basic": "Provide a concise caption for this image.",
            "detailed": "Provide a detailed description of this image, including main subjects, actions, setting, and notable details.",
            "analysis": "Analyze this image in detail, including:\n1. Main subjects and their relationships\n2. Setting and context\n3. Notable visual elements and composition\n4. Mood and atmosphere\n5. Any text or symbols present\n6. Technical aspects (lighting, color, focus)"
        }
        
        return analyzer.analyze(image, prompts.get(detail_level, prompts["basic"])) 
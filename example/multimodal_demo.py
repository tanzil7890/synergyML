#!/usr/bin/env python3
"""
Demonstration of SynergyML's multimodal capabilities.
This script shows various ways to use SynergyML for image analysis, classification,
and multimodal tasks.
"""

import os
from typing import Dict, List
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from synergyml import SynergyMLConfig
from synergyml.multimodal import (
    ImageAnalyzer,
    ImageClassifier,
    MultimodalClassifier,
    ImageCaptioner,
)


def setup_credentials():
    """Set up API credentials."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set OPENAI_API_KEY environment variable"
        )
    SynergyMLConfig.set_openai_key(api_key)


def demo_image_analysis(image_path: str):
    """Demonstrate basic image analysis capabilities.
    
    Args:
        image_path: Path to the image file
    """
    print("\n=== Image Analysis Demo ===")
    
    analyzer = ImageAnalyzer(
        model="gpt-4-vision-preview",
        max_tokens=300
    )
    
    # Basic object detection
    objects_prompt = "List all the main objects visible in this image. Format as a comma-separated list."
    objects = analyzer.analyze(image_path, objects_prompt)
    print("\nObjects detected:", objects)
    
    # Scene analysis
    scene_prompt = """Analyze this scene and describe:
    1. Setting and environment
    2. Main activities or events
    3. Notable visual elements
    Format as bullet points.
    """
    scene = analyzer.analyze(image_path, scene_prompt)
    print("\nScene analysis:", scene)
    
    # Technical analysis
    tech_prompt = """Analyze technical aspects:
    1. Lighting conditions
    2. Color palette
    3. Composition
    Format as bullet points.
    """
    technical = analyzer.analyze(image_path, tech_prompt)
    print("\nTechnical analysis:", technical)


def demo_image_classification(
    train_images: List[str],
    train_labels: List[str],
    test_images: List[str],
    test_labels: List[str]
):
    """Demonstrate image classification capabilities.
    
    Args:
        train_images: List of training image paths
        train_labels: List of training labels
        test_images: List of test image paths
        test_labels: List of test labels
    """
    print("\n=== Image Classification Demo ===")
    
    # Initialize classifier
    clf = ImageClassifier(
        model="gpt-4-vision-preview",
        default_label="unknown"
    )
    
    # Train
    print("\nTraining classifier...")
    clf.fit(train_images, train_labels)
    
    # Predict
    print("\nMaking predictions...")
    predictions = clf.predict(test_images)
    
    # Evaluate
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))


def demo_multimodal_classification(
    train_data: Dict,
    train_labels: List[List[str]],
    test_data: Dict,
    test_labels: List[List[str]]
):
    """Demonstrate multimodal classification capabilities.
    
    Args:
        train_data: Dict with 'text' and 'image' keys for training
        train_labels: List of label lists for training
        test_data: Dict with 'text' and 'image' keys for testing
        test_labels: List of label lists for testing
    """
    print("\n=== Multimodal Classification Demo ===")
    
    # Initialize classifier
    clf = MultimodalClassifier(
        model="gpt-4-vision-preview",
        max_labels=3,
        default_label="unknown"
    )
    
    # Train
    print("\nTraining classifier...")
    clf.fit(train_data, train_labels)
    
    # Predict
    print("\nMaking predictions...")
    predictions = clf.predict(test_data)
    
    # Evaluate
    print("\nResults:")
    for i, (text, pred) in enumerate(zip(test_data['text'], predictions)):
        print(f"\nReview: {text}")
        print(f"Predicted: {pred}")
        print(f"Actual: {test_labels[i]}")
    
    # Calculate metrics
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(test_labels)
    y_pred = mlb.transform(predictions)
    
    # Print confusion matrix for each label
    print("\nConfusion Matrices:")
    for label, matrix in zip(mlb.classes_, multilabel_confusion_matrix(y_true, y_pred)):
        print(f"\nLabel: {label}")
        print(matrix)


def demo_image_captioning(image_path: str):
    """Demonstrate image captioning capabilities.
    
    Args:
        image_path: Path to the image file
    """
    print("\n=== Image Captioning Demo ===")
    
    captioner = ImageCaptioner(
        model="gpt-4-vision-preview",
        max_tokens=500
    )
    
    # Generate captions with different detail levels
    print("\nBasic caption:")
    print(captioner.generate_caption(image_path, detail_level="basic"))
    
    print("\nDetailed description:")
    print(captioner.generate_caption(image_path, detail_level="detailed"))
    
    print("\nDetailed analysis:")
    print(captioner.generate_caption(image_path, detail_level="analysis"))


def main():
    """Main function demonstrating all capabilities."""
    # Setup
    setup_credentials()
    
    # Sample data paths (replace with actual paths)
    SAMPLE_IMAGE = "data/sample_image.jpg"
    
    TRAIN_IMAGES = [
        "data/train_cat1.jpg",
        "data/train_dog1.jpg",
        "data/train_cat2.jpg",
        "data/train_dog2.jpg"
    ]
    TRAIN_LABELS = ["cat", "dog", "cat", "dog"]
    
    TEST_IMAGES = [
        "data/test_cat.jpg",
        "data/test_dog.jpg"
    ]
    TEST_LABELS = ["cat", "dog"]
    
    MULTIMODAL_TRAIN = {
        'text': [
            "Great product, exactly as shown",
            "Poor quality and damaged",
            "Perfect fit and design",
            "Wrong size and cheap material"
        ],
        'image': [
            "data/product1_good.jpg",
            "data/product2_bad.jpg",
            "data/product3_good.jpg",
            "data/product4_bad.jpg"
        ]
    }
    MULTIMODAL_TRAIN_LABELS = [
        ["positive", "accurate"],
        ["negative", "damaged"],
        ["positive", "fit"],
        ["negative", "quality_issue"]
    ]
    
    MULTIMODAL_TEST = {
        'text': [
            "Good but different color",
            "Excellent quality product"
        ],
        'image': [
            "data/test_product1.jpg",
            "data/test_product2.jpg"
        ]
    }
    MULTIMODAL_TEST_LABELS = [
        ["positive", "color_issue"],
        ["positive", "quality"]
    ]
    
    # Run demos
    try:
        # Basic image analysis
        demo_image_analysis(SAMPLE_IMAGE)
        
        # Image classification
        demo_image_classification(
            TRAIN_IMAGES,
            TRAIN_LABELS,
            TEST_IMAGES,
            TEST_LABELS
        )
        
        # Multimodal classification
        demo_multimodal_classification(
            MULTIMODAL_TRAIN,
            MULTIMODAL_TRAIN_LABELS,
            MULTIMODAL_TEST,
            MULTIMODAL_TEST_LABELS
        )
        
        # Image captioning
        demo_image_captioning(SAMPLE_IMAGE)
        
    except Exception as e:
        print(f"Error during demo: {str(e)}")


if __name__ == "__main__":
    main() 
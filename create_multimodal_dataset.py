import json
import os
from typing import Dict, List, Tuple

def create_sample_dataset() -> Tuple[Dict, List[List[str]]]:
    """Create a sample multimodal dataset for e-commerce products."""
    
    # Base path for images
    base_path = "test_dataset/multimodal/images"
    
    # Training data
    X = {
        'text': [
            "Perfect fit! The color matches the picture exactly",
            "Poor quality material, started falling apart after one wash",
            "Great design but runs small, order one size up",
            "Excellent product, exactly as described",
            "The stitching is loose and the color is different from picture",
            "Amazing quality and fast shipping!",
            "Not worth the price, very disappointing",
            "Comfortable and stylish, highly recommend"
        ],
        'image': [
            f"{base_path}/product1_good.jpg",
            f"{base_path}/product2_bad.jpg",
            f"{base_path}/product3_size.jpg",
            f"{base_path}/product4_good.jpg",
            f"{base_path}/product5_bad.jpg",
            f"{base_path}/product6_good.jpg",
            f"{base_path}/product7_bad.jpg",
            f"{base_path}/product8_good.jpg"
        ]
    }
    
    # Multi-label annotations
    y = [
        ["positive", "accurate", "fit"],
        ["negative", "quality_issue", "durability"],
        ["neutral", "size_issue", "design"],
        ["positive", "accurate", "satisfaction"],
        ["negative", "color_mismatch", "quality_issue"],
        ["positive", "service", "quality"],
        ["negative", "value", "disappointment"],
        ["positive", "comfort", "style"]
    ]
    
    return X, y

def save_dataset(X: Dict, y: List[List[str]], base_path: str = "test_dataset/multimodal"):
    """Save the dataset to disk."""
    
    # Create data directory if it doesn't exist
    os.makedirs(f"{base_path}/data", exist_ok=True)
    
    # Save the dataset
    dataset = {
        'features': X,
        'labels': y
    }
    
    with open(f"{base_path}/data/multimodal_dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)

def main():
    # Create the dataset
    X, y = create_sample_dataset()
    
    # Save it
    save_dataset(X, y)
    
    print("Dataset created successfully!")
    print(f"Number of samples: {len(y)}")
    print("\nExample entry:")
    print(f"Text: {X['text'][0]}")
    print(f"Image path: {X['image'][0]}")
    print(f"Labels: {y[0]}")

if __name__ == "__main__":
    main()
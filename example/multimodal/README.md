# SynergyML Multimodal Examples

This directory contains examples demonstrating SynergyML's multimodal capabilities, showcasing how to leverage GPT-4 Vision for various image and text analysis tasks.

## Prerequisites

Before running the examples, ensure you have:

1. SynergyML installed with vision support:
```bash
pip install -e ".[vision]"
```

2. An OpenAI API key with GPT-4V access:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. Required sample images in the `data` directory (see [Data Setup](#data-setup) below)

## Examples Overview

### 1. Image Analysis (`01_image_classification.ipynb`)
Demonstrates basic image classification using GPT-4 Vision:
- Single-label classification
- Custom prompt templates
- Integration with scikit-learn metrics
- Example: Animal classification (cats vs dogs)

### 2. Multimodal Review Classification (`02_multimodal_review_classification.ipynb`)
Shows how to combine text and image analysis for e-commerce:
- Multi-label classification
- Product review analysis
- Image-text correlation
- Performance evaluation with multilabel metrics

### 3. Image Analysis and Captioning (`03_image_analysis_and_captioning.ipynb`)
Explores advanced image analysis capabilities:
- Basic to detailed image captioning
- Scene understanding
- Technical image analysis
- Batch processing and comparisons

### 4. Complete Demo Script (`multimodal_demo.py`)
A single script combining all capabilities:
```bash
python multimodal_demo.py
```

## Data Setup

### Required Directory Structure
```
data/
├── sample_image.jpg          # For general demos
├── train_cat1.jpg           # Training images
├── train_cat2.jpg
├── train_dog1.jpg
├── train_dog2.jpg
├── test_cat.jpg            # Test images
├── test_dog.jpg
├── product1_good.jpg       # E-commerce product images
├── product2_bad.jpg
├── product3_good.jpg
├── product4_bad.jpg
├── test_product1.jpg
└── test_product2.jpg
```

### Sample Data Sources
You can use your own images or get sample images from:
- [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats)
- [E-commerce Product Images Dataset](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images)

## Example Usage

### 1. Basic Image Classification
```python
from synergyml.multimodal import ImageClassifier

# Initialize classifier
clf = ImageClassifier(model="gpt-4-vision-preview")

# Train and predict
X = ["path/to/image1.jpg", "path/to/image2.jpg"]
y = ["cat", "dog"]
clf.fit(X, y)
predictions = clf.predict(["path/to/new_image.jpg"])
```

### 2. Multimodal Review Classification
```python
from synergyml.multimodal import MultimodalClassifier

# Initialize classifier
clf = MultimodalClassifier(
    model="gpt-4-vision-preview",
    max_labels=3
)

# Train with text and images
X = {
    'text': ["Great product!", "Poor quality"],
    'image': ["product1.jpg", "product2.jpg"]
}
y = [["positive", "accurate"], ["negative", "quality_issues"]]
clf.fit(X, y)
```

### 3. Image Captioning
```python
from synergyml.multimodal import ImageCaptioner

# Initialize captioner
captioner = ImageCaptioner(model="gpt-4-vision-preview")

# Generate captions
basic = captioner.generate_caption("image.jpg", detail_level="basic")
detailed = captioner.generate_caption("image.jpg", detail_level="detailed")
analysis = captioner.generate_caption("image.jpg", detail_level="analysis")
```

## Advanced Features

### Custom Prompt Templates
You can customize how the model analyzes images by providing custom prompts:

```python
clf = ImageClassifier(
    model="gpt-4-vision-preview",
    prompt_template="""
    Analyze this image considering:
    1. Main subject and composition
    2. Visual characteristics
    3. Distinguishing features
    
    Classify as one of: {', '.join(self.classes_)}
    Respond with ONLY the category label.
    """
)
```

### Integration with scikit-learn
All classifiers are compatible with scikit-learn's API:

```python
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Evaluate performance
scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
```

## Best Practices

1. **Image Quality**
   - Use clear, well-lit images
   - Recommended size: 512x512 to 2048x2048 pixels
   - Supported formats: JPG, PNG

2. **Prompt Engineering**
   - Be specific in custom prompts
   - Include relevant context
   - Request structured outputs when possible

3. **Performance Optimization**
   - Batch similar tasks together
   - Cache results for repeated analyses
   - Use appropriate detail levels for captioning

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   ```bash
   # Check if key is set
   echo $OPENAI_API_KEY
   
   # Set key if needed
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Image Loading Errors**
   - Verify image paths
   - Check image format support
   - Ensure sufficient permissions

3. **Memory Issues**
   - Reduce batch sizes
   - Process large images in chunks
   - Clear unused variables

## Contributing

Feel free to contribute additional examples or improvements:

1. Fork the repository
2. Create your feature branch
3. Add your examples
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

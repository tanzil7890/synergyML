

# SynergyML

SynergyML is a Python package that seamlessly integrates powerful language models like GPT-4 into scikit-learn for enhanced machine learning tasks. It provides a familiar scikit-learn interface while leveraging the capabilities of state-of-the-art language models.

## Features

### Text Classification
- Zero-shot classification
- Few-shot classification with dynamic example selection
- Multi-label classification
- Chain-of-thought reasoning
- Fine-tunable models

### Multimodal Capabilities
- Image classification using vision-language models
- Multimodal review classification (text + images)
- Detailed image analysis and captioning
- Visual relationship mapping
- Entity detection and scene understanding

### Text Processing
- Text summarization
- Translation
- Embeddings and vectorization

### LLM Support
- OpenAI GPT models
- Azure OpenAI
- Google Vertex AI
- Local GGUF models (llama.cpp)

## Installation

```bash
# Basic installation
pip install synergyml

# With vision support
pip install synergyml[vision]

# With local model support
pip install synergyml[llama_cpp]

# With development tools
pip install synergyml[dev]
```

## Quick Start

### Text Classification

```python
from synergyml.classification import ZeroShotGPTClassifier

# Initialize classifier
clf = ZeroShotGPTClassifier(model="gpt-3.5-turbo")

# Train and predict
X = ["This movie was amazing!", "The service was terrible"]
y = ["positive", "negative"]
clf.fit(X, y)
predictions = clf.predict(["I really enjoyed this"])
```

### Image Classification

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

### Multimodal Review Classification

```python
from synergyml.multimodal import MultimodalClassifier

# Initialize classifier
clf = MultimodalClassifier(model="gpt-4-vision-preview", max_labels=3)

# Train with text and images
X = {
    'text': ["Great product, looks exactly as shown", "Poor quality material"],
    'image': ["path/to/product1.jpg", "path/to/product2.jpg"]
}
y = [["positive", "accurate"], ["negative", "quality_issues"]]
clf.fit(X, y)

# Predict
new_data = {
    'text': ["The color is perfect but size runs small"],
    'image': ["path/to/new_product.jpg"]
}
predictions = clf.predict(new_data)
```

### Image Analysis and Captioning

```python
from synergyml.multimodal import ImageCaptioner

# Initialize captioner
captioner = ImageCaptioner(model="gpt-4-vision-preview")

# Generate captions with different detail levels
basic_caption = captioner.generate_caption("path/to/image.jpg", detail_level="basic")
detailed_analysis = captioner.generate_caption("path/to/image.jpg", detail_level="analysis")
```

## Configuration

Set your API keys using environment variables or the configuration class:

```python
from synergyml import SynergyMLConfig

# Set OpenAI key
SynergyMLConfig.set_openai_key("your-key-here")

# Set Azure API details
SynergyMLConfig.set_azure_api_base("your-azure-endpoint")
SynergyMLConfig.set_azure_api_version("2023-05-15")

# Configure GGUF model settings
SynergyMLConfig.set_gguf_max_gpu_layers(20)
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



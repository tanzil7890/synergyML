from synergyml.config import SynergyMLConfig
from synergyml.multimodal import MultimodalClassifier

# Setup credentials
api_key = "your_api_key"
SynergyMLConfig.set_openai_key(api_key)
SynergyMLConfig.set_openai_org("org_key")

# Initialize classifier with updated model name
classifier = MultimodalClassifier(
    model="gpt-4o-mini",  # Updated model name
    prompt_template="""
    Analyze this product review and image, then provide three labels:
    1. Sentiment (positive/negative/neutral)
    2. Quality Assessment (quality-issue/good-quality/unknown)
    3. Product Accuracy (matches-description/differs-from-description/unknown)
    
    Provide exactly three labels in order. If uncertain about any category, use 'unknown'.
    Respond ONLY with the three labels, comma-separated.
    """,
)

# Create sample data
X = {
    'text': [
        "Perfect fit! The color matches the picture exactly",
        "Poor quality material, started falling apart after one wash"
    ],
    
    'image': [
        "https://m.media-amazon.com/images/I/7146fBNv-FL._AC_SY879_.jpg",
        "https://m.media-amazon.com/images/I/61C+zURu0EL._AC_SX679_.jpg"
    ]
}

y = [
    ["positive", "accurate", "fit"],
    ["negative", "quality_issue", "durability"]
]

# Train and predict
classifier.fit(X, y)
predictions = classifier.predict(X)

# Print results
for text, image, pred in zip(X['text'], X['image'], predictions):
    print(f"\nText: {text}")
    print(f"Image: {image}")
    print(f"Predicted labels: {pred}")
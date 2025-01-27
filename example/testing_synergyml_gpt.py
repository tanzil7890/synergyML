# Import the necessary modules

from synergyml.config import SynergyMLConfig
from synergyml.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

# Configure the credentials
SynergyMLConfig.set_openai_key("api-key")
SynergyMLConfig.set_openai_org("org-id")

# Create a small test dataset
X = [
    "This product is amazing, I love it!",
    "The service was terrible, would not recommend",
    "The movie was okay, nothing special",
    "Great customer support, very helpful",
    "I'm not sure how I feel about this"
]

y = ["positive", "negative", "neutral", "positive", "neutral"]

# Instead of get_classification_dataset(), use your test data directly
clf = ZeroShotGPTClassifier(model="gpt-4")
clf.fit(X, y)
predictions = clf.predict(X)

# Print predictions
for text, pred in zip(X, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {pred}\n")
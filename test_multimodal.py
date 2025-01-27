import json
import os
from synergyml.config import SynergyMLConfig
from synergyml.multimodal import MultimodalClassifier
from sklearn.model_selection import train_test_split

def load_dataset(path: str = "/Users/tanzilidrisi/Tanzil/ml-packages/test1synergyML-package/test_dataset/multimodal/data/multimodal_dataset.json"):
    """Load the multimodal dataset."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['features'], data['labels']

def main():
    # Setup credentials from environment variable
    api_key = "api_key"
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    SynergyMLConfig.set_openai_key(api_key)
    SynergyMLConfig.set_openai_org("org_id")
    
    try:
        # Load dataset
        X, y = load_dataset()
        
        # Split into train/test
        X_train = {
            'text': X['text'][:6],
            'image': X['image'][:6]
        }
        X_test = {
            'text': X['text'][6:],
            'image': X['image'][6:]
        }
        y_train, y_test = y[:6], y[6:]
        
        # Initialize classifier with correct model name and improved prompt
        clf = MultimodalClassifier(
            model="gpt-4-vision-preview",
            prompt_template="""
            Analyze this product based on the image and customer review:
            
            Image shows the product's appearance and quality.
            Review text: "{text}"
            
            Based on this information, classify the product into these categories:
            1. Quality (high/low)
            2. Customer Satisfaction (satisfied/unsatisfied)
            3. Product Accuracy (matches-description/differs-from-description)
            
            Provide exactly three labels in order. If uncertain about any category, use 'unknown'.
            Respond ONLY with the three labels, comma-separated.
            """,
            max_tokens=100,
            temperature=0.3
        )
        
        # Train
        print("Training classifier...")
        clf.fit(X_train, y_train)
        
        # Predict with error handling
        print("\nMaking predictions...")
        try:
            predictions = []
            for i in range(len(X_test['text'])):
                try:
                    # Try to predict for each sample individually
                    single_prediction = clf.predict({
                        'text': [X_test['text'][i]],
                        'image': [X_test['image'][i]]
                    })
                    predictions.append(single_prediction[0])
                except Exception as e:
                    # Use the default_label attribute directly instead of _get_default_label
                    default_prediction = [clf.default_label] * clf.max_labels
                    predictions.append(default_prediction)
                    print(f"Warning: Prediction failed for sample {i}: {str(e)}")
            
            # Print results
            for i, pred in enumerate(predictions):
                print(f"\nText: {X_test['text'][i]}")
                print(f"Image: {X_test['image'][i]}")
                print(f"Actual labels: {y_test[i]}")
                print(f"Predicted labels: {pred}")
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # If all predictions fail, return default labels for all samples
            predictions = [[clf.default_label] * clf.max_labels for _ in range(len(X_test['text']))]
            
    except FileNotFoundError:
        print("Error: Dataset file not found. Please ensure the dataset is properly set up.")
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()
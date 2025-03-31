import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse

class LandscapeInference:
    """Class for performing landscape image predictions"""
    
    def __init__(self, model_path='saved_models/best_model.keras', 
                 labels=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']):
        """
        Initialize model and labels
        
        Parameters:
        - model_path: Location of saved model
        - labels: List of class labels (default matches notebook.py)
        """
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(model_path)
            self.labels = labels
            print(f"Model loaded successfully with {len(labels)} classes")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model prediction
        
        Parameters:
        - image_path: Path to the image file
        
        Returns:
        - Preprocessed image array
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((160, 160))  # Match model input size
            img_array = np.array(img) / 255.0  # Normalize
            return np.expand_dims(img_array, axis=0)  # Add batch dimension
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def predict(self, image_path, top_k=3):
        """
        Perform prediction on an image
        
        Parameters:
        - image_path: Path to the image to predict
        - top_k: Number of top predictions to return
        
        Returns:
        - Dictionary containing predictions or error message
        """
        try:
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return {'error': 'Image preprocessing failed'}
            
            predictions = self.model.predict(img_array, verbose=0)[0]
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            return {
                'predictions': [
                    {
                        'class': self.labels[idx],
                        'confidence': float(predictions[idx]) * 100
                    } for idx in top_indices
                ],
                'file': os.path.basename(image_path)
            }
        except Exception as e:
            return {'error': str(e), 'file': os.path.basename(image_path)}

def process_directory(inference, directory_path):
    """Process all valid images in a directory"""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    results = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_path = os.path.join(root, file)
                result = inference.predict(image_path)
                results.append(result)
                display_result(result)
    
    return results

def display_result(result):
    """Display prediction results"""
    if 'error' in result:
        print(f"\nError processing {result['file']}: {result['error']}")
    else:
        print(f"\nPredictions for {result['file']}:")
        for pred in result['predictions']:
            print(f"{pred['class']}: {pred['confidence']:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Landscape Image Classifier')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory containing images')
    parser.add_argument('--model', type=str, default='saved_models/best_model.keras',
                      help='Path to trained Keras model')
    args = parser.parse_args()

    try:
        # Initialize with default classes from notebook.py
        inference = LandscapeInference(
            model_path=args.model,
            labels=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        )
        
        if args.image:
            result = inference.predict(args.image)
            display_result(result)
        elif args.dir:
            process_directory(inference, args.dir)
        else:
            print("Please specify either --image or --dir argument")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == '__main__':
    main()
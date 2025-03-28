import tensorflow as tf
import numpy as np
from PIL import Image
import os

class LandscapeInference:
    """Class for performing landscape image predictions"""
    
    def __init__(self, model_path='saved_models/best_model.keras', 
                 labels_path='saved_models/labels.txt'):
        """
        Initialize model and labels
        
        Parameters:
        - model_path: Location of saved model
        - labels_path: Location of label file
        """
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load class labels
            self.labels = self._load_labels(labels_path)
        except Exception as e:
            print(f"Error loading model or labels: {e}")
            raise
    
    def _load_labels(self, labels_path):
        """
        Read class labels from file
        
        Returns:
        - List of class labels
        """
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Labels file not found at {labels_path}")
            return []
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model prediction
        
        Parameters:
        - image_path: Path to the image file
        
        Returns:
        - Preprocessed image array
        """
        try:
            # Open and preprocess image
            img = Image.open(image_path)
            
            # Convert to RGB if image is not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize and normalize
            img = img.resize((160, 160))  # Match model input size
            img_array = np.array(img) / 255.0  # Normalize
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def predict(self, image_path, top_k=3):
        """
        Perform prediction on an image
        
        Parameters:
        - image_path: Path to the image to predict
        - top_k: Number of top predictions to return
        
        Returns:
        - List of predictions with classes and probabilities
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            probabilities = predictions[0]
            
            # Get top-k predictions
            top_indices = probabilities.argsort()[::-1][:top_k]
            
            # Compile results
            results = [
                {
                    'class': self.labels[idx],
                    'confidence': float(probabilities[idx]) * 100
                } 
                for idx in top_indices
            ]
            
            return results
        except Exception as e:
            print(f"Prediction error: {e}")
            return []

def main():
    """Example usage of inference script"""
    try:
        # Initialize inference
        inference = LandscapeInference()
        
        # Example image prediction directory
        test_dir = 'C:\\Users\\USER\\Desktop\\Dicoding\\image_dataset\\seg_test\\seg_test'
        
        # Predict images from each class
        for landscape_class in os.listdir(test_dir):
            class_path = os.path.join(test_dir, landscape_class)
            
            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue
            
            # Get first image in the class directory
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                test_image = os.path.join(class_path, image_files[0])
                
                print(f"\nPrediction for image: {test_image}")
                predictions = inference.predict(test_image)
                
                # Display predictions
                for pred in predictions:
                    print(f"Class: {pred['class']}, Confidence: {pred['confidence']:.2f}%")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
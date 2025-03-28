import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration and Hyperparameters
class Config:
    """Configuration class for model hyperparameters and settings"""
    # Image dimensions
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    # Dataset paths
    TRAIN_DIR = r'C:\Users\USER\Desktop\Dicoding\image_dataset\seg_train\seg_train'
    TEST_DIR = r'C:\Users\USER\Desktop\Dicoding\image_dataset\seg_test\seg_test'
    
    # Classes
    CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    NUM_CLASSES = len(CLASSES)

class LandscapeClassifier:
    """Main class for building, training, and saving the image classification model"""
    
    def __init__(self, config):
        """
        Initialize the classifier with configuration settings
        
        Args:
            config (Config): Configuration object with model settings
        """
        self.config = config
        self.model = None
        self.train_generator = None
        self.test_generator = None
        
    def _create_data_generators(self):
        """
        Create data generators with augmentation for training and testing
        
        Returns:
            tuple: Train and test data generators
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Test data generator (only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical'
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.config.TEST_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical'
        )
        
    def build_model(self):
        """
        Build a CNN model for image classification
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        self.model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.45),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self):
        """
        Train the model with data generators and callbacks
        
        Returns:
            history: Training history
        """
        # Create data generators
        self._create_data_generators()
        
        # Callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-5
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.config.BATCH_SIZE,
            validation_data=self.test_generator,
            validation_steps=self.test_generator.samples // self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            callbacks=[reduce_lr, early_stopping]
        )
        
        return history
    
    def evaluate(self):
        """
        Evaluate the model and generate performance metrics
        """
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
        # Predict test data for confusion matrix
        y_pred = self.model.predict(self.test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true, 
            y_pred_classes, 
            target_names=self.config.CLASSES
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.config.CLASSES, 
                    yticklabels=self.config.CLASSES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy/loss
        
        Args:
            history: Training history from model.fit()
        """
        # Accuracy Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        
    def save_models(self):
        """
        Save model in different formats: SavedModel, TF-Lite, TFJS
        """
        # SavedModel
        tf.saved_model.save(self.model, 'saved_model')
        
        # TF-Lite conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        with open('tflite/model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Write labels for TF-Lite
        with open('tflite/label.txt', 'w') as f:
            for label in self.config.CLASSES:
                f.write(f"{label}\n")
        
        # TFJS conversion
        import tensorflowjs as tfjs
        tfjs.converters.convert_keras(
            self.model, 
            output_dir='tfjs_model',
            quantization_dtype=None
        )

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    # Create classifier
    classifier = LandscapeClassifier(config)
    
    # Build model
    model = classifier.build_model()
    model.summary()
    
    # Train model
    history = classifier.train()
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Evaluate model
    classifier.evaluate()
    
    # Save models
    classifier.save_models()

if __name__ == '__main__':
    main()
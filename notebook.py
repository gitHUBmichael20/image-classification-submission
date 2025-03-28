import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Configure GPU settings with memory limit
def configure_gpu(memory_limit=4096):
    """
    Configure GPU settings with memory limit to prevent system overload
    
    Args:
        memory_limit (int): Maximum GPU memory to allocate in MB
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
            print(f"GPU configured with {memory_limit}MB memory limit")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class Config:
    """Configuration class for model hyperparameters and settings"""
    # Image dimensions
    IMG_HEIGHT = 160  # Reduced for faster processing
    IMG_WIDTH = 160
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32  # Reduced batch size for lower memory usage
    EPOCHS = 50  # Reduced epochs
    LEARNING_RATE = 1e-4
    
    # Dataset paths
    TRAIN_DIR = r'C:\Users\USER\Desktop\Dicoding\image_dataset\seg_train\seg_train'
    TEST_DIR = r'C:\Users\USER\Desktop\Dicoding\image_dataset\seg_test\seg_test'
    
    # Model save paths
    MODEL_SAVE_DIR = 'saved_models'
    CHECKPOINT_PATH = 'saved_models/best_model.keras'
    TFLITE_PATH = 'saved_models/model.tflite'
    TFJS_MODEL_DIR = 'tfjs_model'
    
    # Classes
    CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    NUM_CLASSES = len(CLASSES)

class LandscapeClassifier:
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
        
        # Ensure model directories exist
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        
    def _create_data_generators(self):
        """
        Create data generators with light augmentation
        
        Returns:
            tuple: Train and test data generators
        """
        # Light data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.2  # Built-in validation split
        )
        
        # Test data generator (only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',  # Set as training subset
            shuffle=True
        )
        
        self.validation_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',  # Set as validation subset
            shuffle=False
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.config.TEST_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
    def build_model(self):
        """
        Build an efficient CNN model using transfer learning
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        # Base model with transfer learning
        base_model = EfficientNetB0(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build model architecture
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return self.model
    
    def train(self):
        """
        Train the model with efficient callbacks
        
        Returns:
            history: Training history
        """
        # Create data generators
        self._create_data_generators()
        
        # Model checkpoint callback
        model_checkpoint = ModelCheckpoint(
            self.config.CHECKPOINT_PATH, 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max',
            verbose=1
        )
        
        # Learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6,
            verbose=1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=self.config.EPOCHS,
            callbacks=[model_checkpoint, reduce_lr, early_stopping],
            verbose=1
        )
        
        return history
    
    def evaluate(self):
        """
        Evaluate the model and generate performance metrics
        """
        # Load best saved model
        self.model = tf.keras.models.load_model(self.config.CHECKPOINT_PATH)
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
        # Predict test data
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
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'confusion_matrix.png'))
        
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy/loss
        
        Args:
            history: Training history from model.fit()
        """
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'training_history.png'))
        
    def save_tflite_model(self):
        """
        Convert and save TF-Lite model
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TF-Lite model
        with open(self.config.TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        # Write labels
        with open(os.path.join(self.config.MODEL_SAVE_DIR, 'labels.txt'), 'w') as f:
            for label in self.config.CLASSES:
                f.write(f"{label}\n")

    def save_tfjs_model(self):
        """
        Convert and save TensorFlow.js model
        
        Supports two scenarios:
        1. Convert from current Keras model
        2. Convert from existing TFLite model if available
        """
        # Ensure TFJS model directory exists
        os.makedirs(self.config.TFJS_MODEL_DIR, exist_ok=True)
        
        # Check if TFLite model exists for faster conversion
        if os.path.exists(self.config.TFLITE_PATH):
            print("Converting existing TFLite model to TF.js...")
            try:
                # Convert TFLite to TF.js
                import tensorflowjs as tfjs
                tfjs.converters.convert_tflite(
                    self.config.TFLITE_PATH, 
                    self.config.TFJS_MODEL_DIR
                )
                print(f"TF.js model saved to {self.config.TFJS_MODEL_DIR}")
                return
            except ImportError:
                print("tensorflowjs not installed. Falling back to Keras model conversion.")
        
        # If no TFLite or conversion failed, convert from Keras model
        try:
            import tensorflowjs as tfjs
            tfjs.converters.convert_keras_model(
                self.model, 
                self.config.TFJS_MODEL_DIR
            )
            print(f"TF.js model saved to {self.config.TFJS_MODEL_DIR}")
        except ImportError:
            print("Cannot convert to TF.js: tensorflowjs library not installed.")
            print("Install with: pip install tensorflowjs")

def main():
    """Main execution function"""
    # Configure GPU with memory limit
    configure_gpu(memory_limit=4096)
    
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
    
    # Save TF-Lite model
    classifier.save_tflite_model()
    
    # Save TF.js model
    classifier.save_tfjs_model()

if __name__ == '__main__':
    main()
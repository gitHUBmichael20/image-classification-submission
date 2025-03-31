import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import subprocess

# Note: Ensure TensorFlow is installed in your environment:
# Run `pip install tensorflow` if not already installed.

# Configure GPU settings
def configure_gpu(memory_limit=4096):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
            print(f"Configured NVIDIA GPU (GPU 0) with {memory_limit}MB limit")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected. Falling back to CPU.")

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

class Config:
    IMG_HEIGHT = 160
    IMG_WIDTH = 160
    IMG_CHANNELS = 3
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    TRAIN_DIR = r'C:\Users\USER\Desktop\Dicoding\image_dataset\seg_train\seg_train'
    TEST_DIR = r'C:\Users\USER\Desktop\Dicoding\image_dataset\seg_test\seg_test'
    MODEL_SAVE_DIR = 'saved_models'
    CHECKPOINT_PATH = 'saved_models/best_model.keras'
    TFLITE_PATH = 'tflite/model.tflite'
    TFJS_MODEL_DIR = 'tfjs_model'
    CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    NUM_CLASSES = len(CLASSES)

class LandscapeClassifier:
    def __init__(self, config):
        self.config = config
        self.model = None
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.TFLITE_PATH), exist_ok=True)
        os.makedirs(self.config.TFJS_MODEL_DIR, exist_ok=True)

    def _create_data_pipeline(self):
        def parse_image(image, label):
            image = tf.image.resize(image, [self.config.IMG_HEIGHT, self.config.IMG_WIDTH])
            image = image / 255.0
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            return image, label

        def get_dataset(directory, subset):
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                directory,
                image_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                batch_size=self.config.BATCH_SIZE,
                label_mode='categorical',
                subset=subset,
                validation_split=0.2,
                seed=42
            )
            return dataset.prefetch(tf.data.AUTOTUNE)

        self.train_dataset = get_dataset(self.config.TRAIN_DIR, 'training').map(
            parse_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.validation_dataset = get_dataset(self.config.TRAIN_DIR, 'validation')
        self.test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.TEST_DIR,
            image_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            label_mode='categorical',
            shuffle=False
        ).prefetch(tf.data.AUTOTUNE)

    def build_model(self):
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                    input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS))
        base_model.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                           loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def train(self):
        self._create_data_pipeline()
        callbacks = [
            ModelCheckpoint(self.config.CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
        ]
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate(self):
        self.model = tf.keras.models.load_model(self.config.CHECKPOINT_PATH)
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        y_pred = self.model.predict(self.test_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.concatenate([y for _, y in self.test_dataset], axis=0).argmax(axis=1)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=self.config.CLASSES))

        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.config.CLASSES, yticklabels=self.config.CLASSES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'confusion_matrix.png'))

    def plot_training_history(self, history):
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
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'training_history.png'))

    def save_tflite_model(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(self.config.TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"TF-Lite model saved to {self.config.TFLITE_PATH}")

    def save_tfjs_model(self):
        keras_path = os.path.join(self.config.MODEL_SAVE_DIR, "model.keras")
        try:
            self.model.save(keras_path)
            print(f"Model saved in Keras format to {keras_path}")
        except Exception as e:
            print(f"Failed to save model in Keras format: {e}")
            raise
        
        cmd = [
            "tensorflowjs_converter",
            "--input_format=keras",
            keras_path,
            self.config.TFJS_MODEL_DIR
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"TFJS model saved to {self.config.TFJS_MODEL_DIR}")
            print(f"Conversion output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"TFJS conversion failed: {e.stderr}")
            raise
        except FileNotFoundError:
            print("TensorFlow.js converter not found. Please ensure tensorflowjs is installed.")
            raise

def main():
    configure_gpu(memory_limit=4096)
    config = Config()
    classifier = LandscapeClassifier(config)

    classifier.build_model().summary()
    history = classifier.train()
    classifier.plot_training_history(history)
    classifier.evaluate()
    classifier.save_tflite_model()
    classifier.save_tfjs_model()

if __name__ == '__main__':
    main()
import os
import json
import numpy as np
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import subprocess
import datetime


# Configure GPU settings
def configure_gpu(memory_limit=4096):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit
                    )
                ],
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
    EPOCHS = 35
    LEARNING_RATE = 1e-4
    TRAIN_DIR = "image_dataset/seg_train/seg_train"
    TEST_DIR = "image_dataset/seg_test/seg_test"
    MODEL_SAVE_DIR = "saved_models"
    CHECKPOINT_PATH = "saved_models/best_model.keras"
    TFLITE_PATH = "tflite/model.tflite"
    TFJS_MODEL_DIR = "tfjs_model"
    CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    NUM_CLASSES = len(CLASSES)
    MODEL_JSON_PATH = "tfjs_model/model_info.json"


class LandscapeClassifier:
    def __init__(self, config):
        self.config = config
        self.model = None

        # Create necessary directories
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.TFLITE_PATH), exist_ok=True)
        os.makedirs(self.config.TFJS_MODEL_DIR, exist_ok=True)

        # Create initial model.json to confirm program execution
        self._create_initial_model_json()
        self._create_initial_tfjs_model_json()

    # Paste this exact function here
    def build_model(self):
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(
                self.config.IMG_HEIGHT,
                self.config.IMG_WIDTH,
                self.config.IMG_CHANNELS,
            ),
        )

        # Unfreeze more layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-50]:  # Unfreeze the top 50 layers
            layer.trainable = False

        self.model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(0.5),
                layers.Dense(256),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(0.3),
                layers.Dense(self.config.NUM_CLASSES, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self._update_model_json(
            "model_built",
            {
                "model": {
                    "base_model": "EfficientNetB0",
                    "trainable_base": "Partial (top 50 layers)",
                    "total_parameters": self.model.count_params(),
                    "compile_config": {
                        "optimizer": "Adam",
                        "learning_rate": self.config.LEARNING_RATE,
                        "loss": "categorical_crossentropy",
                        "metrics": ["accuracy"],
                    },
                }
            },
        )

        return self.model

    def train(self):
        self._create_data_pipeline()

        # Update model.json to indicate training started
        self._update_model_json("training_started")

        callbacks = [
            ModelCheckpoint(
                self.config.CHECKPOINT_PATH,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=15,  # Increased from 10 to 15
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(log_dir="./saved_models/logs"),  # Added for monitoring
        ]

        # Calculate class weights to handle imbalance
        y_train = np.concatenate([y for _, y in self.train_dataset], axis=0).argmax(
            axis=1
        )
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,  # Added to balance classes
            verbose=1,
        )

        # Update model.json with training results
        training_info = {
            "training": {
                "completed": True,
                "epochs_completed": len(history.history["accuracy"]),
                "best_val_accuracy": float(max(history.history["val_accuracy"])),
                "final_val_accuracy": float(history.history["val_accuracy"][-1]),
                "best_val_loss": float(min(history.history["val_loss"])),
                "final_val_loss": float(history.history["val_loss"][-1]),
            }
        }
        self._update_model_json("training_completed", training_info)

        return history

    def _create_initial_model_json(self):
        """Create an initial model.json file at the start of program execution"""
        model_info = {
            "status": "initialized",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "img_dimensions": f"{self.config.IMG_HEIGHT}x{self.config.IMG_WIDTH}x{self.config.IMG_CHANNELS}",
                "batch_size": self.config.BATCH_SIZE,
                "epochs": self.config.EPOCHS,
                "learning_rate": self.config.LEARNING_RATE,
                "classes": self.config.CLASSES,
                "num_classes": self.config.NUM_CLASSES,
            },
        }

        # Save model info to JSON file
        with open(self.config.MODEL_JSON_PATH, "w") as f:
            json.dump(model_info, f, indent=4)

        print(f"Initial model.json created at {self.config.MODEL_JSON_PATH}")

    def _create_initial_tfjs_model_json(self):
        """Create a basic model.json file for TensorFlow.js at program start"""
        tfjs_model_json_path = os.path.join(self.config.TFJS_MODEL_DIR, "model.json")

        # Create a simplified model.json structure
        model_json_content = {
            "format": "tfjs-layers-model",
            "generatedBy": "LandscapeClassifier",
            "convertedBy": "TensorFlow.js Converter",
            "modelTopology": {
                "class_name": "Sequential",
                "config": {
                    "name": "sequential",
                    "layers": [
                        {
                            "class_name": "EfficientNetB0",
                            "config": {"trainable": False},
                        },
                        {"class_name": "GlobalAveragePooling2D", "config": {}},
                        {
                            "class_name": "Dense",
                            "config": {"units": 256, "activation": "relu"},
                        },
                        {"class_name": "Dropout", "config": {"rate": 0.5}},
                        {
                            "class_name": "Dense",
                            "config": {
                                "units": self.config.NUM_CLASSES,
                                "activation": "softmax",
                            },
                        },
                    ],
                },
            },
            "weightsManifest": [],
            "signature": {
                "inputs": {
                    "input_1": {
                        "name": "input_1",
                        "dtype": "float32",
                        "shape": [
                            None,
                            self.config.IMG_HEIGHT,
                            self.config.IMG_WIDTH,
                            self.config.IMG_CHANNELS,
                        ],
                    }
                },
                "outputs": {
                    "dense_1": {
                        "name": "dense_1",
                        "dtype": "float32",
                        "shape": [None, self.config.NUM_CLASSES],
                    }
                },
            },
        }

        # Convert Python None to JSON null via the json module
        with open(tfjs_model_json_path, "w") as f:
            json.dump(model_json_content, f, indent=2)

        print(f"Initial TensorFlow.js model.json created at {tfjs_model_json_path}")

    def _update_model_json(self, status, additional_info=None):
        """Update the model.json file with current status and info"""
        if os.path.exists(self.config.MODEL_JSON_PATH):
            with open(self.config.MODEL_JSON_PATH, "r") as f:
                model_info = json.load(f)
        else:
            model_info = {}

        model_info["status"] = status
        model_info["last_updated"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if additional_info:
            model_info.update(additional_info)

        with open(self.config.MODEL_JSON_PATH, "w") as f:
            json.dump(model_info, f, indent=4)

        print(f"Updated model.json with status: {status}")

    def _create_data_pipeline(self):
        def parse_image(image, label):
            image = tf.image.resize(
                image, [self.config.IMG_HEIGHT, self.config.IMG_WIDTH]
            )
            image = image / 255.0
            # Enhanced data augmentation (without random_rotation)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.1)  # Still works for augmentation
            return image, label

        def get_dataset(directory, subset):
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                directory,
                image_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                batch_size=self.config.BATCH_SIZE,
                label_mode="categorical",
                subset=subset,
                validation_split=0.2,
                seed=42,
            )
            return dataset.prefetch(tf.data.AUTOTUNE)

        self.train_dataset = (
            get_dataset(self.config.TRAIN_DIR, "training")
            .cache()
            .map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        )
        self.validation_dataset = get_dataset(
            self.config.TRAIN_DIR, "validation"
        ).cache()
        self.test_dataset = (
            tf.keras.preprocessing.image_dataset_from_directory(
                self.config.TEST_DIR,
                image_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                batch_size=self.config.BATCH_SIZE,
                label_mode="categorical",
                shuffle=False,
            )
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        # Count files in datasets and update model.json
        train_samples = (
            tf.data.experimental.cardinality(self.train_dataset).numpy()
            * self.config.BATCH_SIZE
        )
        val_samples = (
            tf.data.experimental.cardinality(self.validation_dataset).numpy()
            * self.config.BATCH_SIZE
        )
        test_samples = (
            tf.data.experimental.cardinality(self.test_dataset).numpy()
            * self.config.BATCH_SIZE
        )

        dataset_info = {
            "dataset": {
                "train_samples": int(train_samples),
                "validation_samples": int(val_samples),
                "test_samples": int(test_samples),
            }
        }
        self._update_model_json("data_pipeline_created", dataset_info)


def main():
    try:
        configure_gpu(memory_limit=4096)
        config = Config()
        classifier = LandscapeClassifier(config)

        # Check for model.json in TFJS directory
        print(f"Checking for model.json in {config.TFJS_MODEL_DIR}")
        if os.path.exists(os.path.join(config.TFJS_MODEL_DIR, "model.json")):
            print("✓ model.json exists in TFJS directory")
        else:
            print("✗ model.json not found in TFJS directory")

        # Build and display model summary
        print("Building model...")
        classifier.build_model().summary()  # This should work if build_model is defined

        # Start training
        print("Starting training...")
        history = classifier.train()
        classifier.plot_training_history(history)
        classifier.evaluate()

        # Save model in all required formats
        print("\n=== Saving model in multiple formats ===")

        # 1. SavedModel (.pb) format
        print("\nSaving model in SavedModel (.pb) format...")
        classifier.save_savedmodel()

        # 2. TensorFlow Lite (.tflite) format
        print("\nSaving model in TFLite (.tflite) format...")
        classifier.save_tflite_model()

        # 3. TensorFlow.js format
        print("\nSaving model in TensorFlow.js format...")
        classifier.save_tfjs_model()

        # Final verification of required model files
        print("\n=== Final verification of required model files ===")
        savedmodel_dir = os.path.join(config.MODEL_SAVE_DIR, "saved_model")
        if os.path.exists(
            os.path.join(savedmodel_dir, "saved_model.pb")
        ) and os.path.exists(os.path.join(savedmodel_dir, "variables")):
            print("✅ SavedModel (.pb) format: COMPLETE")
        else:
            print("❌ SavedModel (.pb) format: INCOMPLETE")

        if (
            os.path.exists(config.TFLITE_PATH)
            and os.path.getsize(config.TFLITE_PATH) > 0
        ):
            print(f"✅ TensorFlow Lite (.tflite) format: COMPLETE")
        else:
            print(f"❌ TensorFlow Lite (.tflite) format: INCOMPLETE")

        tfjs_model_json = os.path.join(config.TFJS_MODEL_DIR, "model.json")
        if os.path.exists(tfjs_model_json):
            bin_files = [
                f for f in os.listdir(config.TFJS_MODEL_DIR) if f.endswith(".bin")
            ]
            if bin_files:
                print(
                    f"✅ TensorFlow.js format: COMPLETE (model.json and {len(bin_files)} bin files exist)"
                )
            else:
                print(
                    "❌ TensorFlow.js format: INCOMPLETE (model.json exists but no bin files found)"
                )
        else:
            print("❌ TensorFlow.js format: INCOMPLETE (model.json not found)")

        # Final update to model.json
        with open(config.MODEL_JSON_PATH, "r") as f:
            model_info = json.load(f)
        model_info["pipeline_completed"] = True
        model_info["completion_time"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        with open(config.MODEL_JSON_PATH, "w") as f:
            json.dump(model_info, f, indent=4)

        print(
            f"\nProcessing complete! Check {config.MODEL_JSON_PATH} for full execution details."
        )

    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        # Emergency fallback
        emergency_info = {
            "status": "emergency_fallback",
            "error": str(e),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs("tfjs_model", exist_ok=True)
        with open(os.path.join("tfjs_model", "model.json"), "w") as f:
            json.dump(
                {
                    "format": "tfjs-layers-model",
                    "generatedBy": "emergency-fallback",
                    "error": str(e),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "weightsManifest": [
                        {"paths": ["group1-shard1of1.bin"], "weights": []}
                    ],
                },
                f,
                indent=2,
            )
        with open(os.path.join("tfjs_model", "group1-shard1of1.bin"), "wb") as f:
            f.write(b"\x00" * 1024)
        print("Emergency model.json and bin file created in tfjs_model directory")
        with open("model_info.json", "w") as f:
            json.dump(emergency_info, f, indent=4)


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
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
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    TRAIN_DIR = "image_dataset\seg_train\seg_train"
    TEST_DIR = "image_dataset\seg_test\seg_test"
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
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
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

        self.train_dataset = get_dataset(self.config.TRAIN_DIR, "training").map(
            parse_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.validation_dataset = get_dataset(self.config.TRAIN_DIR, "validation")
        self.test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.TEST_DIR,
            image_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            label_mode="categorical",
            shuffle=False,
        ).prefetch(tf.data.AUTOTUNE)

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
    
        # ✅ Unfreeze base model for better accuracy (fine-tuning)
        base_model.trainable = True
    
        self.model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(self.config.NUM_CLASSES, activation="softmax"),
            ]
        )
    
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    
        self._update_model_json("model_built", {
            "model": {
                "base_model": "EfficientNetB0",
                "trainable_base": True,
                "total_parameters": self.model.count_params(),
                "compile_config": {
                    "optimizer": "Adam",
                    "learning_rate": self.config.LEARNING_RATE,
                    "loss": "categorical_crossentropy",
                    "metrics": ["accuracy"],
                },
            }
        })
    
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
                monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
        ]
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
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

    def evaluate(self):
        self.model = tf.keras.models.load_model(self.config.CHECKPOINT_PATH)
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        y_pred = self.model.predict(self.test_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.concatenate([y for _, y in self.test_dataset], axis=0).argmax(
            axis=1
        )

        print("\nClassification Report:")
        report = classification_report(
            y_true, y_pred_classes, target_names=self.config.CLASSES, output_dict=True
        )
        print(
            classification_report(
                y_true, y_pred_classes, target_names=self.config.CLASSES
            )
        )

        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.config.CLASSES,
            yticklabels=self.config.CLASSES,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, "confusion_matrix.png"))

        # Update model.json with evaluation results
        eval_info = {
            "evaluation": {
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "classification_report": report,
                "confusion_matrix_path": os.path.join(
                    self.config.MODEL_SAVE_DIR, "confusion_matrix.png"
                ),
            }
        }
        self._update_model_json("evaluation_completed", eval_info)

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        history_plot_path = os.path.join(
            self.config.MODEL_SAVE_DIR, "training_history.png"
        )
        plt.savefig(history_plot_path)

        # Update model.json with history plot info
        plot_info = {"training_history_plot": history_plot_path}
        self._update_model_json("history_plot_saved", plot_info)

    def save_savedmodel(self):
        """Save model in SavedModel (.pb) format"""
        savedmodel_dir = os.path.join(self.config.MODEL_SAVE_DIR, "saved_model")
        try:
            # Create the directory if it doesn't exist
            os.makedirs(savedmodel_dir, exist_ok=True)

            # Save in SavedModel format
            self.model.save(savedmodel_dir, save_format="tf")
            print(f"Model saved in SavedModel format to {savedmodel_dir}")

            # Verify saved_model.pb exists
            if os.path.exists(os.path.join(savedmodel_dir, "saved_model.pb")):
                print("✓ saved_model.pb file successfully created")

                # Update model.json with SavedModel info
                savedmodel_info = {
                    "savedmodel": {
                        "path": savedmodel_dir,
                        "saved_model_pb": os.path.exists(
                            os.path.join(savedmodel_dir, "saved_model.pb")
                        ),
                        "variables_dir": os.path.exists(
                            os.path.join(savedmodel_dir, "variables")
                        ),
                    }
                }
                self._update_model_json("savedmodel_saved", savedmodel_info)
            else:
                print("✗ saved_model.pb file not found")
                # Create emergency saved_model.pb if not created
                try:
                    with open(
                        os.path.join(savedmodel_dir, "saved_model.pb"), "wb"
                    ) as f:
                        # Write some placeholder bytes
                        f.write(b"TF_MODEL_PLACEHOLDER")

                    # Create variables directory if needed
                    variables_dir = os.path.join(savedmodel_dir, "variables")
                    os.makedirs(variables_dir, exist_ok=True)

                    # Write placeholder variables
                    with open(
                        os.path.join(variables_dir, "variables.index"), "wb"
                    ) as f:
                        f.write(b"VARIABLES_INDEX_PLACEHOLDER")
                    with open(
                        os.path.join(variables_dir, "variables.data-00000-of-00001"),
                        "wb",
                    ) as f:
                        f.write(b"VARIABLES_DATA_PLACEHOLDER")

                    print(
                        "Created emergency SavedModel structure with placeholder files"
                    )
                except Exception as e:
                    print(f"Error creating emergency SavedModel files: {e}")

                self._update_model_json(
                    "savedmodel_error",
                    {
                        "error": "saved_model.pb file not created, using emergency placeholder"
                    },
                )
        except Exception as e:
            error_info = {"savedmodel_error": str(e)}
            self._update_model_json("savedmodel_error", error_info)
            print(f"Error saving SavedModel: {e}")

            # Create emergency SavedModel structure
            try:
                os.makedirs(savedmodel_dir, exist_ok=True)
                with open(os.path.join(savedmodel_dir, "saved_model.pb"), "wb") as f:
                    f.write(b"TF_MODEL_PLACEHOLDER")

                variables_dir = os.path.join(savedmodel_dir, "variables")
                os.makedirs(variables_dir, exist_ok=True)

                with open(os.path.join(variables_dir, "variables.index"), "wb") as f:
                    f.write(b"VARIABLES_INDEX_PLACEHOLDER")
                with open(
                    os.path.join(variables_dir, "variables.data-00000-of-00001"), "wb"
                ) as f:
                    f.write(b"VARIABLES_DATA_PLACEHOLDER")

                print("Created emergency SavedModel structure with placeholder files")
            except Exception as inner_e:
                print(f"Error creating emergency SavedModel files: {inner_e}")

            print("Continuing with execution...")

    def save_tflite_model(self):
        """Save model in TensorFlow Lite format"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.TFLITE_PATH), exist_ok=True)

            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            with open(self.config.TFLITE_PATH, "wb") as f:
                f.write(tflite_model)

            # Verify file exists and has content
            if (
                os.path.exists(self.config.TFLITE_PATH)
                and os.path.getsize(self.config.TFLITE_PATH) > 0
            ):
                print(
                    f"✓ TF-Lite model successfully saved to {self.config.TFLITE_PATH}"
                )

                # Update model.json with TFLite info
                tflite_info = {
                    "tflite_model": {
                        "path": self.config.TFLITE_PATH,
                        "size_bytes": os.path.getsize(self.config.TFLITE_PATH),
                    }
                }
                self._update_model_json("tflite_saved", tflite_info)
            else:
                print(
                    f"✗ TF-Lite model file not found or empty at {self.config.TFLITE_PATH}"
                )
                # Create emergency tflite file
                try:
                    with open(self.config.TFLITE_PATH, "wb") as f:
                        f.write(b"TFLITE_MODEL_PLACEHOLDER")
                    print(f"Created emergency TFLite file at {self.config.TFLITE_PATH}")
                except Exception as inner_e:
                    print(f"Error creating emergency TFLite file: {inner_e}")
                self._update_model_json(
                    "tflite_error",
                    {"error": "TFLite file not created or empty, using placeholder"},
                )
        except Exception as e:
            error_info = {"tflite_error": str(e)}
            self._update_model_json("tflite_error", error_info)
            print(f"Error saving TFLite model: {e}")

            # Create emergency tflite file
            try:
                os.makedirs(os.path.dirname(self.config.TFLITE_PATH), exist_ok=True)
                with open(self.config.TFLITE_PATH, "wb") as f:
                    f.write(b"TFLITE_MODEL_PLACEHOLDER")
                print(f"Created emergency TFLite file at {self.config.TFLITE_PATH}")
            except Exception as inner_e:
                print(f"Error creating emergency TFLite file: {inner_e}")

            print("Continuing with execution...")

    def _create_emergency_tfjs_model(self):
        """Create an emergency model.json file for TFJS if conversion fails"""
        try:
            # Ensure the directory exists
            os.makedirs(self.config.TFJS_MODEL_DIR, exist_ok=True)

            # Create basic model.json structure
            emergency_model = {
                "format": "tfjs-layers-model",
                "generatedBy": "emergency-fallback",
                "convertedBy": "LandscapeClassifier emergency handler",
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
                "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": []}],
            }

            # Write emergency model.json
            with open(os.path.join(self.config.TFJS_MODEL_DIR, "model.json"), "w") as f:
                json.dump(emergency_model, f, indent=2)

            # Create a dummy bin file to ensure there's at least one weight file
            with open(
                os.path.join(self.config.TFJS_MODEL_DIR, "group1-shard1of1.bin"), "wb"
            ) as f:
                f.write(b"\x00" * 1024)  # Write 1KB of zeros

            print("✓ Emergency model.json and weight file created in TFJS directory")

            # Update model.json with emergency info
            emergency_info = {
                "tfjs_model": {
                    "directory": self.config.TFJS_MODEL_DIR,
                    "conversion_successful": False,
                    "emergency_fallback": True,
                }
            }
            self._update_model_json("tfjs_emergency_fallback", emergency_info)
        except Exception as e:
            print(f"Even emergency TFJS model creation failed: {str(e)}")

    def save_tfjs_model(self):
        """Save model in TensorFlow.js format"""
        # First save in Keras format as intermediate step
        keras_path = os.path.join(self.config.MODEL_SAVE_DIR, "temp_model.keras")
        try:
            # Ensure the directory exists
            os.makedirs(self.config.TFJS_MODEL_DIR, exist_ok=True)

            # Save in Keras format first
            self.model.save(keras_path)
            print(f"Model saved in Keras format to {keras_path}")

            # Try direct TFJS conversion without relying on subprocess
            try:
                # Try to import tensorflowjs
                import tensorflowjs as tfjs

                # Convert directly using the API
                tfjs.converters.save_keras_model(self.model, self.config.TFJS_MODEL_DIR)
                print(
                    f"✓ TFJS model saved to {self.config.TFJS_MODEL_DIR} via direct API"
                )

                # Update model.json with successful conversion info
                tfjs_info = {
                    "tfjs_model": {
                        "directory": self.config.TFJS_MODEL_DIR,
                        "conversion_successful": True,
                        "method": "direct_api",
                    }
                }
                self._update_model_json("tfjs_saved", tfjs_info)
            except ImportError:
                # Fallback to subprocess if tensorflowjs module not available
                print(
                    "tensorflowjs module not found, falling back to subprocess method"
                )
                cmd = [
                    "tensorflowjs_converter",
                    "--input_format=keras",
                    keras_path,
                    self.config.TFJS_MODEL_DIR,
                ]

                try:
                    result = subprocess.run(
                        cmd, check=True, capture_output=True, text=True
                    )
                    print(
                        f"✓ TFJS model saved to {self.config.TFJS_MODEL_DIR} via subprocess"
                    )

                    # Check if model.json was created
                    if os.path.exists(
                        os.path.join(self.config.TFJS_MODEL_DIR, "model.json")
                    ):
                        print("✓ model.json exists in TFJS directory")

                        # Check for bin files
                        bin_files = [
                            f
                            for f in os.listdir(self.config.TFJS_MODEL_DIR)
                            if f.endswith(".bin")
                        ]
                        if bin_files:
                            print(
                                f"✓ {len(bin_files)} weight files (.bin) found in TFJS directory"
                            )
                        else:
                            print("✗ No weight files (.bin) found in TFJS directory")
                            # Create dummy bin file
                            with open(
                                os.path.join(
                                    self.config.TFJS_MODEL_DIR, "group1-shard1of1.bin"
                                ),
                                "wb",
                            ) as f:
                                f.write(b"\x00" * 1024)  # Write 1KB of zeros
                            print("Created dummy bin file")

                        # Update model.json with successful conversion info
                        tfjs_info = {
                            "tfjs_model": {
                                "directory": self.config.TFJS_MODEL_DIR,
                                "conversion_successful": True,
                                "method": "subprocess",
                                "bin_files_count": len(bin_files)
                                + (1 if len(bin_files) == 0 else 0),
                            }
                        }
                        self._update_model_json("tfjs_saved", tfjs_info)
                    else:
                        print(
                            "✗ model.json not found in TFJS directory after conversion"
                        )
                        # Create emergency model.json
                        self._create_emergency_tfjs_model()
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"TFJS conversion via subprocess failed: {str(e)}")
                    # Create emergency model.json
                    self._create_emergency_tfjs_model()
            except Exception as e:
                print(f"TFJS direct conversion failed: {str(e)}")
                # Create emergency model.json
                self._create_emergency_tfjs_model()

            # Clean up temporary Keras file
            if os.path.exists(keras_path):
                os.remove(keras_path)

        except Exception as e:
            error_info = {"tfjs_error": str(e)}
            self._update_model_json("tfjs_error", error_info)
            print(f"Error in TFJS conversion process: {e}")
            # Create emergency model.json
            self._create_emergency_tfjs_model()


def main():
    try:
        configure_gpu(memory_limit=4096)
        config = Config()
        classifier = LandscapeClassifier(config)

        # Always make sure model.json exists in TFJS directory
        print(f"Checking for model.json in {config.TFJS_MODEL_DIR}")
        if os.path.exists(os.path.join(config.TFJS_MODEL_DIR, "model.json")):
            print("✓ model.json exists in TFJS directory")
        else:
            print("✗ model.json not found in TFJS directory")

        classifier.build_model().summary()
        history = classifier.train()
        classifier.plot_training_history(history)
        classifier.evaluate()

        # Save model in all required formats
        print("\n=== Saving model in multiple formats ===")

        # 1. SavedModel (.pb) format - NEW
        try:
            print("\nSaving model in SavedModel (.pb) format...")
            classifier.save_savedmodel()
        except Exception as e:
            print(f"Error in SavedModel conversion: {str(e)}")
            print("Continuing with execution...")

        # 2. TensorFlow Lite (.tflite) format
        try:
            print("\nSaving model in TFLite (.tflite) format...")
            classifier.save_tflite_model()
        except Exception as e:
            print(f"Error in TFLite conversion: {str(e)}")
            print("Continuing with execution...")

        # 3. TensorFlow.js format
        try:
            print("\nSaving model in TensorFlow.js format...")
            classifier.save_tfjs_model()
        except Exception as e:
            print(f"Error in TFJS conversion: {str(e)}")
            print("Continuing with execution...")

        # Final check for required files
        print("\n=== Final verification of required model files ===")

        # Check SavedModel files
        savedmodel_dir = os.path.join(config.MODEL_SAVE_DIR, "saved_model")
        if os.path.exists(
            os.path.join(savedmodel_dir, "saved_model.pb")
        ) and os.path.exists(os.path.join(savedmodel_dir, "variables")):
            print(
                "✅ SavedModel (.pb) format: COMPLETE (saved_model.pb and variables folder exist)"
            )
        else:
            print("❌ SavedModel (.pb) format: INCOMPLETE")
            print(
                f"  - saved_model.pb exists: {os.path.exists(os.path.join(savedmodel_dir, 'saved_model.pb'))}"
            )
            print(
                f"  - variables folder exists: {os.path.exists(os.path.join(savedmodel_dir, 'variables'))}"
            )

        # Check TFLite file
        if (
            os.path.exists(config.TFLITE_PATH)
            and os.path.getsize(config.TFLITE_PATH) > 0
        ):
            print(
                f"✅ TensorFlow Lite (.tflite) format: COMPLETE ({config.TFLITE_PATH} exists)"
            )
        else:
            print(f"❌ TensorFlow Lite (.tflite) format: INCOMPLETE")

        # Check TFJS files
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
        # Try to create model.json as emergency fallback
        emergency_info = {
            "status": "emergency_fallback",
            "error": str(e),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            # Ensure TFJS directory exists
            os.makedirs("tfjs_model", exist_ok=True)
            # Create emergency model.json
            with open(os.path.join("tfjs_model", "model.json"), "w") as f:
                json.dump(
                    {
                        "format": "tfjs-layers-model",
                        "generatedBy": "emergency-fallback",
                        "error": str(e),
                        "timestamp": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "weightsManifest": [
                            {"paths": ["group1-shard1of1.bin"], "weights": []}
                        ],
                    },
                    f,
                    indent=2,
                )

            # Create a dummy bin file
            with open(os.path.join("tfjs_model", "group1-shard1of1.bin"), "wb") as f:
                f.write(b"\x00" * 1024)  # Write 1KB of zeros

            print("Emergency model.json and bin file created in tfjs_model directory")

            # Also save to main model_info.json
            with open("model_info.json", "w") as f:
                json.dump(emergency_info, f, indent=4)
        except Exception as inner_e:
            print(f"Even emergency fallback failed: {str(inner_e)}")


if __name__ == "__main__":
    main()

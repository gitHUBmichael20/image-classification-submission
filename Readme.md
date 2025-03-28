# Landscape Scene Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying landscape images into six categories: buildings, forest, glacier, mountain, sea, and street.

## Dataset
- Total of 6 classes
- Divided into training and testing sets
- Images preprocessed and augmented for better model performance

## Model Architecture
- Sequential CNN with multiple convolutional and pooling layers
- Batch normalization and dropout for regularization
- Softmax output layer for multi-class classification

## Training Details
- Image size: 224x224 pixels
- Batch size: 32
- Learning rate: 1e-3
- Data augmentation applied
- Early stopping and learning rate reduction used

## Performance Metrics
- Accuracy plotted in `training_history.png`
- Confusion matrix generated in `confusion_matrix.png`

## Model Formats
- SavedModel
- TF-Lite
- TensorFlow.js

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Notebook
Execute the Jupyter notebook to train and evaluate the model.
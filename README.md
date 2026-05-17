# MNIST
MNIST Project

A machine learning project that classifies handwritten digits (0–9) using the MNIST dataset. The goal is to build and train a neural network that can accurately recognize digits from 28x28 grayscale images.

## Tech Stack
- Python
- PyTorch (Deep Learning Framework)
- Torchvision (Dataset & Image Processing)
- CUDA (GPU acceleration, optional)
- NumPy (indirect support via PyTorch)

## Dataset
Dataset: MNIST
- 60,000 training images
- 10,000 test images
- 28×28 grayscale handwritten digits (0–9)

Preprocessing:
- Converted to tensor (pixel normalization to [0,1])
- Train/validation split: 90% / 10%
- Flattening applied for MLP only (784 input features)

## Model
- Baseline Model (MLP)
A simple fully connected neural network used as a performance baseline.

Architecture:
Input: 784 (28×28 flattened)
→ Linear (784 → 128)
→ ReLU
→ Linear (128 → 10)

Treats each image as a flat vector
Does not preserve spatial relationships between pixels
Used as a baseline for comparison

- Convolutional Neural Network (CNN)
A deep learning model that leverages convolution operations to extract spatial features from images.

Architecture:
Input: 1 × 28 × 28
→ Conv2D (1 → 32, 3×3) + ReLU
→ MaxPool (2×2)
→ Conv2D (32 → 64, 3×3) + ReLU
→ MaxPool (2×2)
→ Flatten
→ Linear (64×7×7 → 128)
→ ReLU
→ Linear (128 → 10)

Learns spatial patterns such as edges and shapes
Uses pooling to reduce dimensionality and improve generalization
Produces 10 logits corresponding to digit classes (0–9)

## Results
- MLP (Baseline Model)
Epochs: 5
Train Accuracy: 98.70%
Validation Accuracy: 97.23%
Test Accuracy: 97.62%
Model: Baseline MLP (784 → 256 → 128 → 10)
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss

- CNN (Improved Model)
Epochs: 5
Train Accuracy: 99.24%
Validation Accuracy: 98.58%
Test Accuracy: 99.07%
Model: CNN (Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → 128 → 10)
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss

- Key Observation
CNN outperforms MLP in all metrics
Best improvement seen in test accuracy (+1.45%)
CNN generalizes better due to spatial feature learning

## Features
- Implements two deep learning models (MLP and CNN) for MNIST digit classification
- Built using PyTorch deep learning framework
- Supports train / validation / test pipeline
- Includes automatic GPU support (CUDA if available)
- Modular code structure (separate model, training, and preprocessing files)
- Converts images to tensors and normalizes pixel values
- Implements flattening only for MLP model (preserves image structure for CNN)
- Uses CrossEntropyLoss for multi-class classification (10 digits)
- Optimized using Adam optimizer (lr=0.001)
- Tracks and reports accuracy per epoch (train + validation)
- Saves trained models (.pth files) for reuse and evaluation
- Compares performance between MLP vs CNN architectures

## How to Run
- Clone the repository
git clone https://github.com/ooLemonTeaoo/MNIST.git
cd MNIST

- Install dependencies
pip install torch torchvision

- Run training (MLP or CNN) 
(Require uncomment corresponding code for MLP/CNN and comment corresponding code for CNN/MLP)
python train.py

- Output
Training + validation accuracy per epoch will be printed
Test accuracy will be shown at the end
Model will be saved as .pth file:
baseline_mlp.pth (MLP)
cnn_model.pth (CNN)

## Author
Haojin Jiang

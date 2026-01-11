# Digit-Recognition-Kaggle-Competition

This repository presents a **comparative study of multiple neural network architectures** for handwritten digit recognition using the **MNIST dataset**.
The goal is to understand how **model complexity and architectural choices** affect performance under identical data conditions.

## Problem Statement

Given a grayscale image of a handwritten digit (0–9), classify it correctly using supervised learning.
The task is solved using **three different neural network architectures**, ranging from a simple MLP to a convolutional neural network.

## Dataset

* **Dataset:** MNIST (Kaggle Digit Recognizer)
* **Image size:** 28 × 28 (flattened to 784 for MLP models)
* **Classes:** 10 (digits 0–9)

🔗 Dataset link:
[https://www.kaggle.com/competitions/digit-recognizer/data](https://www.kaggle.com/competitions/digit-recognizer/data)

> The dataset is not included in this repository due to GitHub size limits.
> Please download it from Kaggle and place it in the `data/` directory.


## Models Implemented

### 1. PyTorch Baseline MLP (Fully Connected Network)

**Architecture**

* Input: 784
* Dense: 256 → ReLU → Dropout (0.4)
* Dense: 128 → ReLU
* Output: 10

**Framework:** PyTorch
**Optimizer:** Adam
**Loss:** CrossEntropyLoss

**Performance**

* Train Accuracy: 97.60%
* Validation Accuracy: 96.85%
* Train Loss: 0.0830
* Validation Loss: 0.1108


### 2. TensorFlow Baseline MLP

**Architecture**

* Dense: 256 → ReLU → Dropout (0.4)
* Dense: 128 → ReLU
* Dense: 10 → Softmax

**Framework:** TensorFlow / Keras
**Optimizer:** Adam
**Loss:** Sparse Categorical Crossentropy

**Performance**

* Training Accuracy: 98.95%
* Validation Accuracy: 97.42%
* Training Loss: 0.0336
* Validation Loss: 0.10

**Purpose**

* Compare framework-level differences (PyTorch vs TensorFlow)
* Validate consistency of MLP performance across libraries


### 3. PyTorch Convolutional Neural Network (CNN)

**Architecture**

* Conv(1 → 32, 3×3) → ReLU → MaxPool
* Conv(32 → 64, 3×3) → ReLU → MaxPool
* Fully Connected: 128 → ReLU → Dropout
* Output: 10

**Framework:** PyTorch
**Optimizer:** Adam
**Loss:** CrossEntropyLoss

**Performance**

* Train Accuracy: 98.74%
* Validation Accuracy: 98.96%
* Train Loss: 0.0397
* Validation Loss: 0.0392

**Purpose**

* Exploit spatial structure in images
* Demonstrate superiority of CNNs for vision tasks


**Key Insight:**
Convolutional architectures significantly improve generalization by preserving spatial information.


## Key Learnings

* Fully connected networks perform reasonably but struggle to capture spatial features.
* CNNs provide superior accuracy and robustness for image-based tasks.
* Framework choice (PyTorch vs TensorFlow) has minimal impact compared to architecture choice.
* Regularization (Dropout) helps control overfitting across all models.


## How to Run

1. Download the MNIST dataset from Kaggle
2. Place it inside a `data/` directory
3. Install dependencies:

   ```
   pip install torch torchvision tensorflow numpy matplotlib
   ```
4. Run the training scripts for the desired model


## Author Notes

This project was built to:

* Compare architectures under controlled conditions
* Strengthen understanding of deep learning fundamentals
* Demonstrate clean experimentation and evaluation practices

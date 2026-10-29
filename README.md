# Dog-Cat Image Classification Using CNN

## Overview

This project uses a Convolutional Neural Network (CNN) to classify images into two categories: dogs and cats. The model contains three convolutional layers to learn image features.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Dataset

- **Dataset:** Dogs vs. Cats
- **Source:** [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Classes:** Dog and Cat

The dataset contains labeled images of dogs and cats for training and testing the model.

## Model Architecture

1. Input Image Layer
2. Convolutional Layer 1
3. Convolutional Layer 2
4. Convolutional Layer 3
5. Max Pooling Layers
6. Flatten Layer
7. Fully Connected Layer
8. Output Layer for Classification

## How It Works

1. Load the dog and cat image dataset.
2. Resize images and normalize pixel values.
3. Divide the images into training and validation sets.
4. Build a CNN with three convolutional layers.
5. Train the model using the training images.
6. Evaluate the model on validation images.
7. Predict whether an image contains a dog or a cat.

## Installation

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib

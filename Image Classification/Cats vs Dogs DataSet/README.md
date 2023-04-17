# Cats and Dogs Classifier using Convolutional Neural Networks

This is a simple Convolutional Neural Network (CNN) based project to classify between cats and dogs using Python and TensorFlow.

## Prerequisites

Make sure you have the following libraries installed before running the code:
- tensorflow
- numpy
- pickle
-matplotlib
- cv2
- random

## Dataset

The dataset used in this project is the "Cats and Dogs" dataset from Microsoft. It consists of 25,000 images of cats and dogs, each of size 50x50.

## Code Structure

The code is structured as follows:
- Importing the required libraries
- Pre-processing the data
- Building the CNN model
- Training the model
- Making a prediction

## Pre-processing the data

The 'create_training_data()' function loads the images from the dataset and resizes them to 50x50. It also adds the image data and the corresponding labels to a list called 'training_data'.

## Building the CNN model

The CNN model consists of two Convolutional layers, each with 64 nodes, followed by ReLU activation and Max Pooling layers. This is followed by a Dense layer with 64 nodes and ReLU activation, and finally a Dense layer with a single node and Sigmoid activation.

## Training the model

The model is trained using the 'fit()' function. The data is split into a training set and a validation set using a split of 0.3. The model is trained for 10 epochs with a batch size of 32.

## Making a prediction

To make a prediction, an image is passed to the 'prepare()' function which reads in the image, converts it to grayscale, and resizes it to 50x50. The image is then passed to the trained model for prediction. The result is a value between 0 and 1, where values closer to 0 indicate a prediction of "Dog" and values closer to 1 indicate a prediction of "Cat".

## Tech Stack

**Python**: The programming language used to write the code.

**TensorFlow**: An open-source machine learning library for Python, used to build and train the neural network model in the code.

**Keras**: A high-level API for TensorFlow that allows for easier and faster creation of neural networks, used for building the model architecture in the code.

**NumPy**: A Python library for scientific computing, used for numerical operations on arrays in the code.

**OpenCV**: An open-source computer vision library, used for reading in images and image processing operations in the code.

**Matplotlib**: A Python plotting library, used for visualizing the images in the code.

**Pickle**: A Python module used for object serialization and deserialization, used for saving and loading the training data in the code.

## 🛠 Skills
- Deep Learning
- Convolutional neural networks (CNN)
- Image Processing
- Data Preparation
- Visualization
- Python Programming

## Authors
- [Srikanth Elkoori Ghantala Karnam](https://www.github.com/S-EGK)

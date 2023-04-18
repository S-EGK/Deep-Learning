# Deep-Learning

## Cryptocurrency Price Prediction with LSTM

This project is for building a predictive model using deep learning with a sequential neural network. The model predicts whether to buy, sell or hold cryptocurrencies based on historical price and volume data for several cryptocurrencies (BTC-USD, LTC-USD, ETH-USD, BCH-USD). The script preprocesses the data by normalizing it and collecting sequences of prices and volumes. The data is split into buy and sell sequences, then the model is trained on these sequences. The architecture of the neural network consists of LSTM, Dense, BatchNormalization and Dropout layers. The model is trained with Tensorflow and Keras.

## Image Classification

### Cats and Dogs Classifier using Convolutional Neural Networks

This project demonstrates building a Convolutional Neural Network (CNN) to classify images of cats and dogs. The script loads a dataset of cat and dog images, preprocesses the data by resizing it to a fixed size, converts it to grayscale, and saves the preprocessed data to two pickle files. The CNN is built using the TensorFlow and Keras libraries, and trained on the preprocessed image data. The model is then used to predict the category of a single input image. Finally, the predicted category is printed to the console. The code also includes the use of TensorBoard, a visualization tool provided by TensorFlow, for monitoring the training process.

## Sign Language Recognition with CNN

This project builds and trains a Convolutional Neural Network (CNN) to classify sign language gestures from the Sign Language MNIST dataset. The dataset contains images of sign language gestures, represented as grayscale images of size 28x28 pixels. The code first loads and pre-processes the data, hot one encoding the labels and splitting the data into training and testing sets. The CNN model consists of three convolutional layers with pooling, followed by two fully connected layers with dropout. The model is compiled with the categorical cross-entropy loss function and the Adam optimizer. Finally, the model is trained on the training set and evaluated on the testing set, and the accuracy is plotted over the epochs. The trained model is saved for future use.

# Deep-Learning

## Cryptocurrency Price Prediction with LSTM

This code is for building a predictive model using deep learning with a sequential neural network. The model predicts whether to buy, sell or hold cryptocurrencies based on historical price and volume data for several cryptocurrencies (BTC-USD, LTC-USD, ETH-USD, BCH-USD). The script preprocesses the data by normalizing it and collecting sequences of prices and volumes. The data is split into buy and sell sequences, then the model is trained on these sequences. The architecture of the neural network consists of LSTM, Dense, BatchNormalization and Dropout layers. The model is trained with Tensorflow and Keras.

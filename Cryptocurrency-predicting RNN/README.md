# Cryptocurrency Price Prediction with LSTM

## Introduction

This script uses a Long Short-Term Memory (LSTM) neural network to predict future prices of cryptocurrencies. LSTM networks are capable of handling the sequential data, making it ideal for predicting stock and cryptocurrency prices.

## Dependencies
- pandas
- sklearn
- numpy
- tensorflow

## Code Overview
1. Import required libraries: pandas, os, numpy, random, time, and tensorflow
2. Define hyperparameters: 'SEQ_LEN', 'FUTURE_PERIOD_PREDICT', 'RATIO_TO_PREDICT', 'EPOCHS', 'BATCH_SIZE', 'NAME'
3. Define 'classify()' function to compare current and future prices and return a buy (1) or not buy (0) classification
4. Define 'preprocess_df()' function to preprocess the data. It drops the "future" column, normalizes all columns except for the target, scales the data between 0 and 1, and creates a list of sequences
5. Define 'main_df' and 'ratios'. Iterate over the ratios, read in the data, rename columns, and join them to create 'main_df'
6. Iterate over 'main_df' and create sequences for the LSTM. The sequences are divided into buys and sells
7. Shuffle the buys and sells, get the lower length, combine them, shuffle again, and split the sequences and targets into X and y
8. Train the LSTM model

## Usage
1. Install the required dependencies
2. Run the script

```sh
python crypto_price_prediction.py
```

## Tech Stack

Programming language: **Python**

Libraries: **Pandas**, **os**, **sklearn**, **collections**, **numpy**, **random**, **time**, **tensorflow**

Framework: **TensorFlow**

## 🛠 Skills
- Machine learning algorithms
- Python programming
- Data preprocessing
- Data visualization
- Mathematics and statistics
- LSTM

## Authors
- [Srikanth Elkoori Ghantala Karnam](https://www.github.com/S-EGK)

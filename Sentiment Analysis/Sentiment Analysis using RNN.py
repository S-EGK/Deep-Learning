#%% Libraries
import pandas as pd
import seaborn as sns
import re
'''import nltk'''
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn
patch_sklearn()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, GlobalMaxPooling1D, CuDNNLSTM
import matplotlib.pyplot as plt

#%% Loading Data Set
movie_reviews = pd.read_csv("IMDB Dataset/IMDB Dataset.csv")

# Dataset Exploration
print(f"Shape of the dataset: {movie_reviews.shape}")

print("First 5 elements of the dataset:")
print(movie_reviews.head(5))

# Checking for missing values
print(f'Does the dataset have any null elements: {movie_reviews.isnull().values.any()}')

# Observe distribution of positive / negative sentiments in dataset
sns.countplot(x='sentiment', data=movie_reviews).set(title='Review Distribution')

#%% Data Processing
print("Review before preprocessing:")
print(movie_reviews["review"][2])

# You can see that our text contains punctuations, brackets, HTML tags and numbers
# We will preprocess this text in the next section

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)

'''nltk.download('stopwords')'''

def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only in lowercase'''

    sentence = sen.lower()

    # Remove html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Calling preprocessing_text function on movie_reviews
X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))

# Sample cleaned up movie review
print("Review after preprocessing:")

print(X[2])
# As we shall use Word Embeddings, stemming/lemmatization is not performed as a preprocessing step here

y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# The train set will be used to train our deep learning models
# while test set will be used to evaluate how well our model performs

#%% Preparing embedding layer
# Embedding layer converts our textual data into numeric form. It is then used as the first layer for the deep learning models in Keras

# Embedding layer expects the words to be in numeric form
# Using Tokenizer function from keras.preprocessing.text library
# Method fit_on_text trains the tokenizer
# Method texts_to_sequences converts sentences to their numeric form

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)

# Adding 1 to store dimensions for words for which no pretrained word embeddings exist
vocab_length = len(word_tokenizer.word_index) + 1
print(f"Number of vocabs: '{vocab_length}'")

# Padding all reviews to fixed length 100
maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Load GloVe word embeddings and create an Embeddings Dictionary
embeddings_dictionary = dict()
glove_file = open('glove 6B/glove.6B.300d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# Create Embedding Matrix having 100 columns
# Containing 100-dimensional GloVe word embeddings for all words in our corpus
embedding_matrix = np.zeros((vocab_length, 300))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

print(f"Shape of the embedding_martix is {embedding_matrix.shape}")

#%% Simple Neural Network

# Neural Network Architecture
snn_model = Sequential()
embedding_layer = Embedding(vocab_length, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False)

snn_model.add(embedding_layer)

snn_model.add(Flatten())
snn_model.add(Dense(1, activation='sigmoid'))

# Model Compiling
snn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print("Summary of the simple neural network:")
print(snn_model.summary())

# Model Training
snn_model_history = snn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Predictions on the Test Set
snn_score = snn_model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print(f"SNN Model Test Score: {snn_score[0]}")
print(f"SNN Model Test Accuracy: {snn_score[1]}")

# Model Performance Charts
fig = plt.figure()

ax = fig.add_subplot(2,1,1)
ax.plot(snn_model_history.history['acc'])
ax.plot(snn_model_history.history['val_acc'])
plt.title('SNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')

ax1 = fig.add_subplot(2,1,2)
ax1.plot(snn_model_history.history['loss'])
ax1.plot(snn_model_history.history['val_loss'])
plt.title('SNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')

plt.show()

#%% Convolutional Neural Network

# Neural Network Architecture
cnn_model = Sequential()
embedding_layer = Embedding(vocab_length, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False)

cnn_model.add(embedding_layer)

cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(1, activation='sigmoid'))

# Model Compiling
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print("Summary of the Convolutional Neural Network:")
print(cnn_model.summary())

# Model Training
cnn_model_history = cnn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Predictions on the Test Set
cnn_score = cnn_model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print(f"CNN Model Test Score: {cnn_score[0]}")
print(f"CNN Model Test Accuracy: {cnn_score[1]}")

# Model Performance Charts
fig1 = plt.figure()

ax = fig1.add_subplot(2,1,1)
ax.plot(cnn_model_history.history['acc'])
ax.plot(cnn_model_history.history['val_acc'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')

ax1 = fig1.add_subplot(2,1,2)
ax1.plot(cnn_model_history.history['loss'])
ax1.plot(cnn_model_history.history['val_loss'])
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')

plt.show()

#%% Recurrent Neural Network (cudnnlstm)

# neural Network Architecture
lstm_model = Sequential()
embedding_layer = Embedding(vocab_length, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False)

lstm_model.add(embedding_layer)
lstm_model.add(CuDNNLSTM(128))

lstm_model.add(Dense(1, activation='sigmoid'))

# Model Compiling
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print("Summary of the Recurrent Neural Network:")
print(lstm_model.summary())

# Model training
lstm_model_history = lstm_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Predictions on the test set
lstm_score = lstm_model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print(f"RNN Model Test Score: {lstm_score[0]}")
print(f"RNN Model Test Accuracy: {lstm_score[1]}")

# Model Performance Charts
fig2 = plt.figure()

ax = fig2.add_subplot(2,1,1)
ax.plot(lstm_model_history.history['acc'])
ax.plot(lstm_model_history.history['val_acc'])
plt.title('RNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')

ax1 = fig2.add_subplot(2,1,2)
ax1.plot(lstm_model_history.history['loss'])
ax1.plot(lstm_model_history.history['val_loss'])
plt.title('RNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')

plt.show()

#%% Making Predictions on Live IMDb data
# Load sample IMDb reviews csv, having ~6 movie reviews, along with their IMDb rating
sample_reviews = pd.read_csv('IMDB Dataset/IMDb_Unseen_Reviews.csv')

print("First 6 elements of the unseen dataset:")
print(sample_reviews.head(6))

# Preprocess review text with earlier defined preprocess_text function
unseen_reviews = sample_reviews['Review Text']

unseen_processed = []
for review in unseen_reviews:
    review = preprocess_text(review)
    unseen_processed.append(review)

# Tokenising instance with earlier trained tokeniser
unseen_tokenized = word_tokenizer.texts_to_sequences(unseen_processed)

# Pooling instance to have maxlength of 100 tokens
unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=maxlen)

# Passing tokenized instance to the LSTM model for predictions
unseen_sentiments = lstm_model.predict(unseen_padded)
print("The Sentiments for the unseen reviews are:")
print(unseen_sentiments)

# Writinfg model output back into the file
sample_reviews['Predicted Sentiments'] = np.round(unseen_sentiments*10,1)

df_prediction_sentiments = pd.DataFrame(sample_reviews['Predicted Sentiments'], columns=['Predicted Sentiments'])
df_movie = pd.DataFrame(sample_reviews['Movie'], columns=['Movie'])
df_review_text = pd.DataFrame(sample_reviews['Review Text'], columns=['Review Text'])
df_imdb_rating = pd.DataFrame(sample_reviews['IMDb Rating'], columns=['IMDb Rating'])

dfx = pd.concat([df_movie, df_review_text, df_imdb_rating, df_prediction_sentiments], axis=1)

dfx.to_csv("./IMDB Dataset/IMDB_Unseen_Predictions.csv", sep=',', encoding='UTF-8')
print('The Unseen Predictions File Data:')
print(dfx.head(6))
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras.backend as backend
from sklearn.metrics import accuracy_score
import pyttsx3

# Data
train = pd.read_csv('archive/sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('archive/sign_mnist_test/sign_mnist_test.csv')

# Inspect the traininf data
print(f"First few data points:\n {train.head()}")

# Get the training labels
labels = train['label'].values

# View the unique labels, 24 in total
unique_val = np.array(labels)
print(f"The unique labels in the dataset are:\n{np.unique(unique_val)}")

# Plot the quantities in each class
fig = plt.figure(layout='tight')
ax = fig.add_subplot(111)
sns.countplot(x=labels, ax=ax)

# Drop training labels from our training data
train.drop('label', axis=1, inplace=True)

# Extract the image data from each row in our csv
images = train.values
images = np.array([np.reshape(i, (28,28)) for i in images])
images = np.array([i.flatten() for i in images])

# hot one encode the labels
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

# View the labels
print(f"Labels converted into binary format:\n{labels}")

# Inspect the image
index = 2
print(f"Label of image {index}:\n{labels[index]}")
fig1 = plt.figure(layout='tight')
ax = fig1.add_subplot(111)
ax.imshow(images[index].reshape(28,28))

# Use OpenCV to view 10 random images from the training data
# for i in range(0,10):
#     rand = np.random.randint(0, len(images))
#     input_im = images[rand]

#     sample = input_im.reshape(28,28).astype(np.uint8)
#     sample = cv2.resize(sample, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('sample image', sample)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()

# Split the data into x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

# Define DNN hyperparameters
batch_size = 128
num_classes = 24
epochs = 10

# Scale the images
x_train = x_train/255
x_test = x_test/255

# Reshape the data into the TF and Keras required size
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Show the new formatted image
fig2 = plt.figure(layout='tight')
ax = fig2.add_subplot(111)
ax.imshow(x_train[0].reshape(28,28))

# Memory fraction, used mostly when training multiple agents
MEMORY_FRACTION = 0.3
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

# Create the CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print(model.summary())

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

# Save the model
model.save("sign_minst_cnn.h5")

# View the training history graphically
fig3 = plt.figure(layout='tight')
ax = fig3.add_subplot(111)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train', 'test'], loc='best')

# Reshape the test data to check performance of the model
test_labels = test['label']
test.drop('label', axis=1, inplace=True)

test_images = test.values
test_images = np.array([np.reshape(i, (28,28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

test_images.shape

y_pred = model.predict(test_images)

# Get the accuracy score
print(f'Accuracy of the model o test data: {accuracy_score(test_labels, y_pred.round())}')

# Function to match label to letter
def getletter(result):
    classlabels = {0: 'A',
                   1: 'B',
                   2: 'C',
                   3: 'D',
                   4: 'E',
                   5: 'F',
                   6: 'G',
                   7: 'H',
                   8: 'I',
                   9: 'K',
                   10: 'L',
                   11: 'M',
                   12: 'N',
                   13: 'O',
                   14: 'P',
                   15: 'Q',
                   16: 'R',
                   17: 'S',
                   18: 'T',
                   19: 'U',
                   20: 'V',
                   21: 'W',
                   22: 'X',
                   23: 'Y'}
    try:
        res = int(result)
        return classlabels[res]
    except:
        return 'Error'

# Test on webcam
cap = cv2.VideoCapture(0)

# Collect letters from the user
letters = []

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    # define region of interest
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_CUBIC)

    cv2.imshow('roi scaled and grayed', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320,100), (620,400), (255,0,0), 5)

    roi = roi.reshape(1,28,28,1)

    result = str(np.argmax(model.predict(roi)))
        
    cv2.putText(copy, getletter(result), (300,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    cv2.imshow('frame', copy)
    
    letter = getletter(result)
    letters.append(letter)
    
    if cv2.waitKey(1) == 13: # 13 is enter key
        break

cap.release()
cv2.destroyAllWindows()

# Form words from the collected letters
words = []
word = ''
for letter in letters:
    if letter == ' ':
        words.append(word)
        word = ''
    else:
        word += letter
if word != '':
    words.append(word)

engine = pyttsx3.init()
engine.say(words)
engine.runAndWait()

plt.close('all')

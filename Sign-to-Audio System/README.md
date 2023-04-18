# Sign Language Recognition with CNN

This code implements a convolutional neural network (CNN) for recognizing hand signs in American Sign Language (ASL) using the TensorFlow and Keras libraries.

## Libraries

The following libraries are imported:
- 'numpy' for numerical computation
- 'pandas' for data manipulation and analysis
- 'matplotlib' and seaborn for data visualization
- 'sklearn' for preprocessing, model selection, and performance evaluation
- 'cv2' for image processing
- 'tensorflow' and 'keras' for building and training the CNN model
- 'pyttsx3' for text-to-speech conversion

## Data

The code uses the ASL sign language dataset, which consists of 27,455 grayscale images of 24 hand signs. The sign_mnist_train.csv file contains 24,274 images for training, and the 'sign_mnist_test.csv' file contains 7172 images for testing.

The 'pandas' library is used to read in the CSV files as data frames. The 'train' and 'test' data frames are created to hold the training and testing data, respectively. The 'label' column in the training data frame is dropped and assigned to a separate labels variable, which is then one-hot encoded using the 'LabelBinarizer' function from the 'sklearn' library.

## Data Visualization

The 'seaborn' and 'matplotlib' libraries are used to visualize the data. A bar graph is plotted to show the quantities of each class in the training data. An example image from the dataset is also displayed using 'matplotlib'.

## Data Preprocessing

The training and testing data are split into 'x_train', 'x_test', 'y_train', and 'y_test' using the 'train_test_split' function from 'sklearn'. The data is then scaled by dividing each pixel value by 255. The images are then reshaped into the required format for the CNN using the 'reshape' function.

## CNN Model

A CNN model is built using the 'Sequential' class from 'keras'. The model has 3 convolutional layers with a 'relu' activation function, and each is followed by a max-pooling layer. The output of the third max-pooling layer is flattened and connected to a fully connected layer with a 'relu' activation function, followed by a dropout layer to prevent overfitting. Finally, the output layer has 'softmax' activation with 24 units (one for each class).

## Model Training and Evaluation

The model is compiled using 'categorical_crossentropy' loss and the 'Adam' optimizer. It is then trained on the training data using the 'fit' method with a batch size of 128 and 10 epochs. The 'history' object returned by the 'fit' method is used to visualize the accuracy and loss of the training and testing sets over time.

The trained model is then saved as 'sign_minst_cnn.h5'.

The testing data is then preprocessed in the same way as the training data, and its performance is evaluated using the 'accuracy_score' function from 'sklearn'.

Finally, a plot is displayed to show the training and testing accuracy over time. Additionally, the 'pyttsx3' library is used to output the final test accuracy as speech.

## Tech Stack

**Python** programming language

**NumPy** for numerical operations

**Pandas** for data manipulation and analysis

**Matplotlib** for data visualization

**Seaborn** for statistical data visualization

**Scikit-learn** for machine learning algorithms and data preprocessing

**OpenCV** for computer vision tasks

**TensorFlow** and **Keras** for building and training deep learning models

**Pyttsx3** for text-to-speech conversion

## 🛠 Skills
- Deep Learning
- Computer Vision
- Convolutional neural networks (CNN)

## Authors
- [Srikanth Elkoori Ghantala Karnam](https://www.github.com/S-EGK)

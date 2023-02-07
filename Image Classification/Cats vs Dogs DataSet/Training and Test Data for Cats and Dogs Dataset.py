import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/egksr/OneDrive - University of Cincinnati/Deep Learning/kagglecatsanddogs_5340/PetImages"
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:     # do dogs and cats
    path = os.path.join(DATADIR, category)      # path to cats or dogs dir
    for img in os.listdir(path):        # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)        # convert to array
        plt.imshow(img_array, cmap="gray")      # graph it
        plt.show()      # display!
        break   # we just want one for now so break
    break       #...and one more!

IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:     # do dogs and cats
        path = os.path.join(DATADIR, category)      # path to cats or dogs dir
        class_num = CATEGORIES.index(category)      # get the classification  (0 or a 1). 0=dog 1=cat
        for img in os.listdir(path):        # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)    # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))      # resize to normalize data size
                training_data.append([new_array, class_num])        # add this to our training_data
            except Exception as e:      # in the interest in keeping the output clean...
                pass
            
create_training_data()

#%%

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
           
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
            
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

print(X[1])
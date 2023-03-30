# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn
patch_sklearn()
from sentence_transformers import SentenceTransformer, util
import time
import pickle
from sklearn.neighbors import NearestNeighbors

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Read the dataset
train_df = pd.read_csv(r'C:\Users\Srikanth\OneDrive - University of Cincinnati\Deep Learning\Multi-Modal ML\Data\train.csv')
print(f'First 5 sample of dataset:\n{train_df.head()}')
print(f'Number of examples: {len(train_df)}')
print(f"Number of unique examples: {len(train_df['label_group'].unique())}")

labelGroup_df = train_df.groupby('label_group')

# To view Few Examples uncomment below
# groupCount = 0
# for groupName,groupDf in labelGroup_df:
#     print(groupName)
#     imgCount=0
#     for index,row in groupDf.iterrows():
#         print(row['title'])
#         imagePath = 'C:/Users/Srikanth/OneDrive - University of Cincinnati/Deep Learning/Multi-Modal ML/Data/train_images/'+row['image']
#         pil_im = Image.open(imagePath, 'r')
#         plt.figure()
#         plt.imshow(pil_im)
#         plt.show()
#         imgCount= imgCount+1
#         if (imgCount==3):
#             break

#     groupCount = groupCount +1
#     if (groupCount==10):
#         break

# Generate train-test data
y = train_df.pop('label_group')
X = train_df

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4, random_state=0)

print(f"Shape of training data: {np.shape(X_train)}")

# Use bert model to get word embeddings
model = SentenceTransformer('stsb-distilbert-base')
model.max_seq_length = 128

# Use MobileNet to get image embeddings
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)
img_model = tf.keras.applications.MobileNet(input_shape= (IMG_SIZE, IMG_SIZE, 3), include_top = False, weights='imagenet')

# Function to generate text embeddings
def get_textEmbeddings(model, text):
    text_embedding = model.encode(text, convert_to_tensor=True)
    return text_embedding

# Function to generate image embeddings
def get_imageEmbeddings(model, imagePath):
    image = tf.keras.preprocessing.image.load_img(imagePath, target_size=size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    img_embeddings = model(input_arr)
    meanImgEmb1 = np.mean(img_embeddings, axis=0)
    meanImgEmb2 = np.mean(meanImgEmb1, axis=0)
    meanImgEmb = np.mean(meanImgEmb2, axis=0)
    return meanImgEmb

# Generate the embeddings
text_embeddings = {}
image_embeddings = {}
start_time= time.time()
for index, row in X_train.iterrows():
    txt_emb = get_textEmbeddings(model, str(row[3]))
    imagePath = 'C:/Users/Srikanth/OneDrive - University of Cincinnati/Deep Learning/Multi-Modal ML/Data/train_images/'+row[1]
    img_emb = get_imageEmbeddings(img_model, imagePath)
    text_embeddings[row[0]] = txt_emb
    image_embeddings[row[0]] = img_emb
end_time = time.time()
print(f"Time taken to generate Embeddings: {str(end_time-start_time)}")

#%% Save the embeddings
with open('./textEmb', 'wb') as handle:
    pickle.dump(text_embeddings, handle)
with open('./imgEmb', 'wb') as handle:
    pickle.dump(image_embeddings, handle)

keyList = []
cembList = []
imageList = []
titleList = []
for index, row in X_train.iterrows():
    txt_emb = text_embeddings[row[0]]
    img_emb = image_embeddings[row[0]]
    cmb_emb = np.concatenate((txt_emb, img_emb), axis=0)
    norm = np.linalg.norm(cmb_emb)
    cmb_emb_normal = cmb_emb/norm
    keyList.append(row[0])
    cembList.append(cmb_emb_normal)
    imageList.append(row[1])
    titleList.append(row[3])

#%% Get the Nearest Neighbors

from sklearn.neighbors import NearestNeighbors

kneigh = NearestNeighbors(n_neighbors=5,leaf_size=5000,algorithm='kd_tree')
kneigh.fit(cembList)

def getNeighbours(query_emb):
    posting_id_list=[]
    neigh_dist,neigh_ind = kneigh.kneighbors(X=query_emb.reshape(1,-1), n_neighbors=5, return_distance=True)
    for ind in neigh_ind:
        #print(str(ind))
        for ind1 in ind:
            posting_id_list.append(str(ind1))
    return posting_id_list

postingidList=[]
matchesList=[]
index =0
for val in keyList:
    query_emb = cembList[index]
    postingid_list = getNeighbours(query_emb)
    postingidList.append(val)
    matchesList.append(" ".join(postingid_list))
    index =index +1
    if index==100:
        break

index =0
for item in postingidList:
    print(titleList[index])
    print(keyList[index])
    imagePath = 'C:/Users/Srikanth/OneDrive - University of Cincinnati/Deep Learning/Multi-Modal ML/Data/train_images/'+ imageList[index]
    pil_im = Image.open(imagePath, 'r')
    plt.figure()
    plt.imshow(pil_im)
    plt.show()
    matching_indices = matchesList[index].split(' ')
    print('==================')
    for ind in matching_indices:
        print(titleList[int(ind)])
        print(keyList[int(ind)])
        imagePath = 'C:/Users/Srikanth/OneDrive - University of Cincinnati/Deep Learning/Multi-Modal ML/Data/train_images/'+ imageList[int(ind)]
        pil_im = Image.open(imagePath, 'r')
        plt.figure()
        plt.imshow(pil_im)
        plt.show()
    index= index +1
    if index == 10:
        break

#%%

testkeyList=[]
testcembList=[]
testimageList=[]
testtitleList=[]
for index, row in X_test.iterrows():
    #start_time=time.time()
    txt_emb = get_textEmbeddings(model,str(row[3]))
    imagePath = 'C:/Users/Srikanth/OneDrive - University of Cincinnati/Deep Learning/Multi-Modal ML/Data/test_images/'+row[1]
    img_emb = get_imageEmbeddings(img_model,imagePath)
    text_embeddings[row[0]] = txt_emb
    image_embeddings[row[0]] = img_emb
    cmb_emb = np.concatenate((txt_emb,img_emb),axis=0)
    norm = np.linalg.norm(cmb_emb)
    cmb_emb_normal = cmb_emb/norm
    testkeyList.append(row[0])
    testcembList.append(cmb_emb_normal)
    testimageList.append(row[1])
    testtitleList.append(row[3])

testpostingidList=[]
testmatchesList=[]
index =0
for val in testkeyList:
    query_emb = testcembList[index]
    postingid_list = getNeighbours(query_emb)
    testpostingidList.append(val)
    testmatchesList.append(" ".join(postingid_list))
    index =index +1
    if index==100:
        break

index =10
while index <20:
    print(testtitleList[index])
    print(testkeyList[index])
    imagePath = 'C:/Users/Srikanth/OneDrive - University of Cincinnati/Deep Learning/Multi-Modal ML/Data/test_images/'+ testimageList[index]
    pil_im = Image.open(imagePath, 'r')
    plt.figure()
    plt.imshow(pil_im)
    plt.show()
    matching_indices = testmatchesList[index].split(' ')
    print('==================')
    for ind in matching_indices:
        print(titleList[int(ind)])
        print(keyList[int(ind)])
        imagePath = 'C:/Users/Srikanth/OneDrive - University of Cincinnati/Deep Learning/Multi-Modal ML/Data/test_images/'+ imageList[int(ind)]
        pil_im = Image.open(imagePath, 'r')
        plt.figure()
        plt.imshow(pil_im)
        plt.show()
    index= index +1
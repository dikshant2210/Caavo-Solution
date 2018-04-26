
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os

dirs = glob.glob('input/dataset/train/*')
images = list()
labels = list()
for d in dirs:
    images += glob.glob(d + '/*')
    labels += [int(os.path.basename(d))] * len(glob.glob(d + '/*'))
    
from sklearn.utils import shuffle
df = pd.DataFrame({'images': images, 'labels': labels})
df = shuffle(df)
df.head()


# In[2]:


df.shape


# In[3]:


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base_model = VGG19(weights='imagenet', include_top=False)
    
model = Model(inputs=base_model.input, outputs=base_model.output)


# In[4]:


from tqdm import tqdm
import pickle as pkl

count = 0
for img_path, label in zip(df.images, df['labels']):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    features = model.predict(np.expand_dims(x, axis=0))
    features = np.squeeze(features)

    y = np.zeros(15)
    y[label] = 1
    
    name = os.path.basename(img_path).split()[0]
    with open('input/dataset/train_features/{}.pkl'.format(name), 'wb') as file:
        pkl.dump([features, y], file)
        
    count += 1
    if count % 3000 == 0:
        print(count)


# In[5]:


with open('input/dataset/train_features/{}.pkl'.format(name), 'rb') as file:
    ip = pkl.load(file)

ip[0].shape, ip[1].shape


# In[6]:


from tqdm import tqdm
import pickle as pkl

test_images = os.listdir('input/dataset/test/')

for img_path in tqdm(test_images):
    img_path = os.path.join('input/dataset/test/', img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    features = model.predict(np.expand_dims(x, axis=0))
    
    name = os.path.basename(img_path).split()[0]
    with open('input/dataset/test_features/{}.pkl'.format(name), 'wb') as file:
        pkl.dump(features, file)


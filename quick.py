
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os
import pickle as pkl


# In[2]:


dirs = glob.glob('input/dataset/train/*')
dirs


# In[3]:


images = list()
labels = list()
for d in dirs:
    images += glob.glob(d + '/*')
    labels += [int(os.path.basename(d))] * len(glob.glob(d + '/*'))


# In[4]:


images[:10], labels[:10]


# In[5]:


from sklearn.utils import shuffle
df = pd.DataFrame({'images': images, 'labels': labels})
df = shuffle(df)
df.head()


# In[6]:


df.shape


# In[7]:


train = df.iloc[:52258, :]
validation = df.iloc[52258:, :]
train.shape, validation.shape


# In[8]:


batch_size = 128
target_vector_size = df['labels'].unique().shape


# In[9]:


def train_generator():
    while True:
        for start in range(0, len(train), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train))
            train_batch = train.iloc[start:end, :]
            for img_path, label in zip(train_batch['images'], train_batch['labels']):
                name = os.path.basename(img_path).split()[0]
                with open('input/dataset/train_features/{}.pkl'.format(name), 'rb') as file:
                    data = pkl.load(file)
                    x_batch.append(data[0])
                    y_batch.append(data[1])
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch
            
def valid_generator():
    while True:
        for start in range(0, len(validation), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(validation))
            validation_batch = validation.iloc[start:end, :]
            for img_path, label in zip(validation_batch['images'], validation_batch['labels']):
                name = os.path.basename(img_path).split()[0]
                with open('input/dataset/train_features/{}.pkl'.format(name), 'rb') as file:
                    data = pkl.load(file)
                    x_batch.append(data[0])
                    y_batch.append(data[1])
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch


# In[10]:


for x, y in train_generator():
    print(x.shape, y.shape)
    break


# In[22]:


import tensorflow as tf

def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)


# In[27]:


from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Input, Dropout, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam

input_features = Input(shape=(7, 7 , 512))
features = GlobalAveragePooling2D()(input_features)
hidden = Dense(4096, activation='sigmoid')(features)
hidden = Dropout(0.2)(hidden)
hidden = Dense(2048, activation='sigmoid')(hidden)
hidden = Dropout(0.5)(hidden)
# hidden = Dense(256, activation='relu')(hidden)
# hidden = Dropout(0.6)(hidden)
predictions = Dense(15, activation='softmax')(hidden)
    
model = Model(inputs=input_features, outputs=predictions)

sgd = SGD(lr=0.01, momentum=0.8)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[f2_score])

callbacks = [EarlyStopping(monitor='val_f2_score',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=1e-4,
                               mode='min'),
             ModelCheckpoint(monitor='val_f2_score',
                             filepath='best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max')]


# In[28]:


model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(train)) / float(batch_size)),
                    epochs=50,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(validation)) / float(batch_size)))


# In[29]:


model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(train)) / float(batch_size)),
                    epochs=20,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(validation)) / float(batch_size)))


# In[30]:


test_images = os.listdir('input/dataset/test/')


# In[31]:


os.path.basename(test_images[5])


# In[32]:


model.load_weights('best_weights.hdf5')


# In[33]:


from tqdm import tqdm
image_name, category = list(), list()
for name in tqdm(test_images):
    name = name.split()[0]
    with open('input/dataset/test_features/{}.pkl'.format(name), 'rb') as file:
        features = pkl.load(file)
    
    image_name.append(name)
    label = np.argmax(model.predict(features))
    category.append(label)


# In[34]:


res = pd.DataFrame({'image_name': image_name, 'category': category}, columns=['image_name', 'category'])
# res = res.append(res.iloc[:5412, :])
print(res.shape)
res.to_csv('submission.csv', index=False)


# In[35]:


res.category.value_counts()


# In[36]:


model.load_weights('best_weights.hdf5')
model.evaluate_generator(generator=valid_generator(), 
                        steps=np.ceil(float(len(validation)) / float(batch_size)))


# In[37]:


preds = model.predict_generator(generator=valid_generator(), 
                               steps=np.ceil(float(len(validation)) / float(batch_size)))


# In[38]:


preds = np.argmax(preds, axis=1)
preds.shape


# In[39]:


validation['labels'].value_counts()


# In[40]:


pd.DataFrame({'preds': preds})['preds'].value_counts()


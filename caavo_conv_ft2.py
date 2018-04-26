
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os


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


batch_size = 32
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
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = preprocess_input(x)
                
                y = np.zeros(15)
                y[label] = 1
                
                x_batch.append(x)
                y_batch.append(y)
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
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = preprocess_input(x)
                
                y = np.zeros(15)
                y[label] = 1
                
                x_batch.append(x)
                y_batch.append(y)
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch


# In[10]:


import tensorflow as tf

def f1_score(y_true, y_pred):
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


# In[11]:


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers


model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=model.output_shape[1:]))
top_model.add(Dense(4096, activation='sigmoid'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2048, activation='sigmoid'))
top_model.add(Dropout(0.5))
top_model.add(Dense(15, activation='softmax'))

top_model.load_weights('best_weights_4096_0.2_2048_0.5_sigmoid.hdf5')

ft_model = Sequential()
for layer in model.layers:
    ft_model.add(layer)
for layer in top_model.layers:
    ft_model.add(layer)

for layer in ft_model.layers[:20]:
    layer.trainable = False
    
ft_model.summary()

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=1e-4,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath='best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min')]

ft_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy', f1_score])


# In[13]:


ft_model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(train)) / float(batch_size)),
                    epochs=30,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(validation)) / float(batch_size)))


# In[14]:


test_images = os.listdir('input/dataset/test/')


# In[15]:


os.path.basename(test_images[5])


# In[16]:


from tqdm import tqdm
image_name, category = list(), list()
for img_path in tqdm(test_images):
    img_path = os.path.join('input/dataset/test/', img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    image_name.append(os.path.basename(img_path))
    label = np.argmax(ft_model.predict(np.expand_dims(x, axis=0)))
    category.append(label)


# In[17]:


len(image_name), len(category)


# In[18]:


len(os.listdir('input/dataset/test/'))


# In[19]:


res = pd.DataFrame({'image_name': image_name, 'category': category}, columns=['image_name', 'category'])
# res = res.append(res.iloc[:5412, :])
print(res.shape)
res.to_csv('submission.csv', index=False)


# In[20]:


res.category.value_counts()


# In[21]:


preds = ft_model.predict_generator(generator=valid_generator(), 
                               steps=np.ceil(float(len(validation)) / float(batch_size)))
preds = np.argmax(preds, axis=1)
preds.shape


# In[22]:


pd.DataFrame({'preds': preds})['preds'].value_counts()


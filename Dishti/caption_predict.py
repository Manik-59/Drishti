#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import re 
import cv2
import json
import collections
import tensorflow as tf
import keras
from time import time 
import nltk
import pickle
# from nltk.corpus import stopwards
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Dropout,Dense,Flatten,Input,LSTM,Embedding
from keras.layers.merge import add


# In[12]:

# config = tensorflow.ConfigProto(
#     device_count={'GPU': 1},
#     intra_op_parallelism_threads=1,
#     allow_soft_placement=True
# )

# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6

# session = tensorflow.Session(config=config)

# keras.backend.set_session(session)



model_new = ResNet50(weights="imagenet",input_shape=(224,224,3))

model_new=Model(model_new.input,model_new.layers[-2].output)
# graph1=tf.get_default_graph()


model_new._make_predict_function()



# In[13]:


def preprocess_img(img):
    
    img=image.load_img(img,target_size=(224,224))
    
    img=image.img_to_array(img)
    
    img=np.expand_dims(img,axis=0)
    
    # normalisation
    
    img=preprocess_input(img)
    
    return img


# In[14]:
# session = keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)

def encode(img):
    # global graph1
    # with graph1.as_default():
    
    img=preprocess_img(img)

    
    feature_vector=model_new.predict(img)

    feature_vector=feature_vector.reshape((1,2048))

    return feature_vector


# In[15]:
model = load_model("./model_9.h5")
model._make_predict_function()
# graph1=tf.get_default_graph()

# In[16]:


with open("Dictionary/word_to_idx.pkl","rb") as f:
    word_to_idx = pickle.load(f)
    
with open("Dictionary/idx_to_word.pkl","rb") as f:
    idx_to_word = pickle.load(f)


def predict_caption(photo):
    # global graph1
    # with graph1.as_default():
    in_text = "startseq"
    max_len=35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')


        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word

        if word =='endseq':
            break
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


# In[18]:


def caption_it(image):
    
    encoded_img=encode(image)
    
    return predict_caption(encoded_img)




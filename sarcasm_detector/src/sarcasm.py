from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
# Helper libraries
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import re
pd.set_option('display.max_colwidth', -1)
#column names MAY NOT NEED
column_names = ['Link','Sentence','Sarcasm Score'] 
#load training data data
data_path = "../data/Sarcasm_Headlines_Dataset.json"
raw_data = pd.read_json(data_path, lines=True)

dataset = raw_data.copy()
#Setting up X & Y values for training
X = dataset.headline
Y = dataset.is_sarcastic
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size =0.2)

#Setup our word pools
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

#Define neural network
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,64,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer) 
    layer = Activation('relu')(layer)
    # layer = Dense(256,name='FC2')(layer)
    # layer = Dense(128,name='FC3')(layer)
    
    # layer = Dense(128,name='FC4')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

#Build the model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#Fit the model
model.fit(sequences_matrix,Y_train,batch_size=100,epochs=5,
          validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

#test model
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#Model accuracy
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# print("Saving model")



# #check our data
# print("Before data formatting")
# print(dataset.head())
# print(dataset.tail())
# #Remove unneccasary article links, we only need the headline and is_sarcastic for this test
# #If we were to classify model based on links model could form bias claiming specific
# #news sources less sarcastic than others, when this is not the case.

# dataset.pop("article_link")
# print("...printing after popping article link...")
# print(dataset.head())
# print(dataset.index)
# #create train labels and features variables
# train_labels = dataset['headline']
# train_sarcasm = dataset['is_sarcastic']
# result = text_to_word_sequence(dataset)
# train_dataset = dataset.sample(frac=0.8,random_state=0)

# train_stats = train_dataset.describe()
# print(train_dataset)
# print(train_stats)
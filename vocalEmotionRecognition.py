#!/usr/bin/python

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import librosa
import keras
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.backend import clear_session
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Activation, MaxPooling1D, Flatten
from keras.models import model_from_json

import json
import os

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

feeling_list=[]
training_audio = os.listdir('./Audio_Speech_Actors/training')


for item in training_audio:
  if item[6:-16] == '01' or 'neutral' in item:
    feeling_list.append('calm')
  elif item[6:-16] == '03' or 'happy' in item:
    feeling_list.append('happy') 
  elif item[6:-16] == '04' or 'sad' in item:
    feeling_list.append('sad')
  elif item[6:-16] == '05' or 'angry' in item:
    feeling_list.append('angry')


labels = pd.DataFrame(feeling_list)


df = pd.DataFrame(columns=['feature'])
idx = 0
for index,y in enumerate(training_audio):
  X, sample_rate = librosa.load('./Audio_Speech_Actors/training/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
  sample_rate = np.array(sample_rate)
  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
  feature = mfccs
  df.loc[idx] = [feature] # = mfccs
  idx = idx + 1

#Je recuperes le df contenant les spectre et les traduit en liste (permet d'avoir une liste contenant des sous liste de chaque audio)
df_tab = pd.DataFrame(df['feature'].values.tolist())

#etiquette le tableau avec le label d emotions
cross_df = pd.concat([df_tab,labels], axis=1)

cross_df = cross_df.rename(index=str, columns={"0": "label"})

cross_df = shuffle(cross_df)
#fill empty label with 0
cross_df = cross_df.fillna(0)

#Separe ma db en 2 train et test
newdf = np.random.rand(len(cross_df)) < 0.8
train_data = cross_df[newdf]
test_data = cross_df[~newdf]

trainfeatures = train_data.iloc[:, :-1] # spectre des donnees d'entrainement
trainlabel = train_data.iloc[:, -1:] # etiquette des donnees d'entrainement

testfeatures = test_data.iloc[:, :-1] # spectre des donnees d'evaluation
testlabel = test_data.iloc[:, -1:] # etiquette des donnees d'entrainement


x_train = np.array(trainfeatures)
y_train = np.array(trainlabel)

x_test = np.array(testfeatures)
y_test = np.array(testlabel)

print("y train")
print(y_train)
print("y test")
print(y_test)

lb = LabelEncoder()

#convertie la liste d'entier en une classe binaire
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
y_train = np_utils.to_categorical(lb.fit_transform(y_train))


x_traincnn = np.expand_dims(x_train, axis=2)
x_testcnn = np.expand_dims(x_test, axis=2)
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6) 


def train():

  model = Sequential()

  # 250 filtres convolution de taille 216
  model.add(Conv1D(256, 5, padding='same', input_shape=(216,1)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Conv1D(128, 5, padding='same'))
  model.add(Activation('relu'))
  # max d'operations de mise en commun des donnees
  model.add(MaxPooling1D(pool_size=(8)))

  # applatit l'input
  model.add(Flatten())

  #desite du modele 
  model.add(Dense(5))
  model.add(Activation('softmax'))
  #optimizer descente de gradient prenant en compte les variabilite
  # compile le modele en definissant la fonction de perte metrics as accuracy for classification pb
  model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

  #entraine le modele sur un nombre d iterations = epoch
  cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=50, validation_data=(x_testcnn, y_test))


  plt.plot(cnnhistory.history['loss'])
  plt.plot(cnnhistory.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  model_name = 'Emotion_Voice_Detection_Modelv2.h5'
  save_dir = os.path.join(os.getcwd(), './')
  # Save model and weights
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  model_path = os.path.join(save_dir, model_name)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Emotion_Voice_Detection_Modelv2.h5")
loaded_model._make_predict_function()

graph = tf.get_default_graph()


print("Loaded model from disk")



# evaluate loaded model on test data
# loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


def analyse_emotions(url):
  #decoupe et resemple l'audio entrant de la meme maniere que ceux du modele pour avoir un specte de meme taille
  X, sample_rate = librosa.load(url, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
  sample_rate = np.array(sample_rate)
  #recupere les coeficients du spectre 3 fois
  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
  featurelive = mfccs
  livedf2 = featurelive

  livedf2 = pd.DataFrame(data=livedf2)
  livedf2 = livedf2.stack().to_frame().T
  
  print(livedf2)

  #insert a new axis
  twodim = np.expand_dims(livedf2, axis=2)

  #predict save old graph configuration
  global graph
  with graph.as_default():
    livepreds = loaded_model.predict(twodim, 
                          batch_size=32,
                          verbose=1)


    livepreds1 = livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    livepredictions = (lb.inverse_transform((liveabc)))
    return livepredictions

import pandas as pd
import os.path
import _pickle as cPickle
import numpy as np
import keras.utils
import time
from keras.callbacks import TensorBoard, CSVLogger
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv1D,GlobalMaxPool1D,Dropout,Bidirectional
from keras.layers.embeddings import Embedding
from tensorflow.keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model
from nltk.corpus import stopwords
import operator
from para_list import *
from pre_processing import *

def predict(name, words, pos,dep,meta):
    model1 = load_model(name + '_weights_best.hdf5')
    # print(model1.summary())
    # print(words.shape)
    # print(pos.shape)
    # print(dep.shape)
    # print(meta.shape)
    pred = model1.predict([words, pos, dep, meta],batch_size = 40)
    label_list = ['false', 'true']
    label_pred = label_list[np.argmax(pred)]
    percent_pred = pred[0][1]
    return label_pred, percent_pred

def help():
    with open('help.txt','r') as f:
        print(f.read())
    return
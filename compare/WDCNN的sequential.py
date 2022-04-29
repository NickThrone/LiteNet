# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Nick Throne                               
# @software   : PyCharm
# @file       : WDCNN的sequential.py
# @Time       : 2022/3/9 16:44
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorboard
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU,Dense,Flatten,Conv1D,BatchNormalization,Activation,MaxPooling1D,Dropout,GlobalAveragePooling1D,concatenate,GlobalMaxPooling1D
from tensorflow.keras import layers, Sequential
from tensorflow.keras.regularizers import l2
import pandas as pd

import time
import datetime,os

tf.compat.v1.disable_eager_execution()



WDCNN=Sequential([
    layers.Conv1D(filters=16,kernel_size=(64,),strides=8,padding='same',name='conv1',batch_input_shape=(None,1024,1)),#[b,1024,1]=>[b,128,16]
    layers.MaxPooling1D(pool_size=2,strides=2,padding='valid',name='pooling1'),#[b,128,16]=>[b,64,16]
    layers.Conv1D(filters=32,kernel_size=(3,),strides=1,padding='same',name='conv2'),#[b,64,16]=>[b,64,32]
    layers.MaxPooling1D(pool_size=2,strides=2,padding='valid',name='pooling2'),#[b,64,32]=>[b,32,32]
    layers.Conv1D(filters=64,kernel_size=(3,),strides=1,padding='same',name='conv3'),#[b,32,32]=>[b,32,64]
    layers.MaxPooling1D(pool_size=2,strides=2,padding='valid',name='pooling3'),#[b,32,64]=>[b,16,64]
    layers.Conv1D(filters=64,kernel_size=(3,),strides=1,padding='same',name='conv4'),#[b,16,64]=>[b,16,64]
    layers.MaxPooling1D(pool_size=2,strides=2,padding='valid',name='pooling4'),#[b,16,64]=>[b,8,64]
    layers.Conv1D(filters=64,kernel_size=(3,),strides=1,padding='same',name='conv5'),#[b,8,64]=>[b,8,64]
    layers.MaxPooling1D(pool_size=2,strides=2,padding='valid',name='pooling5'),#[b,8,64]=>[b,4,64]
    layers.Conv1D(filters=64,kernel_size=(3,),strides=1,padding='same',name='conv6'),#[b,4,64]=>[b,4,64]
    layers.GlobalAveragePooling1D(name='pooling6'),#[b,4,64]=>[b,64]
    layers.Dense(units=109,activation='softmax',name='dense1')
])
WDCNN.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
WDCNN.summary()
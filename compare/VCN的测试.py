# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Nick Throne                               
# @software   : PyCharm
# @file       : VCN的测试.py
# @Time       : 2022/3/8 17:15
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorboard
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU,Dense,Flatten,Conv1D,BatchNormalization,Activation,MaxPooling1D,Dropout,GlobalAveragePooling1D,concatenate,GlobalMaxPooling1D

from tensorflow.keras.regularizers import l2
import pandas as pd

import time
import datetime,os

tf.compat.v1.disable_eager_execution()











#############################################################
input = tf.keras.Input(shape=(1000,1))


x = Conv1D(filters=16,kernel_size=5,strides=1,padding='valid',kernel_regularizer=l2(1e-4))(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x)

x2 = Conv1D(filters=32,kernel_size=3,strides=1,padding='valid',kernel_regularizer=l2(1e-4))(x)
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x2)

x3 = Conv1D(filters=64,kernel_size=3,strides=1,padding='valid',kernel_regularizer=l2(1e-4))(x2)
x3 = BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x3)

x3 = Flatten()(x3)
x3 = Dense(100,activation='relu',kernel_regularizer=l2(1e-4))(x3)

model = Model(inputs=input,outputs=x3)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss= 'categorical_crossentropy',
              metrics=['accuracy'])



a = np.zeros(shape=(10,1000))
a = a[:,:,np.newaxis]

b = model.predict(a)

print(b.shape)
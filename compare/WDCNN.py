# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Nick Throne                               
# @software   : PyCharm
# @file       : WDCNN.py
# @Time       : 2022/3/8 17:25
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



x_train = np.load(file='../dataprocess/sk_x_train_3.npy')
x_test = np.load(file='../dataprocess/sk_x_test_3.npy')
y_train = np.load(file='../dataprocess/sk_y_train_3.npy')
y_test = np.load(file='../dataprocess/sk_y_test_3.npy')

x_train,x_test = x_train[:,:,np.newaxis],x_test[:,:,np.newaxis]








#############################################################
input = tf.keras.Input(shape=(1024,1))


x = Conv1D(filters=16,kernel_size=64,strides=8,padding='same',kernel_regularizer=l2(1e-4))(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x)



x2 = Conv1D(filters=32,kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(1e-4))(x)
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x2 = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x2)

x3 = Conv1D(filters=64,kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(1e-4))(x2)
x3 = BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x3)

x4 = Conv1D(filters=64,kernel_size=3,strides=1,padding='same',kernel_regularizer=l2(1e-4))(x3)
x4 = BatchNormalization()(x4)
x4 = Activation('relu')(x4)
x4 = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x4)

x5 = Conv1D(filters=64,kernel_size=3,strides=1,padding='valid',kernel_regularizer=l2(1e-4))(x4)
x5 = BatchNormalization()(x5)
x5 = Activation('relu')(x5)
x5 = MaxPooling1D(pool_size=2,strides=2,padding='valid')(x5)

x6 = Flatten()(x5)
x6 = Dense(100,activation='relu')(x6)
x6 = Dense(15,activation='softmax')(x6)

model = Model(inputs=input,outputs=x6)

cosindecay_decay = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.01,decay_steps=3000
)



model.compile(optimizer='Adam',
              loss= 'categorical_crossentropy',
              metrics=['accuracy'])





start = time.clock()
history = model.fit(x_train,y_train,batch_size=50,epochs=100,validation_data=(x_test,y_test),shuffle=True,verbose=2)
end = time.clock()

print(end-start)




# a = np.zeros(shape=(10,1000))
# a = a[:,:,np.newaxis]
#
# b = model.predict(a)
#
# print(b.shape)

# model.summary()
import os
import sys
import random
import warnings
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import time

x_train = np.load(file='../dataprocess/x_train.npy')
x_test = np.load(file='../dataprocess/x_test.npy')
y_train = np.load(file='../dataprocess/y_train.npy')
y_test = np.load(file='../dataprocess/y_test.npy')

x_train,x_test = x_train[:,:,np.newaxis],x_test[:,:,np.newaxis]



def stack_conv_block(x, filters, kernel_size, strides=1):
   C1 = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
   C1 = BatchNormalization()(C1)
   C1 = Activation('relu')(C1)
   C2 = Conv1D(filters, kernel_size, strides=strides, padding='same')(C1)
   C2 = BatchNormalization()(C2)
   xx = Conv1D(filters, 1, strides=strides, padding='same')(x)
   C2 = C2 + xx
   y = Activation('relu')(C2)

   return y

def MSC_1DCNN():
    input_signal = Input(shape=(1024,1))
    x0 = Conv1D(64,kernel_size = 7, padding='same', strides=1)(input_signal)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)
    #print(x0.shape)
    x0 = MaxPooling1D(pool_size=2)(x0)

    x01 = stack_conv_block(x0, 64, 3, strides=1)
    x01 = stack_conv_block(x01, 128, 3, strides=1)
    x01 = stack_conv_block(x01, 256, 3, strides=1)
    x01 = GlobalAveragePooling1D()(x01)

    x02 = stack_conv_block(x0, 64, 5, strides=1)
    x02 = stack_conv_block(x02, 128, 5, strides=1)
    x02 = stack_conv_block(x02, 256, 5, strides=1)
    x02 = GlobalAveragePooling1D()(x02)

    x03 = stack_conv_block(x0, 64, 7, strides=1)
    x03 = stack_conv_block(x03, 128, 7, strides=1)
    x03 = stack_conv_block(x03, 256, 7, strides=1)
    x03 = GlobalAveragePooling1D()(x03)

    cat_feature_vec = concatenate([x01,x02,x03])
    x4 = Dense(768, activation='relu')(cat_feature_vec)

    output = Dense(10, activation='softmax')(x4)

    model = Model(inputs=input_signal, outputs=output)
    return model

model = MSC_1DCNN()

# a = np.zeros(shape=(10,512))
# a = a[:,:,np.newaxis]
#
# b = my_model.predict(a)
#
# print(b.shape)

cosindecay_decay = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.01,decay_steps=3000
)


boundaries = [50, 100, 150, 200, 250, 300]
values = [0.05,0.005,0.0005,0.00005,0.000005,0.0000005,0.00000005]
piece_wise_constant_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=boundaries,values=values,name=None,
)



model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=cosindecay_decay),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

start = time.clock()
history = model.fit(x_train,y_train,batch_size=50,epochs=100,validation_data=(x_test,y_test),shuffle=True,verbose=2)#
end = time.clock()

print(end-start)
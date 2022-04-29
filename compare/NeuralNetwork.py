# !/usr/bin python3
# encoding    : utf-8 -*-
# @author     : Nick Throne
# @software   : PyCharm
# @file       : p52.py
# @Time       : 2021/8/25 9:12
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

x = Flatten()(input)
x1 = Dense(64,activation='relu',kernel_regularizer=l2(1e-4))(x)

x2 = Dense(32,activation='relu',kernel_regularizer=l2(1e-4))(x1)

x3= Dense(64,activation='relu',kernel_regularizer=l2(1e-4))(x2)




b = tf.reduce_mean(x3,axis=1)


xx2 = Dense(15,activation='softmax',kernel_regularizer=l2(1e-4))(x3)


model = Model(inputs=input, outputs=xx2)





#####################################################################################################学习率
cosindecay_decay = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.01,decay_steps=4000
)

########################################################################################################


def multi_category_focal_loss2_fixed(y_true, y_pred):
    epsilon = 1.e-7    #就是1e-7
    gamma=2.           #就是2
    alpha = tf.constant(1, dtype=tf.float32)  #把0.5转化成float

    y_true = tf.cast(y_true, tf.float32)    #把tensor转化为float32
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon) #输入一个tensor,把tensor的每一个元素都压缩在min,max之间，防止log0等错误出现

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha) #tf.ones_like把其中的元素都变成1
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)

    ce = -tf.compat.v1.log(y_t)  #y_t的编程是对的，已经验证过，-y_true*log(y_pred)-(1-y_true)*log(1-y_pred)
    weight = tf.pow(tf.subtract(1., y_t), gamma) #tf.subtract()是相减，可以是一个数和一个矩阵相减，也可以是矩阵之间的相减，tf.pow(x,y)是 x^y的意思


    fl = ce#tf.multiply(weight,ce)

    loss = tf.reduce_mean(fl)+(tf.square(b))*0.5#+0.5*tf.square(a-c)#++tf.square(b)*0.5 #+ tf.subtract(1., (tf.pow(y_t,4)))*0.5
    return loss

###################################################################

model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=cosindecay_decay),
              loss= multi_category_focal_loss2_fixed,
              metrics=['accuracy'])


checkpoint_path = "SK/N+FA/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个检查点回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,period=100)

#
# start = time.clock()
history = model.fit(x_train,y_train,batch_size=50,epochs=100,validation_data=(x_test,y_test),shuffle=True,verbose=2,callbacks=[cp_callback]
                    )

# end = time.clock()
#
# print(end-start)

latest = tf.train.latest_checkpoint(checkpoint_dir)


########################################################





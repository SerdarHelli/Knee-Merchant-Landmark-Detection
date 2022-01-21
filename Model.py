# -*- coding: utf-8 -*-
"""
@author: serdarhelli
"""
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,concatenate,Conv2DTranspose,Dropout,AveragePooling2D,Add
import tensorflow as tf
from Adaptive_Wing_Loss import Adaptive_Wing_Loss

def Spatial_Configuration1(convc) :
  sconv = AveragePooling2D(pool_size=(16, 16))(convc)

  sconv = Conv2D(128,(11,11),padding="same",  kernel_initializer = 'he_normal')(sconv)
  sconv=BatchNormalization()(sconv)
  sconv=tf.keras.layers.LeakyReLU( alpha=0.1)(sconv)
  sconv=Dropout(0.5)(sconv)

  sconv = Conv2D(128,(11,11), padding="same" , kernel_initializer = 'he_normal')(sconv)
  sconv_c=BatchNormalization()(sconv)
  sconv=tf.keras.layers.LeakyReLU( alpha=0.1)(sconv)
  sconv=Dropout(0.5)(sconv)

  sconv = Conv2D(128,(11,11),padding="same", kernel_initializer = 'he_normal')(sconv)
  sconv=BatchNormalization()(sconv)
  sconv=tf.keras.layers.LeakyReLU(alpha=0.1)(sconv)
  sconv=Dropout(0.5)(sconv)

  sconv = Conv2D(6,(11,11), activation = "tanh", padding="same", kernel_initializer =tf.keras.initializers.RandomNormal(stddev=0.0001),kernel_regularizer=tf.keras.regularizers.l2(0.0005))(sconv)
  sconv1 = tf.keras.layers.UpSampling2D(size=(16, 16),interpolation='bilinear')(sconv)
  return sconv1


def UNet(inputs,x_dim,y_dim):
  
  u = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  u=BatchNormalization()(u)
  u = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.1)(u)
  u1 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  

  u = AveragePooling2D(pool_size=(2, 2))(u1)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.2)(u)
  u2 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)

  u = AveragePooling2D(pool_size=(2, 2))(u2)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.3)(u)
  u3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  

  u = AveragePooling2D(pool_size=(2, 2))(u3)
  u = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.3)(u)
  u4 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  

  u = AveragePooling2D(pool_size=(2, 2))(u4)
  u = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u) 
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u) 
  u=BatchNormalization()(u)
  u = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u) 
  u=BatchNormalization()(u)
  u = Conv2D(512,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=BatchNormalization()(u)


  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(256,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(256,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=Add()([u,u4])

  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=Add()([u,u3])

  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=Add()([u,u2])

  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=Add()([u,u1])

  u = Conv2D(64,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(32,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)

  return u

def getModel(input_shape,points_number):
    AWL=Adaptive_Wing_Loss()
    inputs=tf.keras.layers.Input(shape=(input_shape))
    
    unet_output=UNet(inputs,input_shape[0],input_shape[1])
    local_app = Conv2D(points_number,(1,1), activation = "linear", padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.0001),kernel_regularizer=tf.keras.regularizers.l2(0.0005))(unet_output)
    
    config_output=Spatial_Configuration1(local_app)
    outputs = tf.keras.layers.Multiply()([local_app,config_output])
    
    model = tf.keras.Model(inputs = inputs, outputs = [local_app,outputs])
    
    return model

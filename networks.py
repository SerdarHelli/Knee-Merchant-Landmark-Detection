from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,concatenate,Conv2DTranspose,Dropout,AveragePooling2D,Add
import tensorflow as tf


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


def UNet(inputs):
  
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

def build_model(shape,num_classes):
  inputs=tf.keras.layers.Input(shape=shape)

  unet_output=UNet(inputs)
  local_app = Conv2D(num_classes,(1,1), activation = "linear", padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.0001),kernel_regularizer=tf.keras.regularizers.l2(0.0005))(unet_output)

  config_output=Spatial_Configuration1(local_app)
  outputs = tf.keras.layers.Multiply()([local_app,config_output])


  #28-12-2021 Learning Rate  1e-5  to 1e-6 

  model = tf.keras.Model(inputs = inputs, outputs = [local_app,outputs])
  return model
# -*- coding: utf-8 -*-
"""
@author: serdarhelli
"""

import tensorflow as tf 
from tensorflow.keras import backend as K

import numpy as np


#Adaptive Wing Loss
class Adaptive_Wing_Loss():
    def __init__(self, alpha=float(2.1), omega=float(5), epsilon=float(1),theta=float(0.5)):   
        self.alpha=alpha
        self.omega=omega
        self.epsilon=epsilon
        self.theta=theta
    def Loss(self,y_true,y_pred):
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y_true)))*(self.alpha-y_true)*((self.theta/self.epsilon)**(self.alpha-y_true-1))/self.epsilon
        C = self.theta*A - self.omega*tf.math.log(1+(self.theta/self.epsilon)**(self.alpha-y_true))
        loss=tf.where(tf.math.greater_equal(tf.math.abs(y_true-y_pred), self.theta),A*tf.math.abs(y_true-y_pred) - C,self.omega*tf.math.log(1+tf.math.abs((y_true-y_pred)/self.epsilon)**(self.alpha-y_true)))
        return tf.reduce_mean(loss)



class Segmentation_Loss(tf.keras.losses.Loss):
    def __init__(self,smooth=100,  **kwargs):
        super().__init__(**kwargs)
        self.smooth=100

    def dice_coef(self,y_true, y_pred, smooth):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
        return dice

    def dice_coef_multilabel(self,y_true, y_pred, M, smooth):
        dice = 0
        y_true=(y_true>0.25)*1
        y_pred=(y_pred>0.25)*1
        for index in range(M):
            dice += self.dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], smooth)
        return np.float32(dice)

    
    def call(self,y_true,y_pred):
        M=tf.shape(y_true)[-1]
        dice_coef=tf.numpy_function(self.dice_coef_multilabel,[y_true,y_pred,int(M),self.smooth],tf.float32)
        loss=(1*float(M))-dice_coef

        return tf.reduce_mean(loss)

 
class PointMSE(tf.keras.losses.Loss):
  def __init__(self, alpha=float(2.1), omega=float(5), epsilon=float(1),theta=float(0.5), **kwargs):
      super().__init__(**kwargs)
      self.MSE= tf.keras.losses.MeanSquaredError()

    
  def points_max_value(self,predict):
      batch_size=predict.shape[0]
      points=np.zeros([int(batch_size),6,2])
      for j in range(int(batch_size)):
        for i in range(6):
            pre=predict[j,:,:,i]
            points_max=np.where(pre == pre.max())
            points[j,i,0],points[j,i,1]=points_max[0][0],points_max[1][0]
      return np.float32(np.fliplr(points))

  def call(self,y_true,y_pred):
      points_pred=tf.numpy_function(self.points_max_value,[y_pred],tf.float32)
      points_true=tf.numpy_function(self.points_max_value,[y_true],tf.float32)


      return self.MSE(points_pred, points_true)*(0.01)
 
 
 
 

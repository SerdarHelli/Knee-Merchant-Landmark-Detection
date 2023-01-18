
import tensorflow as tf
from networks import *
from loss import *

class ModelKnee(tf.keras.Model):
    def __init__(
        self
    ):
        super(ModelKnee, self).__init__()
        self.shape=(256,256,1)
        self.num_classes=6
        self.model = build_model(self.shape,self.num_classes)
        self.loss1 = tf.keras.metrics.Mean(name="loss1*1e3")
        self.loss2 = tf.keras.metrics.Mean(name="loss2*1e3")
        self.point_loss = tf.keras.metrics.Mean(name="point_loss")
        self.dice_segmentloss=tf.keras.metrics.Mean(name="Dice_loss")
        self.total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.MSE = tf.keras.metrics.Mean(name="MSE*1e4")

    def compile(self):
        super(ModelKnee, self).compile()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = Adaptive_Wing_Loss()
        self.metric_fn=tf.keras.losses.MeanSquaredError()
        self.loss_points=PointMSE()
        self.dice_segmentloss_fn=Segmentation_Loss()

    @property
    def metrics(self):
        return [
            self.total_loss,
            self.loss1,
            self.loss2,
            self.dice_segmentloss,
            self.point_loss,
            self.MSE
        ]





    def train_step(self, batch):

        X,Y=batch
        with tf.GradientTape() as tape:
            predict_1,predict_2 = self.model(X, training=True)
            loss1 =self.loss_fn(Y,predict_1) *1000
            loss2 =self.loss_fn(Y,predict_2) *1000
            dice_segmentloss=self.dice_segmentloss_fn(Y,predict_2)
            point_loss=self.loss_points(Y,predict_2)
            total_loss=((loss1+loss2)/2)+point_loss+dice_segmentloss
            MSE=self.metric_fn(Y,predict_2)*1000
            
        gen_gradient = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gen_gradient, self.model.trainable_variables)
        )
        self.loss1.update_state(loss1)
        self.loss2.update_state(loss2)
        self.dice_segmentloss.update_state(dice_segmentloss)

        self.total_loss.update_state(total_loss)
        self.MSE.update_state(MSE)
        self.point_loss.update_state(point_loss)
        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self,batch):
        X,Y=batch
        predict_1,predict_2 = self.model(X, training=True)
        loss1 =self.loss_fn(Y,predict_1) *1000
        loss2 =self.loss_fn(Y,predict_2) *1000   
        dice_segmentloss=self.dice_segmentloss_fn(Y,predict_2)
        point_loss=self.loss_points(Y,predict_2)
        total_loss=((loss1+loss2)/2)+point_loss+dice_segmentloss
        MSE=self.metric_fn(Y,predict_2)*1000
        self.loss1.update_state(loss1)
        self.loss2.update_state(loss2)
        self.total_loss.update_state(total_loss)
        self.MSE.update_state(MSE)
        self.point_loss.update_state(point_loss)
        self.dice_segmentloss.update_state(dice_segmentloss)

        results = {m.name: m.result() for m in self.metrics}
        return results
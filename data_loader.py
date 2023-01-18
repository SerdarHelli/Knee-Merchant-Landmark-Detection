import albumentations as A
import cv2
import tensorflow as tf

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, x,y,
                 batch_size,augment_size
                 ):
        
        self.x=x
        self.y=y
        self.augment_size=augment_size
        self.n = len(x)+(len(x)*augment_size)
        self.max_index=(len(x)//batch_size)-1
        self.batch_size=batch_size
        
   
    
    def __apply_augmentation(self, x,y):
        aug = A.Compose([
            A.OneOf([A.RandomCrop(width=256, height=256),
                        A.PadIfNeeded(min_height=256, min_width=256, p=0.5)],p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25,p=0.25),
            A.Compose([A.RandomScale(scale_limit=(-0.15, 0.15), p=1, interpolation=1),
                                    A.PadIfNeeded(256, 256, border_mode=cv2.BORDER_CONSTANT), 
                                    A.Resize(256, 256, cv2.INTER_NEAREST), ],p=0.25),
            A.ShiftScaleRotate (shift_limit=0.325, scale_limit=0.15, rotate_limit=15,border_mode=cv2.BORDER_CONSTANT, p=0.25),
            A.Rotate(15,p=0.25),
            A.Blur(blur_limit=1, p=0.25),
            A.Downscale(scale_min=0.15, scale_max=0.25,  always_apply=False, p=0.25),
            A.GaussNoise(var_limit=(0.05, 0.1), mean=0, per_channel=True, always_apply=False, p=0.25),
            A.OneOf([A.ElasticTransform(alpha=270, sigma=270 * 0.05, alpha_affine=270 * 0.03, p=.5),
                      A.GridDistortion(p=.5),
                      ], p=.25),
        ])
        x_aug2=np.copy(x)
        y_aug2=np.copy(y)
        for i in range(len(x)):
            augmented=aug(image=x[i,:,:,:],mask=y[i,:,:,:])
            x_aug2[i,:,:,:]= augmented['image']
            y_aug2[i,:,:,:]= augmented['mask']

          
        return x_aug2,y_aug2
      
    def __getitem__(self, index):
        if index>self.max_index:
            index=index%self.max_index
            X = self.x[index * self.batch_size : (index + 1) * self.batch_size,:,:,:]
            Y = self.y[index * self.batch_size : (index + 1) * self.batch_size,:,:,:]
            X,Y=self.__apply_augmentation(X,Y)
            return X,Y

        X = self.x[index * self.batch_size : (index + 1) * self.batch_size,:,:,:]
        Y = self.y[index * self.batch_size : (index + 1) * self.batch_size,:,:,:]

        return X,Y
    
    def __len__(self):
        return self.n // self.batch_size
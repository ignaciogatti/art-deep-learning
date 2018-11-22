import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from abc import ABC, abstractmethod


class Abstract_Model(ABC):
    
    @abstractmethod
    def get_model(self, img_Height, img_Width, n_classes, use_imagenet):
        pass
    
    
class Simple_Model(Abstract_Model):
    
    def __init__(self):
        self.model = None

    def get_model(self, img_Height, img_Width, n_classes, use_imagenet = False):

        self.model = Sequential()

        #block 1
        self.model.add(Conv2D(32, (3, 3), input_shape=(img_Height, img_Width, 3), name='CONV2_BLOCK1'))
        self.model.add(BatchNormalization(axis = 3, name = 'BATCHNORM_BLOCK1'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='MAXPOOL_BLOCK1'))

        #block 2
        self.model.add(Conv2D(32, (3, 3), name='CONV2_BLOCK2'))
        self.model.add(BatchNormalization(axis = 3, name = 'BATCHNORM_BLOCK2'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='MAXPOOL_BLOCK2'))

        #block 3
        self.model.add(Conv2D(64, (3, 3), name='CONV2_BLOCK3'))
        self.model.add(BatchNormalization(axis = 3, name = 'BATCHNORM_BLOCK3'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='MAXPOOL_BLOCK3'))

        #fully connected
        self.model.add(Flatten(name='flatten'))  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(64, activation='relu', name='fc1'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='sigmoid', name='predictions'))

        return self.model
    

class Inception_model(Abstract_Model):
    
    def __init__(self):
        self.model = None
        
    def get_model(self, img_Height, img_Width, n_classes, use_imagenet = False):
        # load pre-trained model graph, don't add final layer
        base_model = keras.applications.InceptionV3(include_top=False, input_shape=(img_Height, img_Width, 3),
                                              weights='imagenet' if use_imagenet else None)
        # add global pooling just like in InceptionV3
        new_output = keras.layers.GlobalAveragePooling2D()(base_model.output)
        # add new dense layer for our labels
        new_output = keras.layers.Dense(n_classes, activation='sigmoid')(new_output)
        self.model = keras.engine.training.Model(base_model.inputs, new_output)
        return self.model
    

class ResNet_model(Abstract_Model):
    
    def __init__(self):
        self.model = None
        
    def get_model(self, img_Height, img_Width, n_classes, use_imagenet = False):
        
        base_model = keras.applications.ResNet50(include_top=False, input_shape=(img_Height, img_Width, 3), 
                                            weights='imagenet' if use_imagenet else None)
    
        x = base_model.output
        x = Flatten(name='flatten')(x)
        
        #use sigmoid for binary classification - softmax has trouble
        prediction = Dense(n_classes, activation='sigmoid')(x)

        self.model = keras.engine.training.Model(base_model.inputs, prediction)

        return model

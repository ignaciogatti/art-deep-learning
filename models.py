import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Reshape, LeakyReLU, Dropout, UpSampling2D, Conv2DTranspose
from abc import ABC, abstractmethod


class Abstract_Model(ABC):
    
    @abstractmethod
    def get_model(self, input_shape, n_classes, use_imagenet):
        pass
    
    
class Simple_Model(Abstract_Model):
    
    def __init__(self):
        self.model = None

    def get_model(self, input_shape, n_classes, use_imagenet = False):

        self.model = Sequential()

        #block 1
        self.model.add(Conv2D(32, (3, 3), input_shape= input_shape, name='CONV2_BLOCK1'))
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
        
    def get_model(self, input_shape, n_classes, use_imagenet = False):
        # load pre-trained model graph, don't add final layer
        base_model = keras.applications.InceptionV3(include_top=False, input_shape=input_shape,
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
        
    def get_model(self, input_shape, n_classes, use_imagenet = False):
        
        base_model = keras.applications.ResNet50(include_top=False, input_shape=input_shape, 
                                            weights='imagenet' if use_imagenet else None)
    
        x = base_model.output
        x = Flatten(name='flatten')(x)
        
        #use sigmoid for binary classification - softmax has trouble
        prediction = Dense(n_classes, activation='sigmoid')(x)

        self.model = keras.engine.training.Model(base_model.inputs, prediction)

        return self.model
    
    
class Generator_model(Abstract_Model):
    
    def __init__(self):
        self.model = None
    
    
    def get_model(self, input_shape, n_classes = 1, use_imagenet = False):
            net = Sequential()
            dropout_prob = 0.4

            net.add(Dense(8*8*512, input_dim=input_shape))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())
            net.add(Reshape((8, 8, 512)))
            net.add(Dropout(dropout_prob))

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(512, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(256, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(128, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(64, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2D(3, 5, padding='same'))
            net.add(Activation('sigmoid'))
            
            self.model = net

            return self.model


        
class Generator_model_complex(Generator_model):
    
    def __init__(self, filters, code_size):
        self.model = None
        self.filters = filters
        self.code_size = code_size
    
    
    def get_model(self, input_shape = None, n_classes = 1, use_imagenet = False):
        generator = Sequential()

        #input shape: 1x1xNOISE_LEN
        #output shape: 4x4xfilters*32
        generator.add(Conv2DTranspose(self.filters * 32, kernel_size=(4, 4),input_shape=(1, 1, self.code_size)))
        generator.add(BatchNormalization())
        generator.add(Activation(activation='relu'))

        #output shape: 8x8xfilters*16
        generator.add(Conv2DTranspose(self.filters * 16, kernel_size=(4,4), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(Activation(activation='relu'))

        #output shape: 16x16xfilters*8
        generator.add(Conv2DTranspose(self.filters * 8, kernel_size=(4,4), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(Activation(activation='relu'))

        #output shape: 32x32xfilters*4
        generator.add(Conv2DTranspose(self.filters * 4, kernel_size=(4,4), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(Activation(activation='relu'))

        #output shape: 64x64xfilters*2
        generator.add(Conv2DTranspose(self.filters * 2, kernel_size=(4,4), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(Activation(activation='relu'))

        #output shape: 128x128xfilters
        generator.add(Conv2DTranspose(self.filters, kernel_size=(4,4), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(Activation(activation='relu'))

        #output shape: 256x256x3
        generator.add(Conv2DTranspose(3, kernel_size=(4,4), strides=(2,2), padding='same'))
        generator.add(BatchNormalization())
        generator.add(Activation(activation='sigmoid'))
        
        self.model = generator

        return self.model


class Discriminator_model(Abstract_Model):
    
    def __init__(self,filters, code_size):
        self.model = None
        self.filters = filters
        self.code_size = code_size
        
        
    def get_model(self, input_shape, n_classes = 1, use_imagenet = False):
        disc = Sequential()

        #input shape: 256x256x3
        #output shape: 128x128xfilters
        disc.add(Conv2D(self.filters, kernel_size=(4,4), strides=(2,2), padding='same', input_shape=input_shape))
        disc.add(BatchNormalization(momentum=0.8))
        disc.add(LeakyReLU())

        #output shape: 64x64xfilters*2
        disc.add(Conv2D(self.filters*2, kernel_size=(4,4), strides=(2,2), padding='same'))
        disc.add(BatchNormalization(momentum=0.8))
        disc.add(LeakyReLU())

        #output shape: 32x32xfilters*4
        disc.add(Conv2D(self.filters*4, kernel_size=(4,4), strides=(2,2), padding='same'))
        disc.add(BatchNormalization(momentum=0.8))
        disc.add(LeakyReLU())

        #output shape: 16x16xfilters*8
        disc.add(Conv2D(self.filters*8, kernel_size=(4,4), strides=(2,2), padding='same'))
        disc.add(BatchNormalization(momentum=0.8))
        disc.add(LeakyReLU())

        #output shape: 8x8xfilters*16
        disc.add(Conv2D(self.filters*16, kernel_size=(4,4), strides=(2,2), padding='same'))
        disc.add(BatchNormalization(momentum=0.8))
        disc.add(LeakyReLU())

        #output shape: 4x4xfilters*32
        disc.add(Conv2D(self.filters*32, kernel_size=(4,4), strides=(2,2), padding='same'))
        disc.add(BatchNormalization(momentum=0.8))
        disc.add(LeakyReLU())

        #output shape: 1x1xcode_size
        disc.add(Conv2D(self.code_size, kernel_size=(4,4)))
        disc.add(BatchNormalization(momentum=0.8))
        disc.add(LeakyReLU())
        
        disc.add(Flatten())
        disc.add(Dense(n_classes, activation='sigmoid', name='predictions'))
        
        self.model = disc

        return self.model

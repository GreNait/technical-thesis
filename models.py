# This modul contains and returns models for training.
# The idea is, to make a better automised training later on.
# This modul needs to be modular but also contains past used models.

#import basic config
import default
IMG_WIDTH, IMG_HEIGHT, DIMENSION, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_SHAPE, _, _ = default.getAllParameters()

import tensorflow as tf
print(tf.__version__) #Should be 2.1.x or higher

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

class FirstModel():
    def returnModel(self):
        model = Sequential([
            Conv2D(16, (3,3), input_shape = X.shape[1:], activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(256, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        return model

class SecondModel():
    def returnModel(self):
        model = Sequential([
        Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Dense(150, activation='relu'),
        Dense(150, activation='relu'),
        Dense(3)
        ])
        return model

class ModelComparison():
    def __init__(self,input_shape = None, activation = None, output_shape = None):
        if input_shape == None:
            self.input_shape = (IMG_HEIGHT, IMG_WIDTH, DIMENSION)
        else:
            self.input_shape = input_shape #(Height, Width, dimension) -> (113,150,3)

        if activation == None:
            self.activation = DEFAULT_ACTIVATION
        else:
            self.activation = activation

        if output_shape == None:
            self.output_shape = DEFAULT_OUTPUT_SHAPE
        else:
            self.output_shape = output_shape #How many figures?

    def returnSmall(self):
        model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(150, activation= self.activation),
            Dense(150, activation= self.activation),
            Dense(150, activation= self.activation),
            Dense(self.output_shape)
        ])

        return (model,"small")

    def returnCnnSmall(self):
        model = Sequential([
            Conv2D(8, (3,3), input_shape=self.input_shape, activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(16, (3,3), activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, (3,3), activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(150, activation= self.activation),
            Dense(150, activation= self.activation),
            Dense(150, activation= self.activation),
            Dense(self.output_shape)
        ])

        return (model,"cnn_small")

    def returnMedium(self):
        model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(250, activation= self.activation),
            Dense(250, activation= self.activation),
            Dense(250, activation= self.activation),
            Dense(self.output_shape)
        ])

        return (model,"medium")

    def returnCnnMedium(self):
        model = Sequential([
            Conv2D(8, (3,3), input_shape=self.input_shape, activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(16, (3,3), activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, (3,3), activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(250, activation= self.activation),
            Dense(250, activation= self.activation),
            Dense(250, activation= self.activation),
            Dense(self.output_shape)
        ])

        return (model,"cnn_medium")

    def returnBig(self):
        model = Sequential([
            Flatten(input_shape=self.input_shape),
            Dense(400, activation= self.activation),
            Dense(400, activation= self.activation),
            Dense(400, activation= self.activation),
            Dense(self.output_shape)
        ])

        return (model,"big")

    def returnCnnBig(self):
        model = Sequential([
            Conv2D(8, (3,3), input_shape=self.input_shape, activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(16, (3,3), activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, (3,3), activation=self.activation),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(400, activation= self.activation),
            Dense(400, activation= self.activation),
            Dense(400, activation= self.activation),
            Dense(self.output_shape)
        ])

        return (model,"cnn_big")

    def returnAllModels(self):
        models = [
            self.returnSmall(), 
            self.returnMedium(), 
            self.returnBig(), 
            self.returnCnnSmall(), 
            self.returnCnnMedium(), 
            self.returnCnnBig()
            ]

        return models


# %%
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

IMG_HEIGHT, IMG_WIDTH = 150,113

path_to_image = "/home/grenait/Desktop/technical_thesis/technical-thesis/smartPhone_batman.jpeg"
path_to_other_image = "/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/deadpool/2020-01-31 15:47:02_deadpool.png"
path_to_model = "/home/grenait/Desktop/technical_thesis/technical-thesis/trained_models/second_model.h5"

model = tf.keras.models.load_model(str(path_to_model))
# %%
test_gen = ImageDataGenerator(rescale=1./255)
test_gen = test_gen.flow_from_directory( batch_size=5,
                                        directory="/home/grenait/Desktop/technical_thesis/technical-thesis/testin_prediction_set/",
                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                        class_mode=None,
                                        shuffle=False)

propability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
prediction = propability_model.predict(test_gen)
print(prediction)
# %%
for i in prediction:
    print(np.argmax(i))


# %%

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_to_figures = "/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/overfitting"

# The names of the figures must be the same as the folder containing them
# we use these names to get the folder and load the data later into the 
# dataset.
figures = ["batman", "deadpool"]

# %%
img_array = []

for figure in figures:
    path = os.path.join(path_to_figures, figure)
    
    for img in os.listdir(path):
        img_array.append(cv2.imread(os.path.join(path,img)))
        break
    break

# %%
def left_right(images, move=[-200,200]):
    samples = expand_dims(images, 0)
    datagen = ImageDataGenerator(width_shift_range=move)
    it = datagen.flow(samples, batch_size = 1)

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.show()

left_right(img_array[0])
left_right(img_array[0],[200,-200])

# %%
def up_down(images, range):
    samples = expand_dims(images, 0)
    datagen = ImageDataGenerator(height_shift_range=range, fill_mode= "constant")
    it = datagen.flow(samples, batch_size = 1)

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.show()

up_down(img_array[0], 0.5)
up_down(img_array[0], -0.5)

# %%
def flip_horizontal(images):
    samples = expand_dims(images, 0)
    datagen = ImageDataGenerator(horizontal_flip = True, fill_mode= "constant")
    it = datagen.flow(samples, batch_size = 1)

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.show()

flip_horizontal(img_array[0])

# %%
def flip_vertical(images):
    samples = expand_dims(images, 0)
    datagen = ImageDataGenerator(vertical_flip = True, fill_mode= "constant")
    it = datagen.flow(samples, batch_size = 1)

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.show()

flip_vertical(img_array[0])

# %%
def randomRotation(images):
    samples = expand_dims(images, 0)
    datagen = ImageDataGenerator(rotation_range = 180, fill_mode= "constant")
    it = datagen.flow(samples, batch_size = 1)

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.show()

randomRotation(img_array[0])

# %%
def randomBrightness(images, range=[0.2,1.0]):
    samples = expand_dims(images, 0)
    datagen = ImageDataGenerator(brightness_range=range)
    it = datagen.flow(samples, batch_size = 1)

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.show()

randomBrightness(img_array[0])


# %%
def randomZoom(images, range=[0.5,1.0]):
    samples = expand_dims(images, 0)
    datagen = ImageDataGenerator(zoom_range=range)
    it = datagen.flow(samples, batch_size = 1)

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.show()

randomZoom(img_array[0])

# %%

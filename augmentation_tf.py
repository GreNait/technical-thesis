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
from tensorflow.keras.layers import GaussianNoise

IMG_HEIGHT, IMG_WIDTH = 150,113

# %%
def left_right(move=[-200,200]):
    return ImageDataGenerator(rescale=1./255,width_shift_range=move,fill_mode= "constant",cval=255)

def up_down(range):
    return ImageDataGenerator(rescale=1./255,height_shift_range=range, fill_mode= "constant",cval=255)

def flip_horizontal():
    return ImageDataGenerator(rescale=1./255,horizontal_flip = True, fill_mode= "constant",cval=255)

def flip_vertical():
    return ImageDataGenerator(rescale=1./255,vertical_flip = True, fill_mode= "constant",cval=255)

def randomRotation():
    return ImageDataGenerator(rescale=1./255,rotation_range = 180, fill_mode= "constant",cval=255)

def randomBrightness(range=[0.2,1.0]):
    return ImageDataGenerator(rescale=1./255,brightness_range=range)

def randomZoom(range=[0.5,1.0]):
    return ImageDataGenerator(rescale=1./255,zoom_range=range)

def allAugmentations(move=[-20,20], height_shift_range=[-20,20], constant_value = 255, rotation_range = 180, brightness_range=[0.2,1.0],random_zoom_range=[0.7,1.3], vsplit=0.2):
    return ImageDataGenerator(  rescale=1./255,
                                width_shift_range=move,
                                height_shift_range = height_shift_range,
                                horizontal_flip = True,
                                vertical_flip = True,
                                rotation_range = rotation_range,
                                brightness_range=brightness_range,
                                fill_mode= "constant",
                                validation_split = vsplit,
                                cval=constant_value)

def loadDataIntoDatagen(path, batch_size, datagen, subset):
    return datagen.flow_from_directory( batch_size=batch_size,
                                        directory=path,
                                        shuffle=True,
                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                        class_mode='binary',
                                        subset=subset)


def showImage(img_array):
    fig, axes = plt.subplots(1, 5, figsize=(IMG_HEIGHT,IMG_WIDTH))
    axes = axes.flatten()
    for img, ax in zip( img_array, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def returnTrainDataGenerator():
    path_to_test_set = "/home/grenait/Desktop/technical_thesis/technical-thesis/test_set/"
    training_image_generator = allAugmentations()
    train_data_gen = loadDataIntoDatagen(path_to_test_set, 15, training_image_generator, 'training')

    # # Displaying the first 5 images
    # sample_training_images, _ = next(train_data_gen)
    # showImage(sample_training_images[:5])

    return train_data_gen

def returnValidationDataGenerator():
    path_to_test_set = "/home/grenait/Desktop/technical_thesis/technical-thesis/test_set/"
    validation_image_generator = allAugmentations()
    validation_data_gen = loadDataIntoDatagen(path_to_test_set, 15, validation_image_generator, 'validation')

    return validation_data_gen


# %%

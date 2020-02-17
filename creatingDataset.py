
#%%
# from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

#%%
#Specifiying the path, where the image data is stored
path_to_figures = "/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/overfitting"

#%%
# The names of the figures must be the same as the folder containing them
# we use these names to get the folder and load the data later into the 
# dataset.
figures = ["batman", "deadpool"]

#%%
for figure in figures:
    path = os.path.join(path_to_figures, figure)

    print(path)

    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break

#%%        
# We are normalizing the images to a smaller image for faster training
# in later models, we might use the bigger pictures. The first model is for proofing
# the drivers, data, and get an general sense, therefore accuracy is not as important
# as getting something to "run"    
img_width = 150 # 640 -> 150
img_heigth = 113 # 400 -> 112,5 for this case it was rounded up

normalized_array = cv2.resize(img_array, (img_width, img_heigth))
plt.imshow(normalized_array)
plt.show()

#%%
# Creating an empty dataset, which will later contain the image  
# and the label
training_data = []

def creatingData():
    figures = ["batman", "deadpool"]

    for figure in figures:
        path = os.path.join(path_to_figures, figure)

        # defining the label -> index 0 = batman, index 1 = deadpool etc.
        class_num = figures.index(figure)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                normalized_array = cv2.resize(img_array, (img_width, img_heigth))

                # adding the normalized image + label
                training_data.append([normalized_array, class_num])
            except Exception as e:
                print(e)

creatingData()
print(len(training_data))

# %%
# Now we are shuffling the data
import random

random.shuffle(training_data)

# %%
# To see that they are random, we print the label and the numbers should be shuffled

for sample in training_data[:10]:
    # print(sample[1])
    plt.imshow(sample[0])
    plt.show()

# %%
# Feature set, capital X normally -> used later for the model
X = []
# labels
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# The model expects numpy arrays -> -1 defines everythin is a feature inside of the array, the
# 3 is for 3 hannels (color) a 1 would be greysacle
X = np.array(X).reshape(-1, img_width, img_heigth, 3)

# %%
# Because we will tweak the model later on, we save the images
# so we do not have to reshape, normalize, etc. everything again
np.save('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/overfitting/training_data_X', X)
np.save('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/overfitting/training_data_y', y)


# %%

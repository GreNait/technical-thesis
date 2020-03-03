'''
Loading the configured images, add gaussian noise to them and
split them into training data and validation data.
'''
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2

#%%
# The names of the figures must be the same as the folder containing them
# we use these names to get the folder and load the data later into the 
# dataset.
figures = ["batman", "deadpool"]
path_to_collections = "/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/"
path_to_figures = [os.path.join(path_to_collections,figure) for figure in figures]

path_to_symlink =  "/home/grenait/Desktop/technical_thesis/technical-thesis/test_symlink/"

for figure in figures:
    try:
        os.mkdir(os.path.join(path_to_symlink, figure))
    except Exception as e:
        pass

symlink_path_to_figures = [os.path.join(path_to_symlink,figure) for figure in figures]

# %%
# Creating symlinks for every image
for symlink, path in zip(symlink_path_to_figures,path_to_figures):
    files = os.listdir(path)

    for f in files:
        

   

    
# %%
def addNoise(img_array):
    return random_noise(img_array[0], mode='gaussian')
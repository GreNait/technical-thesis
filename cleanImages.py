# Cleaning the images. Removing the stepper motor by filling the pixels bellow
# with white pixels. At the same time, generate a image from the cleaned image
# with gaussian noise added

# %%
from skimage.util import random_noise
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# "batman", "deadpool"
figures = ["antman"]
dataCollection_folder = Path("/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/")
saving_folder = Path("/home/grenait/Desktop/technical_thesis/technical-thesis/test_set/")

def removeBottom(img, row):
    (values, counts) = np.unique(img, return_counts=True)
    ind=np.argmax(counts)

    for i in range(row, len(img)):
        img[i] = np.array([values[ind],values[ind],values[ind]])

    return img

def addNoise(img):
    return random_noise(img, mode='gaussian')

def showImg(img):
    plt.imshow(img)
    plt.show()
    
# %%
for figure in figures:
    image_folder = dataCollection_folder/figure
    
    for img in image_folder.iterdir():
        img = plt.imread(str(img))
        showImg(img)
        break

    row = input("Starting at which row?")

    for img in image_folder.iterdir():
        img = plt.imread(str(img))
        clean_img = removeBottom(img, int(row))
        label = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"_"+str(f"{figure}_clean.png")
        matplotlib.image.imsave(str(saving_folder/figure/label), clean_img)

        noise_img = addNoise(img) 
        label = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"_"+str(f"{figure}_noise.png")
        matplotlib.image.imsave(str(saving_folder/figure/label), noise_img)
        
    
# %%

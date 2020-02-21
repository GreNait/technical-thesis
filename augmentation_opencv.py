# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %%
def moveLeftRight(img, factor):

    height, width = img.shape[:2]
    tx = width/factor

    translation_matrix = np.float32([
        [1,0,tx],
        [0,1,0]
    ])

    final_size = (width, height)
    image_translate = cv2.warpAffine(img, translation_matrix, final_size)

    return image_translate

# %%
def moveUpDown(img, factor):
    height, width = img.shape[:2]
    ty = height/factor

    translation_matrix = np.float32([
        [1,0,0],
        [0,1,ty]
    ])

    final_size = (width, height)
    image_translate = cv2.warpAffine(img, translation_matrix, final_size)

    return image_translate

# %%
def moveDiagonal(img, factor):
    height, width = img.shape[:2]
    tx = width/factor
    ty = height/factor

    translation_matrix = np.float32([
        [1,0,tx],
        [0,1,ty]
    ])

    final_size = (width, height)
    image_translate_1 = cv2.warpAffine(img, translation_matrix, final_size)

    translation_matrix = np.float32([
        [1,0,tx],
        [0,1,-ty]
    ])

    final_size = (width, height)
    image_translate_2 = cv2.warpAffine(img, translation_matrix, final_size)

    return image_translate_1, image_translate_2

# %%
def rotate(img, degree):
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), degree, 1
    )

    image_rotated = cv2.warpAffine(
        img, rotation_matrix, (width, height)
    )

    return image_rotated


# %%
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
        plt.imshow(moveLeftRight(img_array, 3))
        plt.show()
        plt.imshow(moveLeftRight(img_array,4))
        plt.show()
        plt.imshow(moveLeftRight(img_array,-3))
        plt.show()
        plt.imshow(moveLeftRight(img_array,-4))
        plt.show()        
        plt.imshow(moveUpDown(img_array,4))
        plt.show()
        plt.imshow(moveUpDown(img_array,5))
        plt.show()
        plt.imshow(moveUpDown(img_array,-4))
        plt.show()
        plt.imshow(moveUpDown(img_array,-5))
        plt.show()
        break

# %%
moveDiagonal(5)
moveDiagonal(-5)

#%%
rotate(45)
rotate(-45)
rotate(90)
rotate(-90)
rotate(135)
rotate(-135)
rotate(180)

# %%

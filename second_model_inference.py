# %%
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

path_to_image = "/home/grenait/Desktop/technical_thesis/technical-thesis/smartPhone_batman.jpeg"
path_to_other_image = "/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/deadpool/2020-01-31 15:47:02_deadpool.png"
path_to_model = "/home/grenait/Desktop/technical_thesis/technical-thesis/trained_models/second_model.h5"
# %%
def reshapeImages(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150,113))
    plt.imshow(img)
    plt.show()
    img = img.reshape(-1,150,113,3)
    img = img / 255
    return img

# %%
test_images = []
test_images.append(reshapeImages(path_to_image))
test_images.append(reshapeImages(path_to_other_image))
# %%
model = tf.keras.models.load_model(str(path_to_model))
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# %%
prediction = probability_model.predict(test_images)

print(np.argmax(prediction))


# %%

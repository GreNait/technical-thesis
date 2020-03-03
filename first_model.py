#%%
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

# %%
# Normal data
X = np.load('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/training_data_X.npy')
y = np.load('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/training_data_y.npy')

#%%
# Data for overfitting
# X = np.load('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/overfitting/training_data_X.npy')
# y = np.load('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/overfitting/training_data_y.npy')

# %%
# Normalizing the image data 255 -> 1, 0 -> 0
X = X/255

# %%
model = Sequential()
model.add(Conv2D(16, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()
# %%
model.fit(X,y, batch_size = 10, epochs=300, validation_split=0.1)



# %%

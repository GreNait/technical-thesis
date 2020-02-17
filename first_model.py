#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
  except RuntimeError as e:
    print(e)

NAME = f"First-model-test {int(time.time())}"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')


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
model.add(Conv2D(256, (3,3), input_shape = X.shape[1:]))
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

# %%
model.fit(X,y, batch_size = 20, epochs=1000, validation_split=0.1, callbacks=[tensorboard])


# %%

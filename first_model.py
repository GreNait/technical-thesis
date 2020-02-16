#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

NAME = f"First-model-test {int(time.time())}"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')


# %%
X = np.load('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/training_data_X.npy')
y = np.load('/home/grenait/Desktop/technical_thesis/technical-thesis/dataCollection/training_data/training_data_y.npy')

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
model.fit(X,y, batch_size = 25, epochs=100, validation_split=0.1, callbacks=[tensorboard])


# %%

#%%
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time
import augmentation_tf
import os
import matplotlib.pyplot as plt
from pathlib import Path

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

IMG_HEIGHT, IMG_WIDTH = 150,113
batch_size = 15
epochs = 15

# Load the training data generator from the augmentation modul
train_data_gen = augmentation_tf.returnTrainDataGenerator()
validation_data_gen = augmentation_tf.returnValidationDataGenerator()

# %%
# total_train = len(os.listdir("/home/grenait/Desktop/technical_thesis/technical-thesis/test_set/batman")) + len(os.listdir("/home/grenait/Desktop/technical_thesis/technical-thesis/test_set/deadpool"))
total_train = 2313
total_val = 577

# %%
model = Sequential([
    Conv2D(16, 3, padding='same', activation='sigmoid', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.1),
    Conv2D(32, 3, padding='same', activation='sigmoid'),
    MaxPooling2D(),
    Dropout(0.1),
    Conv2D(64, 3, padding='same', activation='sigmoid'),
    MaxPooling2D(),
    Dropout(0.1),
    Flatten(),
    Dense(512, activation='sigmoid'),
    Dropout(0.1),
    Dense(256, activation='sigmoid'),
    Dropout(0.1),
    Dense(128, activation='sigmoid'),
    Dense(2)
])

# %%
# Saving the model in checkpoints is valuable if somethin happens during the training
checkpoint_dir = Path.cwd() / "checkpoints" / "second_model_cp_{epoch:04d}.cpkt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = str(checkpoint_dir),
    verbose=1,
    save_weights_only=True,
    period=3
)
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# %%
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    callbacks=[cp_callback],
    validation_data=validation_data_gen
)

# %%
acc = history.history['accuracy']
# only working with validation data
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
# Saving the model
model.save(str(Path.cwd() / "trained_models" / "second_model.h5"))


# %%

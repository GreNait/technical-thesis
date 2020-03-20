# This modul trains several models retained by models.py.
# It also saves the models and the plots for later analysis.
# This should be automised as much as possible, so only the models 
# structure can be changed later and trained/compared afterwards.

#%%
#import basic config
import default
IMG_WIDTH, IMG_HEIGHT, DIMENSION, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_SHAPE, BATCH_SIZE, EPOCHS = default.getAllParameters()

import tensorflow as tf
print(tf.__version__) #Should be 2.1.x or higher
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import numpy as np
from pathlib import Path
import os

# %%
# import the modul containing all models
from models import ModelComparison
mc = ModelComparison()
models = [
            mc.returnSmall(), 
            mc.returnMedium(), 
            mc.returnBig(), 
            mc.returnCnnSmall(), 
            mc.returnCnnMedium(), 
            mc.returnCnnBig()
            ]

for m in models:
    print(f'Loaded model: {m[1]}')

# %%
# Creating a folder to save the resuölt of the models later on
cwd = Path.cwd()
if Path(cwd / 'trained_models').is_dir():
    print("Trained model folder exists")
else:
    os.mkdir(cwd / 'trained_models')
    print("Trained model folder created")

path_trained_models = Path(cwd / 'trained_models')

# %%
# Creating the optimizer for each model
for m in models:
    m[0].compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
# Savin the model summary of every model in the trained model folder as a JSON
for m in models:
    with open(path_trained_models/f'json_{m[1]}','w') as f:
        f.write(m[0].to_json())

# %%
# Load the training data generator from the augmentation modul
import augmentation_tf
train_data_gen = augmentation_tf.returnTrainDataGenerator()
validation_data_gen = augmentation_tf.returnValidationDataGenerator()

# total_train = len(os.listdir("/home/grenait/Desktop/technical_thesis/technical-thesis/test_set/batman")) + len(os.listdir("/home/grenait/Desktop/technical_thesis/technical-thesis/test_set/deadpool"))
total_train = 3476
total_val = 867
# %%
# Creting callback to visualize the model and stop if the model is not
# learning anymore
callbacks = [
    TensorBoard(
    log_dir=str(path_trained_models/'logs'), histogram_freq=1, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2),
    EarlyStopping(
    monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False),
    ModelCheckpoint(
    filepath = str(path_trained_models / 'cp_{epoch:04d}.cpkt'),
    verbose=1,
    save_weights_only=True,
    period=3)
]

# %%
# Train each model and show the plot afterwards
for m in models:
    history = m[0].fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=1, # set to 1 epoch for testing
        callbacks=[callbacks],
        validation_data=validation_data_gen
    )

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

    model.save(str(path_trained_models / f'{m[1]}.h5'))
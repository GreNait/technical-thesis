# This modul contains default parameters for all other
# moduls to be read in. Later this modul should be able 
# to read a JSON file and get the information and parameters
# based on that file

IMG_WIDTH = 150
IMG_HEIGHT = 113
DIMENSION = 3
DEFAULT_ACTIVATION = 'relu'
DEFAULT_OUTPUT_SHAPE = 2
BATCH_SIZE = 15
EPOCHS = 15

def getAllParameters():
    return IMG_WIDTH, IMG_HEIGHT, DIMENSION, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_SHAPE, BATCH_SIZE, EPOCHS

#TODO Add JSON read in
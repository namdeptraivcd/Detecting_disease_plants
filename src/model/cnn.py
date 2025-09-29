import tensorflow as tf
from keras import models, layers
import keras
from config.config import Config as cfg


class CNN():
    def __init__ (self):
        
        self.classes = cfg.CLASSES
        self.input_shape = (cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS)

        self.resize_and_rescale = keras.Sequential([
            keras.layers.Resizing(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            keras.layers.Rescaling(1./cfg.IMAGE_SIZE - 1)
        ])

        self.data_augmentation = keras.Sequential([
            layers.RandomFlip("Horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])

    def convolutional_neural_network(self):

        net = models.Sequential([
            self.resize_and_rescale,
            self.data_augmentation,
            layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape= self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.classes, activation='softmax'),
        ])





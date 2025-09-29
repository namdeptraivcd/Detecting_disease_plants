from utils.utils import data_loader, load_image
import tensorflow as tf


def main():
    dataset = load_image("data/PlantViallge")

    train_dataset, val_dataset, test_dataset = data_loader(dataset)


    """
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)  
    """

    
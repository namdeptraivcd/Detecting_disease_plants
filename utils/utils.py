import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def load_image(path_to_folder, batch_size: int, image_size: tuple[int, int]):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(path_to_folder,
                                                    shuffle = True,
                                                    image_size = image_size,
                                                    batch_size = batch_size)

    return dataset

def visualize (dataset, class_names):
    plt.figure(figsize= (10, 10))
    for image_batch, batch_labels in dataset.take(1):
        for i in range (12):
            x = plt.subplot(4, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[batch_labels[i]])
            plt.axis("off")
    plt.savefig("preview.png")


def get_data (dataset, train_split=0.8, val_split = 0.1, test_split= 0.1, shuffle = True ):
    data_size = len(dataset)

    dataset = dataset.shuffle(buffer_size = len(dataset))

    train_size = int(train_split * data_size)
    val_size = int (val_split * data_size)
    test_size = int (test_split * data_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size)

    return train_dataset, val_dataset, test_dataset


#Test function
if __name__ == "__main__":
    dataset = load_image("data/PlantVillage", 32,  (256, 256))
    class_names = dataset.class_names

    train_dataset, val_dataset, test_dataset = get_data(dataset)
    print(len(train_dataset))

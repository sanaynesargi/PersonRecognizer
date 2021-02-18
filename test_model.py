import matplotlib.pyplot as plt
import numpy as np
import os, sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from random import choice
from cv2 import imread
from train_model import load_data, TEST_SIZE
from sklearn.model_selection import train_test_split

def main():
    
    filepath = './saved_models/model'
    model = load_model(filepath)

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(list(labels))
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    for i in range(len(x_test[:3])):
        plt.imsave(f"Image #{i}.jpg", x_test[i])

    predictions = model.predict(x_test[:1])
    print(x_test[:1].shape, x_test[0].shape)
    classes = np.argmax(predictions, -1)
    truth_table = {
        0: "Atom",
        1: "Sanay",
        2: "Aarav"
    }


    for i in range(len(classes)):
        print(f"Image #{i} is {truth_table[classes[i]]}")




if __name__ == "__main__":
    main()
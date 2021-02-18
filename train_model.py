import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split

EPOCHS = 1
IMG_WIDTH = 383
IMG_HEIGHT = 505
NUM_CATEGORIES = 3
TEST_SIZE = 0.4
CHECKPOINT_PATH = "training/cp.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)



def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [checkpoint_dir]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(list(labels))
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Get a compiled neural network
    model = get_model()


    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[cp_callback], verbose=1)

    # # Evaluate neural network performance
    # model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    
    filename = "saved_models/model"
    model.save(filename)
    print(f"Model saved to {filename}.")



def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = list()
    labels = list()
    
    for directory in range(NUM_CATEGORIES):

        dir_path = os.path.join(data_dir, str(directory))

        dir_path = os.path.join(data_dir, str(directory))
        
        for img in os.listdir(dir_path):    
            image = cv2.imread(os.path.join(dir_path, img))
            images.append(image)
            labels.append(directory)
            

    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    layers = []

    layers.extend([
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", 
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2))
    ])

    layers.extend([
        # Flatten
        tf.keras.layers.Flatten(),

        # Add Hidden Layers with 128 units with reLU activation
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),

        # Add dropout to reduce chance of overfitting
        tf.keras.layers.Dropout(0.5),

        # Make output layer 1 unit for each category with softmax activation
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])


    model = tf.keras.Sequential(layers=layers)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    if len(sys.argv) == 3:
        model.load_weights(CHECKPOINT_PATH)

        
    return model
    

if __name__ == "__main__":
    main()
import sys
import os
import tensorflow as tf
from os import path
from tensorflow import keras
from trainConvnet import TrainConvnet
from freezeModel import FreezeModel
from keras import layers, models, utils
from keras.models import load_model


def main():
    """Main method run at start up."""

    args = parse_args(sys.argv)
    __training_path = args[0]
    __initialize = args[1]
    __freeze = args[2]
    __image_width = 300
    __image_height = 300
    __model_path = path.abspath(path.dirname(path.dirname(__file__)) + "/model")
    __model_file = "/model.h5"
    __model = initialize_convent(__model_path, __model_file, __initialize, __image_height, __image_width, __training_path)

    if not __training_path == "":
        __model = train_convnet(__training_path, __model, __image_width, __image_height)
        __model.save(__model_path + "/" + __model_file)
        __model.save_weights(__model_path + "/saved_weights.h5")
        model_json = __model.to_json()
        with open(__model_path + '/keras_model.json', "w") as json_file:
            json_file.write(model_json)

    if __freeze:
        if export_convnet(__model_path, __model_file):
            print("Model frozen successfully.")


def parse_args(argv):
    """This method parses the command line arguments from starting the application and returns them to the main
     function.

     argv: the arguments passed in from the command prompt"""

    args = ["", "", False, False]
    last_arg = ""

    for arg in argv:

        if last_arg == "-training":

            if not path.exists(arg):
                print("Path " + arg + " does not exist. Terminating.")
                exit()
            else:
                args[0] = path.abspath(arg)
                print("Training path is " + str(args[0]))

        if arg == "-init":
            i = input("User is requesting to initialize model. Are you sure? [Type \'Yes\']: ")

            if i == "Yes":
                print("Model will be initialized.")
                args[1] = True
            else:
                print("Model will not be initialized. Terminating.")
                exit()

        if arg == "-freeze":
            args[2] = True

        if arg == "-help":
            print("BinBot Training software use:")
            print("-training [path]: Designate the path to the training images.")
            print("-test [path]: Designate the path to the test images.")
            print("-init: Initialize the neural network data model before training.")
            print("-freeze: Freeze the existing model after other processes.")
            exit()

        last_arg = arg

    if not (path.exists(args[0])) and not args[2]:
        print("Not enough valid path arguments found. Terminating.")
        exit()

    return args


def initialize_convent(model_path, model_file, initialize, image_height, image_width, training_path):
    """This method will load the existing convnet, or initialize it if none exists

    model_path: the expected file path to the existing model"""
    """
    __model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(image_height, image_width, 3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    """

    training_list = os.listdir(training_path)
    classes = len(training_list)
    input_shape = (image_height, image_width, 3)

    __model = models.Sequential([
        layers.Conv2D(16, kernel_size=(3, 3), padding='same', strides=1, input_shape=input_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(16, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=1),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        layers.Dense(1240),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(640),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(480),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(120),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(62),
        layers.LeakyReLU(alpha=0.2),

        layers.Dense(classes + 4),
        layers.LeakyReLU(alpha=0.2)
    ])

    if not path.exists(model_path):
        os.mkdir(model_path)

    if path.exists(path.abspath(model_path + model_file)) and not initialize:
        __model = load_model(path.abspath(model_path + model_file), compile=False)
        print("Existing model loaded.")
    else:
        __model.save(path.abspath(model_path + model_file))
        print("New model created and saved.")

    return __model


def train_convnet(training_path, model, image_width, image_height):
    """This method will begin the training of the convnet by calling the TrainConvnet class

    training_path: file path to the set of training images
    test_path: file path to the set of test images
    model: the convnet data model"""
    __training = TrainConvnet(training_path, model, image_width, image_height)
    __model = __training.start()
    return __model


def export_convnet(model_path, model_file):
    """This method will export the model to a file by calling the FreezeModel class

    model: the convnet data model"""
    __freeze = FreezeModel(model_path, model_file)
    __freeze.start()
    return __freeze.success


main()

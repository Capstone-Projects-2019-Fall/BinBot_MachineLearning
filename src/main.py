import sys
import os
from os import path
import tensorflow as tf
from trainConvnet import TrainConvnet
from freezeModel import FreezeModel
from keras import layers, models, utils


def main():
    """Main method run at start up."""

    args = parse_args(sys.argv)
    __training_path = args[0]
    __test_path = args[1]
    __initialize = args[2]

    for arg in args:
        print(str(arg))

    __model_path = path.abspath(path.dirname(path.dirname(__file__)) + "/model")
    initialize_convent(__model_path, __initialize)


def parse_args(argv):
    """This method parses the command line arguments from starting the application and returns them to the main
     function.

     argv: the arguments passed in from the command prompt"""

    args = ["", "", False]
    last_arg = ""

    for arg in argv:

        if last_arg == "-training":

            if not path.exists(arg):
                print("Path " + arg + " does not exist. Terminating.")
                exit()
            else:
                args[0] = path.abspath(arg)
                print("Training path is " + str(args[0]))

        if last_arg == "-test":

            if not path.exists(arg):
                print("Path " + arg + " does not exist. Terminating.")
                exit()
            else:
                args[1] = path.abspath(arg)
                print("Test path is " + str(args[1]))

        if arg == "-init":
            i = input("User is requesting to initialize model. Are you sure? [Type \'Yes\']: ")

            if i == "Yes":
                print("Model will be initialized.")
                args[2] = True
            else:
                print("Model will not be initialized. Terminating.")
                exit()

        if arg == "-help":
            print("BinBot Training software use:")
            print("-training [path]: Designate the path to the training images.")
            print("-test [path]: Designate the path to the test images.")
            print("-init: Initialize the neural network data model before training.")
            exit()

        last_arg = arg

    if not path.exists(args[0]) or not path.exists(args[1]):
        print("Not enough valid path arguments found. Terminating.")
        exit()

    return args


def initialize_convent(model_path, initialize):
    """This method will load the existing convnet, or initialize it if none exists

    model_path: the expected file path to the existing model"""
    __model = models.Sequential()
    __model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    __model.add(layers.MaxPooling2D((2, 2)))
    __model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    __model.add(layers.MaxPooling2D((2, 2)))
    __model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    if not path.exists(model_path):
        if __name__ == '__main__':
            os.mkdir(model_path)

    model_file = "/model.ckpt"

    if path.exists(model_path + model_file) and not initialize:
        __model.load_weights(model_path + model_file)
        print("Existing model loaded.")
    else:
        __model.save_weights(model_path + model_file)
        print("New model created and saved.")

    return __model


def train_convnet(training_path, test_path, model):
    """This method will begin the training of the convnet by calling the TrainConvnet class

    training_path: file path to the set of training images
    test_path: file path to the set of test images
    model: the convnet data model"""
    __training = TrainConvnet(training_path, test_path, model)


def export_convnet(model):
    """This method will export the model to a file by calling the FreezeModel class

    model: the convnet data model"""
    __freeze = FreezeModel(model)
    return __freeze.success


main()

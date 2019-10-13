import argparse
from src.trainConvnet import TrainConvnet
from src.freezeModel import  FreezeModel
from keras import layers, models, utils


def main():
    """Main method run at start up."""
    print("Main function called.")


def parse_args(args):
    """This method parses the command line arguments from starting the application and returns them to the main
     function.

     args: the arguments passed in from the command prompt"""
    __parser = argparse.ArgumentParser()
    args = __parser.parse_args()
    return args


def initialize_convent(model_path):
    """This method will load the existing convnet, or initialize it if none exists

    model_path: the expected file path to the existing model"""
    __file = utils.get_file(model_path)

    __model = models.Sequential()
    __model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    __model.add(layers.MaxPooling2D((2, 2)))
    __model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    __model.add(layers.MaxPooling2D((2, 2)))
    __model.add(layers.Conv2D(64, (3, 3), activation='relu'))

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

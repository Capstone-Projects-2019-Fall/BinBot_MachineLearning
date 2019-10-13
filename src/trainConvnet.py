import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt


class TrainConvnet:
    def __init__(self, training_path, test_path, model):
        self.__training_path = training_path
        self.__test_path = test_path
        self.__model = model

    def start(self):
        """This method will be called externally by main to begin training the model"""
        pass

    def __load_images(self, path):
        """This method will load the images from the designated directory for training / testing

        path: File path to the relevant set of images being loaded"""
        pass

    def __train_model(self, model):
        """This method will operate the loop that trains the convnet on the loaded images

        model: the convnet data model"""
        pass

    def __display_results(self):
        """This method will output the results of the training to the command prompt, such as the success and
        loss between training and testing the model"""
        pass




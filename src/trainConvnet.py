import tensorflow as tf
import os
from tensorflow import keras
from keras import datasets, layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class TrainConvnet:
    def __init__(self, training_path, test_path, model):
        self.__training_path = training_path
        self.__test_path = test_path
        self.__training_total = 0
        self.__test_total = 0
        self.__training_classes = 0
        self.__test_classes = 0
        self.__model = model
        self.__training_gen = ImageDataGenerator()
        self.__test_gen = ImageDataGenerator()

    def start(self):
        """This method will be called externally by main to begin training the model"""
        training_list = os.listdir(self.__training_path)
        test_list = os.listdir(self.__test_path)
        self.__training_classes = len(training_list)
        self.__test_classes = len(os.listdir(self.__test_path))
        training_classes = []
        test_classes = []

        for directory in training_list:
            training_classes.append(directory)

        for directory in test_list:
            test_classes.append(directory)

        self.__training_gen = self.__load_images(self.__training_path)
        self.__test_gen = self.__load_images(self.__test_path)

    def __load_images(self, path):
        """This method will load the images from the designated directory for training / testing

        path: File path to the relevant set of images being loaded"""
        generator = ImageDataGenerator(rescale=1./255)
        data_gen = generator.flow_from_directory(batch_size=128,
                                                 directory=path,
                                                 shuffle=True,
                                                 target_size=(150, 150),
                                                 class_mode='binary')
        return data_gen

    def __train_model(self, model):
        """This method will operate the loop that trains the convnet on the loaded images

        model: the convnet data model"""
        pass

    def __display_results(self):
        """This method will output the results of the training to the command prompt, such as the success and
        loss between training and testing the model"""
        pass




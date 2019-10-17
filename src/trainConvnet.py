from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
from os import path
from tensorflow import keras
from keras import datasets, layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


class TrainConvnet:
    def __init__(self, training_path, test_path, model, image_width, image_height):
        self.__training_path = training_path
        self.__test_path = test_path
        self.__training_total = 0
        self.__test_total = 0
        self.__training_classes = 0
        self.__test_classes = 0
        self.__model = model
        self.__training_gen = ImageDataGenerator()
        self.__test_gen = ImageDataGenerator()
        self.__image_width = image_width
        self.__image_height = image_height

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
            self.__training_total += len(os.listdir(path.abspath(self.__training_path + '/' + directory)))

        for directory in test_list:
            test_classes.append(directory)
            self.__test_total += len(os.listdir(path.abspath(self.__test_path + '/' + directory)))

        print("Training total, test total = " + str(self.__training_total) + ', ' + str(self.__test_total))

        self.__training_gen = self.__load_images(self.__training_path)
        self.__test_gen = self.__load_images(self.__test_path)
        history = self.__train_model(self.__model)
        self.__display_results(history)

    def __load_images(self, image_path):
        """This method will load the images from the designated directory for training / testing

        path: File path to the relevant set of images being loaded"""
        generator = ImageDataGenerator(rescale=1./255)
        data_gen = generator.flow_from_directory(batch_size=128,
                                                 directory=image_path,
                                                 shuffle=True,
                                                 target_size=(self.__image_height, self.__image_width),
                                                 class_mode='binary')
        return data_gen

    def __train_model(self, model):
        """This method will operate the loop that trains the convnet on the loaded images

        model: the convnet data model"""
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        history = model.fit_generator(
            self.__training_gen,
            steps_per_epoch=self.__training_total,
            epochs=15,
            validation_data=self.__test_gen,
            validation_steps=self.__test_total


        )
        return history

    def __display_results(self, history):
        """This method will output the results of the training to the command prompt, such as the success and
        loss between training and testing the model"""
        accuracy = history.history['accuracy']
        value_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        value_loss = history.history['val_loss']
        epochs_range = range(15)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, accuracy, label='Training Accuracy')
        plt.plot(epochs_range, value_accuracy, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, value_loss, label='Validation Loss')
        plt.legend('upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def __plot_images(self, images):
        """Optional method to display training images for testing / validation."""
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()





import tensorflow as tf
import os
from os import path


class FreezeModel:
    def __init__(self, model):
        self.__model = model
        self.success = False

    def start(self):
        """This method will be called externally by main to begin exporting the model to a file"""
        self.__freeze_convnet()

    def __freeze_convnet(self):
        """This method will export the convnet model to an external file"""
        export_path = path.abspath(path.dirname(path.dirname(__file__)) + "/frozen")
        if not path.exists(export_path):
            os.mkdir(export_path)

        # TODO: Code to freeze model and export

        print("Exporting model to " + export_path)

        self.success = True

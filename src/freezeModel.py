import tensorflow as tf


class FreezeModel:
    def __init__(self, model):
        self.__model = model
        self.success = False

    def start(self):
        """This method will be called externally by main to begin exporting the model to a file"""
        pass

    def __freeze_convnet(self):
        """This method will export the convnet model to an external file"""
        pass

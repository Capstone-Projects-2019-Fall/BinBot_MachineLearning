from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from tensorflow import keras
from PIL import Image, ImageDraw
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.backend import maximum, minimum
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


class TrainConvnet:
    def __init__(self, training_path, model, image_width, image_height):
        self.__training_path = training_path
        self.__training_total = 0
        self.__test_total = 0
        self.__training_classes = []
        self.__test_classes = []
        self.__training_boxes = []
        self.__test_boxes = []
        self.__model = model
        self.__training_images = ImageDataGenerator()
        self.__test_images = ImageDataGenerator()
        self.__image_width = image_width
        self.__image_height = image_height

    def start(self):
        """This method will be called externally by main to begin training the model"""
        self.__training_images = self.__load_images(self.__training_path)
        self.__training_boxes, self.__training_classes = self.__load_bounding_boxes(self.__training_path)
        print("Training total = " + str(len(self.__training_images)))

        boxes = np.array(self.__training_boxes)
        encoder = LabelBinarizer()
        classes_onehot = encoder.fit_transform(self.__training_classes)

        Y = np.concatenate([boxes, classes_onehot], axis=1)
        X = np.array(self.__training_images)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

        history = self.__train_model(x_train, x_test, y_train, y_test)
        self.__display_results(history)

        target_boxes = y_test * self.__image_height
        pred = self.__model.predict( x_test )
        pred_boxes = pred[ ... , 0 : 4 ] * self.__image_height
        pred_classes = pred[ ... , 4 : ]
        iou_scores = self.__calculate_iou( target_boxes , pred_boxes )

        print( 'Class Accuracy is {} %'.format( self.__calculate_class_accuracy( y_test[ ... , 4 : ] , pred_classes ) * 100 ))

        boxes = self.__model.predict( x_test )
        for i in range( boxes.shape[0] ):
            b = boxes[ i , 0 : 4 ] * self.__image_height
            img = x_test[i] * 255
            source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )
            source_img = source_img.resize((2992, 2000))
            draw = ImageDraw.Draw( source_img )
            draw.rectangle( b , outline="red" )
            if not os.path.exists("inference_images"):
                os.mkdir("inference_images")
            source_img.save( 'inference_images/image_{}.png'.format( i + 1 ) , 'png' )

        return self.__model

    def __load_images(self, image_path):
        """This method will load the images from the designated directory for training / testing

        path: File path to the relevant set of images being loaded"""
        classes = os.listdir(image_path)
        images = []

        for class_name in classes:
            image_paths = glob.glob(image_path + "/" + class_name + '/*.jpg')

            for image_file in image_paths:
                image = Image.open(image_file).resize((self.__image_width, self.__image_height))
                image = np.asarray(image) / 255.0
                images.append(image)

        return images

    def __load_bounding_boxes(self, image_path):
        """This method will load the bounding box data from the designated directory for training / testing

        path: File path to the relevant set of xml files being loaded"""
        classes = os.listdir(image_path)
        bounding_boxes = []
        classes_raw = []

        for class_name in classes:
            annotation_paths = glob.glob(image_path + "/" + class_name + '/*.xml')

            for xml_file in annotation_paths:
                x = xmltodict.parse(open(xml_file, 'rb'))
                bounding_box = x['annotation']['object']['bndbox']
                bounding_box = np.array([int(bounding_box['xmin']), int(bounding_box['ymin']), int(bounding_box['xmax']),
                                         int(bounding_box['ymax'])])
                bounding_box2 = [None] * 4
                bounding_box2[0] = bounding_box[0]
                bounding_box2[1] = bounding_box[1]
                bounding_box2[2] = bounding_box[2]
                bounding_box2[3] = bounding_box[3]
                bounding_box2 = np.array(bounding_box2) / self.__image_width
                bounding_boxes.append(bounding_box2)
                classes_raw.append(x['annotation']['object']['name'])

        return bounding_boxes, classes_raw

    def __train_model(self, x_train, x_test, y_train, y_test):
        """This method will operate the loop that trains the convnet on the loaded images

        model: the convnet data model"""

        self.__model.compile(
            optimizer=Adam(lr=0.0001),
            loss=self.__calculate_loss,
            metrics=[self.__calculate_iou]
        )

        self.__model.summary()

        history = self.__model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=100,
            batch_size=3
        )

        return history

    def __calculate_iou(self, target_boxes, predicted_boxes):
        xa = maximum(target_boxes[..., 0], predicted_boxes[..., 0])
        ya = maximum(target_boxes[..., 1], predicted_boxes[..., 1])
        xb = minimum(target_boxes[..., 2], predicted_boxes[..., 2])
        yb = minimum(target_boxes[..., 3], predicted_boxes[..., 3])
        inter_area = maximum(0.0, xb - xa) * maximum(0.0, yb - ya)
        box_a_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        box_b_area = (predicted_boxes[..., 2] - predicted_boxes[..., 0]) * (predicted_boxes[..., 3] - predicted_boxes[..., 1])
        iou = inter_area / (box_a_area + box_b_area - inter_area)
        return iou

    def __calculate_loss(self, y_true, y_predicted):
        mse = tf.losses.mean_squared_error(y_true, y_predicted)
        iou = self.__calculate_iou(y_true, y_predicted)
        return mse + (1-iou)

    def __calculate_class_accuracy(self, target_classes, predicted_classes):
        target_classes = np.argmax(target_classes, axis=1)
        predicted_classes = np.argmax(predicted_classes, axis=1)
        return (target_classes == predicted_classes).mean()

    def __display_results(self, history):
        """This method will output the results of the training to the command prompt, such as the success and
        loss between training and testing the model"""
        # accuracy = history.history['accuracy']
        # value_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        value_loss = history.history['val_loss']
        epochs_range = range(100)

        plt.figure(figsize=(8, 8))
        """
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, accuracy, label='Training Accuracy')
        plt.plot(epochs_range, value_accuracy, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        """
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





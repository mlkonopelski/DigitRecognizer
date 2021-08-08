import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import xception
import matplotlib.pyplot as plt
from config import preprocess_params
from visualizations.image_visualizations import visualize_image
import os


class PreprocessImage:
    def __init__(self, img_path):
        self.image = cv.imread(cv.samples.findFile(img_path))

        def preprocess_image(img):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # if pictures where in colour
            img = cv.blur(img, (3, 3))
            img = cv.Canny(img, preprocess_params.canny_treshold, preprocess_params.canny_treshold * 2)
            return img

        def return_rectangles(img):
            contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            rectangles = []
            for i in range(len(contours)):
                hull = cv.convexHull(contours[i])
                x, y, w, h = cv.boundingRect(hull)
                rectangles.append([x, y, w, h])

            rectangles = cv.groupRectangles(rectangles, 1)
            return rectangles

        def cut_numbers_from_canvas(img, rectangles):
            resized_numbers = []
            for i in range(len(rectangles[0])):
                x, y, w, h = rectangles[0][i]
                croped = img[y:y + h, x:x + w]
                resized = cv.resize(croped, (299, 299))
                preprocessed = xception.preprocess_input(resized).reshape(299, 299, 3)
                resized_numbers.append(preprocessed)
            resized_numbers = np.stack(resized_numbers, axis=0)
            return resized_numbers

        self.rectangles = return_rectangles(preprocess_image(self.image))
        self.numbers = cut_numbers_from_canvas(self.image, self.rectangles)
        self.dataset = tf.data.Dataset.from_tensor_slices(self.numbers).batch(32)


def preprocess_images():
    images = []
    included_extensions = ['jpg', 'jpeg', 'bmp', 'png']
    file_names = [fn for fn in os.listdir('test_samples') if any(fn.endswith(ext) for ext in included_extensions)]
    for img_path in file_names:
        preprocessed_image = PreprocessImage(os.path.join('test_samples', img_path))
        images.append([img_path, preprocessed_image.rectangles, preprocessed_image.numbers])
    if len(images) == 0:
        print('There are no pictures to be predicted')
    return images


if __name__ == '__main__':

    images = PreprocessImage('../test_samples/Example1.png').numbers

    for img in images:
        visualize_image(img)

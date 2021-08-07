import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import xception
import matplotlib.pyplot as plt
from config import preprocess_params
from visualizations.show_image import visualize_image


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


if __name__ == '__main__':

    images = PreprocessImage('../test_samples/Example1.png').numbers

    for img in images:
        visualize_image(img)

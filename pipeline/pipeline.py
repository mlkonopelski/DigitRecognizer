import csv
import os
from datetime import date
from urllib.request import urlretrieve

import numpy as np

from config import model_params
from preprocess_images.preprocess import preprocess_images
from train_model.model import TrainModel
from visualizations.image_visualizations import visualize_image_with_bounding_box


class Pipeline:

    def __init__(self, args):

        self.stages = args
        self.images = preprocess_images()

    def run(self):

        if self.stages.download_model:
            self.download_model()

        self.model = self.train_model()
        self.predictions = self.make_predictions()

        self.store_results()

    @staticmethod
    def download_model():
        urlretrieve(model_params.model_url_path, os.path.join('train_model', 'saved_model', model_params.model_name))

    def train_model(self):
        model = TrainModel()
        model.fit(model_params)
        return model

    def make_predictions(self):
        predictions = []
        for img_path, rectangles, numbers_to_predict in self.images:
            prediction = self.model.model.predict(numbers_to_predict)
            prediction = np.sort(prediction.argmax(axis=-1))
            prediction = ''.join(map(str, prediction))
            predictions.append([img_path, rectangles, prediction])
        return predictions

    def store_results(self):
        # default version - print on screen
        if not self.stages.csv and not self.stages.showpicture:
            for img in self.predictions:
                print(f'Image: {img[0]} | Prediction: {img[2]}')

        else:
            if self.stages.csv:
                self.create_csv_file()

            if self.stages.showpicture:
                self.show_pictures()

    def create_csv_file(self):
        file = open(os.path.join('test_samples', 'predicted_samples', f'Predictions_{date.today().strftime("%Y%m%d")}.csv'), 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerow(['Image', 'Prediction'])
            write.writerows(self.predictions)

    def show_pictures(self):
        for i in range(len(self.predictions)):
            prediction = self.predictions[i]
            visualize_image_with_bounding_box(prediction)

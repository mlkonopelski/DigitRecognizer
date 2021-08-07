from urllib.request import urlretrieve
from config import model_params
from train_model.model import TrainModel
from preprocess_images.preprocess import preprocess_images
import os
import numpy as np
import csv
from datetime import date


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
        for img_path, numbers_to_predict in self.images:
            prediction = self.model.predict(numbers_to_predict)
            prediction = np.sort(prediction.argmax(axis=-1))
            prediction = ''.join(map(str, prediction))
            predictions.append([img_path, prediction])
        return predictions

    def store_results(self):
        # default version - print on screen
        if not self.stages.csv and not self.stages.showpicture:
            [print(f'Image: {img[0]} | Prediction: {img[1]}') for img in self.predictions]

        else:
            if self.stages.csv:
                self.create_csv_file()

            if self.stages.showpicture:
                #TODO: prepare visualization class in vosializations module
                pass

    @staticmethod
    def create_csv_file(data):
        file = open('test_images', f'Predictions_{date.today().strftime("%Y%m%d")}.csv', 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerow(['Image', 'Prediction'])
            write.writerows(data)

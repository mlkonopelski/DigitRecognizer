import csv
import logging
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
        '''
        Pipeline class will be used to run the whole analysis and predict all pictures from test_samples directory
        :arg -> argaparser with --bash like arguemets.
        '''

        self.stages = args
        self.images = preprocess_images()

    def run(self):
        '''
        Run all of the stages of pipeline
        '''

        if self.stages.download_model:
            self.download_model()

        self.model = self.train_model()
        self.predictions = self.make_predictions()

        self.store_results()

    @staticmethod
    def download_model():
        #TODO: This is not working. I need better place to host this model to be able to download it directly
        logging.warning('This is not working as model is saved in google drive in which I have problem to download directly')
        urlretrieve(model_params.model_url_path, os.path.join('train_model', 'saved_model', model_params.model_name))

    def train_model(self):
        '''
        If model was not saved this functions starting training process on mnist dataset and Xception model.
        :return: tensorflow model instance. Easy to use by calling .predict() method on it.
        '''
        model = TrainModel()
        model.fit(model_params)
        return model

    def make_predictions(self):
        '''
        Run prediction on each picture in test_samples directory
        :return: predictions which is a list of 3 objects:
                * img_path - name of picture. Used as identifier.
                * retangles - [(x, y, w, h)] - coordinates of bounding boxes around each number
                * predictions - string - sorted predicted numbers
        '''
        logging.info(f'Making predictions. In progress.')
        predictions = []
        for img_path, rectangles, numbers_to_predict in self.images:
            logging.info(f'\tMaking predictions of image: {img_path}. In progress.')
            prediction = self.model.model.predict(numbers_to_predict)
            prediction = np.sort(prediction.argmax(axis=-1))
            prediction = ''.join(map(str, prediction))
            predictions.append([img_path, rectangles, prediction])
            logging.info(f'\tMaking predictions of image: {img_path}. Success!')
        logging.info(f'Making predictions. Success!')
        return predictions

    def store_results(self):
        '''
        Store results in on the methods:
            * print on screen
            * save as csv file
            * save each prediction as original picture with bounding boxes and prediction
        '''
        # default version - print on screen
        if not self.stages.csv and not self.stages.showpicture:
            logging.info(f'Printing results of prediction. In progress.')
            for img in self.predictions:
                print(f'Image: {img[0]} | Prediction: {img[2]}')
            logging.info(f'Printing results of prediction. Success!')

        else:
            if self.stages.csv:
                self.create_csv_file()

            if self.stages.showpicture:
                self.show_pictures()

    def create_csv_file(self):
        logging.info(f'Creating CSV file. In progress.')
        file = open(os.path.join('test_samples', 'predicted_samples', f'Predictions_{date.today().strftime("%Y%m%d")}.csv'), 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerow(['Image', 'Prediction'])
            write.writerows(self.predictions)
        logging.info(f'Creating CSV file. Success!')

    def show_pictures(self):
        logging.info(f'Saving predictions as .png in test_samples/predicted_samples. In progress.')
        for i in range(len(self.predictions)):
            prediction = self.predictions[i]
            visualize_image_with_bounding_box(prediction)
        logging.info(f'Saving predictions as .png in test_samples/predicted_samples. Success!')

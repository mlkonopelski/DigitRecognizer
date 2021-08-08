import os

import numpy as np
from tensorflow.keras.models import load_model

from config import model_params
from preprocess_images.preprocess import PreprocessImage
from utils.utility_functions import timeit
from train_model.model import load_saved_model


@timeit
def predict_picture(img, model):
    '''
    Simple function which uses already trained model to predict numbers on pictures
    :params:
        img: path to picture to be scored
        model: tensorflow model isntance already fitted. Best to use as below by loading saved model.
    :return string - ordered numbers on canvas
    '''
    pred = model.predict(PreprocessImage(img).numbers)
    pred = np.sort(pred.argmax(axis=-1))
    pred = ''.join(map(str, pred))
    return pred


if __name__ == '__main__':

    model = load_saved_model(model_params)
    images = [os.path.join('test_samples', fn) for fn in os.listdir('test_samples') if any(fn.endswith(ext) for ext in ['jpg', 'jpeg', 'bmp', 'png'])]

    for img in images:
        prediction = predict_picture(img, model)
        print(img, ': ', prediction)

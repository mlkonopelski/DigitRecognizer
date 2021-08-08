import os

import numpy as np
from tensorflow.keras.models import load_model

from config import model_params
from preprocess_images.preprocess import PreprocessImage
from utils.utility_functions import timeit


@timeit
def predict_picture(img, model):
    pred = model.predict(PreprocessImage(img).numbers)
    pred = np.sort(pred.argmax(axis=-1))
    pred = ''.join(map(str, pred))
    return pred


if __name__ == '__main__':

    model = load_model(os.path.join('train_model', 'saved_model', model_params.model_name))
    images = [os.path.join('test_samples', fn) for fn in os.listdir('test_samples') if any(fn.endswith(ext) for ext in ['jpg', 'jpeg', 'bmp', 'png'])]

    for img in images:
        prediction = predict_picture(img, model)
        print(img, ': ', prediction)

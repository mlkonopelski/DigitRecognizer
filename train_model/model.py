import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, metrics, applications
import tensorflow_datasets as tfds
from config import model_params
import warnings


def model_saved():
    if os.path.isfile(os.path.join('train_model', 'saved_model', model_params.model_name)):
        return True
    else:
        for fname in os.listdir('saved_model'):
            if fname.endswith('.h5'):
                warnings.warn("The model found have wrong name. It's not guaranteed to work.")
                return True
        else:
            return False


def preprocess_image(image, label):
  resized_image = tf.image.resize(image, size=[299, 299])
  rgb_image = tf.image.grayscale_to_rgb(resized_image, name=None)
  final_image = applications.xception.preprocess_input(rgb_image)
  return final_image, label


def prepare_dataset():
    (train, valid), ds_info = tfds.load('mnist',
                                              split=['train[:90%]', 'train[90%:]'],
                                              as_supervised=True,
                                              batch_size=32,
                                              with_info=True)
    train = train.map(preprocess_image)
    valid = valid.map(preprocess_image)

    return train, valid


def train_model(dataset, parameters):
    train, valid = dataset
    base_model = applications.Xception(include_top=False,
                                       weights="imagenet",
                                       classifier_activation="softmax")
    avg = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(10, activation='softmax')(avg)
    model = keras.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01),
                  metrics=['accuracy'])

    history = model.fit(train, epochs=10, validation_data=valid)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001),
                  metrics=['accuracy'])

    history = model.fit(train, epochs=3, validation_data=valid)

    return model


class TrainModel:

    @classmethod
    def fit(self, parameters):
        if model_saved():
            self.model = tf.keras.models.load_model(os.path.join('train_model', 'saved_model', model_params.model_name))

        else:
            self.dataset = prepare_dataset()
            self.model = train_model(self.dataset, parameters)
            self.model.save(os.path.join('saved_model', model_params.model_name))
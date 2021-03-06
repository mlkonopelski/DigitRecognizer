import logging
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.models import model_from_json


def model_saved(parameters):
    if os.path.isfile(os.path.join('train_model', 'saved_model', parameters.model_name)):
        return True
    else:
        for fname in os.listdir(os.path.join('train_model', 'saved_model')):
            if fname.endswith('.h5'):
                return True
        else:
            logging.info(f'Saved model has NOT been found and IT WILL BE TRAINED.')
            return False


def load_saved_model(parameters):
    '''
    If whole model in .h5 is available load it.
    Other wise build model from json and weigths. Then save compiled model as h5.
    :return tensorflow model instance fitted.
    '''
    if os.path.isfile(os.path.join('train_model', 'saved_model', parameters.model_name)):
        logging.info(f'Saved model has been found and it will be load.')
        model = tf.keras.models.load_model(os.path.join('train_model', 'saved_model', parameters.model_name))
        return model
    else:
        logging.info("Compiling model from json and weights. In progress.")
        with open(os.path.join('train_model', 'saved_model', f'{parameters.model_name[:-3]}.json'), 'r') as json_file:
            loaded_model_json = json_file.read()

        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join('train_model', 'saved_model', f'{parameters.model_name[:-3]}_weights.h5'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(learning_rate=parameters.all_layers_training_lr,
                                                     momentum=parameters.all_layers_training_momentum,
                                                     decay=parameters.all_layers_training_decay),
                      metrics=['accuracy'])
        model.save(os.path.join('train_model', 'saved_model', parameters.model_name))
        logging.info("Compiling model from json and weights. Success!")
        return model


def preprocess_image(image, label):
  resized_image = tf.image.resize(image, size=[299, 299])
  rgb_image = tf.image.grayscale_to_rgb(resized_image, name=None)
  final_image = applications.xception.preprocess_input(rgb_image)
  return final_image, label


def prepare_dataset(parameters):
    logging.info('Downloading MNIST dataset. In progress.')
    (train, valid), ds_info = tfds.load('mnist',
                                              split=[f'train[:{parameters.train_subset}%]', f'train[{parameters.train_subset}%:]'],
                                              as_supervised=True,
                                              batch_size=32,
                                              with_info=True)
    logging.info('Downloading MNIST dataset. Success!')
    if parameters.local_training:
        logging.info('SAMPLING: For local training. 100 train samples and 15 validation.')
        train = train.take(100)
        valid = valid.take(15)

    train = train.map(preprocess_image)
    valid = valid.map(preprocess_image)

    logging.info('Downloading MNIST dataset. Success!')
    return train, valid


def train_model(dataset, parameters):
    train, valid = dataset

    logging.info('Training model. In progress.')
    logging.info('\tCompiling model with top layers frozen. In progress.')
    base_model = applications.Xception(include_top=False,
                                       weights="imagenet",
                                       classifier_activation="softmax")
    avg = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(10, activation='softmax')(avg)
    model = keras.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=parameters.top_layers_training_lr,
                                                 momentum=parameters.top_layers_training_momentum,
                                                 decay=parameters.top_layers_training_decay),
                  metrics=['accuracy'])
    logging.info('\tCompiling model with top layers frozen. Success!')

    logging.info('\tFitting model with top layers frozen. In progress.')
    model.fit(train, epochs=parameters.top_layers_training_epochs, validation_data=valid)
    logging.info('\tFitting model with top layers frozen. Success!')

    logging.info('\tCompiling model with all layers unfrozen. In progress.')
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=parameters.all_layers_training_lr,
                                                 momentum=parameters.all_layers_training_momentum,
                                                 decay=parameters.all_layers_training_decay),
                  metrics=['accuracy'])
    logging.info('\tCompiling model with all layers unfrozen. Success!')

    logging.info(
        f'\tFitting model with all layers unfrozen and for {parameters.all_layers_training_epochs} epochs. In progress.')
    model.fit(train, epochs=parameters.all_layers_training_epochs, validation_data=valid)
    logging.info(
        f'\tFitting model with all layers unfrozen. Success!')

    logging.info('Training model. Success!')
    return model


class TrainModel:

    @classmethod
    def fit(self, parameters):
        if model_saved(parameters):
            self.model = load_saved_model(parameters)

        else:
            self.dataset = prepare_dataset(parameters)
            self.model = train_model(self.dataset, parameters)
            self.model.save(os.path.join('train_model', 'saved_model', parameters.model_name))

import os
import os
import warnings

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, applications


def model_saved(parameters):
    if os.path.isfile(os.path.join('train_model', 'saved_model', parameters.model_name)):
        return True
    else:
        for fname in os.listdir(os.path.join('train_model', 'saved_model')):
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


def prepare_dataset(parameters):
    (train, valid), ds_info = tfds.load('mnist',
                                              split=[f'train[:{parameters.train_subset}%]', f'train[{parameters.train_subset}%:]'],
                                              as_supervised=True,
                                              batch_size=32,
                                              with_info=True)
    if parameters.local_training:
        train = train.take(100)
        valid = valid.take(15)

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
                  optimizer=keras.optimizers.SGD(learning_rate=parameters.top_layers_training_lr,
                                                 momentum=parameters.top_layers_training_momentum,
                                                 decay=parameters.top_layers_training_decay),
                  metrics=['accuracy'])

    model.fit(train, epochs=parameters.top_layers_training_epochs, validation_data=valid)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=parameters.all_layers_training_lr,
                                                 momentum=parameters.all_layers_training_momentum,
                                                 decay=parameters.all_layers_training_decay),
                  metrics=['accuracy'])

    model.fit(train, epochs=parameters.all_layers_training_epochs, validation_data=valid)

    return model


class TrainModel:

    @classmethod
    def fit(self, parameters):
        if model_saved(parameters):
            self.model = tf.keras.models.load_model(os.path.join('train_model', 'saved_model', parameters.model_name))

        else:
            self.dataset = prepare_dataset(parameters)
            self.model = train_model(self.dataset, parameters)
            self.model.save(os.path.join('saved_model', parameters))

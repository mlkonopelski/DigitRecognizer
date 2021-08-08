from dataclasses import dataclass


@dataclass()
class ModelParams:
    def __init__(self):
        self.model_name = 'Xception_transfered_to_mnist.h5'
        self.model_url_path = 'https://drive.google.com/uc?export=download&confirm=sDCG&id=1ZZ-TJDjGL8jRgqXOeJGECEt5-1udSTuH'
        self.train_subset = 90  # 90% training & 10% validation
        self.local_training = False  # My computer is too slow for this model so for local testing I use 100 traning samples
        self.top_layers_training_epochs = 2
        self.top_layers_training_lr = 0.2
        self.top_layers_training_momentum = 0.9
        self.top_layers_training_decay = 0.01
        self.all_layers_training_epochs = 1
        self.all_layers_training_lr = 0.01
        self.all_layers_training_momentum = 0.9
        self.all_layers_training_decay = 0.001


@dataclass()
class PreprocessParams:
    def __init__(self):
        self.canny_treshold = 100
        self.group_rectangles_eps = 0.05


model_params = ModelParams()
preprocess_params = PreprocessParams()

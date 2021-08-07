from dataclasses import dataclass


@dataclass()
class ModelParams:
    def __init__(self):
        self.model_name = 'Xception_transfered_to_mnist.h5'
        self.model_url_path = 'https://drive.google.com/file/d/1ZZ-TJDjGL8jRgqXOeJGECEt5-1udSTuH/view?usp=sharing'
        self.epochs = 10



@dataclass()
class PreprocessParams:
    def __init__(self):
        self.canny_treshold = 100


model_params = ModelParams()
preprocess_params = PreprocessParams()


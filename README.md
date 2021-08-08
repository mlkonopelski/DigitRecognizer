# DigitRecognizer
DigitRecognizer is a model which recognizes hand written numbers on picture.

**Installation and usage**  
After cloning this repository (size is ~160Mb) please install all packages in new virtual environment (recommended Python 3.8.5):

```pip install -r requierements.txt```

The easiest way to run this script is recognizing pictures saved in test_samples folder. 
You can add your examples there as well. The only requierements are: 
* one of supported file types: .jpg, .png, .bmp
* size of images: 140 x 140 
* digits should not everlap each other on canvas

**Important!**  
The classification model (Xception transfered from imagenet) training takes roughly 3.5 hours on GPU (Google Colab) 
but the accuracy is fairly good as it scored 253/7027 on Kaggle when I built it 6 months ago: 
https://www.kaggle.com/mateuszkonopelski/competitions?tab=active
But if you want to retrain it just remove it from train_model/saved_model directory and the process will start from training.


**Modes**  
To run the script in default mode (print name of the image file and digits in order):  
```python main.py```   

If you are using linux distro or wsl you can use bash arguments to adjust the execution. It can be run in following modes:  
```python main.py --help```
1. Save the predictions in csv file: ```--csv```
1. [unstable] Open each picture with bounding boxes and prediction in title: ```--showpicture```. This is unstable
version because it requires additional installations on OS site. For convenience we save each of test samples in this 
format in dir: test_samples/predicted_samples.
![showpicture](https://github.com/mlkonopelski/DigitRecognizer/blob/main/utils/DigitRecognizerExample.PNG?raw=true "Optional Title")
1. As I don't know how this will be used, I also created utility function ```predict_picture``` in predict_picture script which
takes only one argument ```img``` which is picture as array or path to image in one of suported formats and 
return ordered string.

Example:   
```python main.py --csv```

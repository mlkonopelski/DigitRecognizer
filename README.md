# DigitRecognizer
DigitRecognizer is a model which recognizes hand written numbers on picture.

**Installation and usage**  
After cloning this repository please install all packages in new virtual environment (recommended Python 3.8.5):

```pip install -r requierements.txt```

The easiest way to run this script is recognizing pictures saved in test_samples folder. 
You can add your examples there as well. The only requierements are: 
* one of supported file types: .jpg, .png, 
* size of images: 140 x 140 
* digits should not everlap each other on canvas

**Important!**  
During the first run, the classification model will be trained on mnist dataset. 
I didn't included trained model because of size (~150Mb) in repo. 
But it's possible to download it from my drive by using arg ```--download_model```
The next runs, it should read the saved model. 
Otherwise, model training takes roughly 2 hours on GPU but the accuracy is fairly good as it scored 253/7027 on Kaggle when I built it 6 months ago: https://www.kaggle.com/mateuszkonopelski/competitions?tab=active



**Modes**  
To run the script in default mode (print name of the file and digits in order):  
```python main.py```   

But using bash arguments we can adjust the execution. It can be run in following modes:  
```python main.py --help```
1. Save the predictions in csv file: ```--csv```
1. Open each picture with prediction in title: ```--showpicture```
1. [unstable] Open each picture with contours and bounding boxes ```--adv_showpicture``` 

Example:   
```python main.py --showpicture```


**Test samples**  
If you don't have your own testing samples, I created a script to generate them from mnist dataset. 
To use it, before running ```main.py``` run ```python test_samples/generate_test_samples.py``` with 
number of samples as argument e.g. ```--samples=100```
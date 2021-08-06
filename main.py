import cv2 as cv
import numpy  as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_numbers(img_path, model_path):
    src = cv.imread(cv.samples.findFile(img_path))
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    thresh = 100
    canny_output = cv.Canny(src_gray, thresh, thresh * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        x, y, w, h = cv.boundingRect(hull)
        rectangles.append([x, y, w, h])

    rectangles = cv.groupRectangles(rectangles, 1)

    resized_numbers = []
    for i in range(len(rectangles[0])):
        x, y, w, h = rectangles[0][i]
        croped = src[y:y + h, x:x + w]
        resized = cv.resize(croped, (299, 299))
        preprocessed = tf.keras.applications.xception.preprocess_input(resized).reshape(299, 299, 3)
        resized_numbers.append(preprocessed)

    resized_numbers = np.stack(resized_numbers, axis=0)

    test_dataset = tf.data.Dataset.from_tensor_slices(resized_numbers).batch(32)

    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(test_dataset)
    predictions = np.sort(y_pred.argmax(axis=-1))

    return predictions


def visualize(img_path, predictions):
    image = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f'Predicted values = {predictions}')
    plt.show()


if __name__ == '__main__':

    IMAGE_PATH = 'test_samples/Example1.png'
    MODEL_PATH = 'train_model/saved_model/Xception_transfered_to_mnist.h5'

    predictions = predict_numbers(IMAGE_PATH, MODEL_PATH)

    visualize(IMAGE_PATH, predictions)

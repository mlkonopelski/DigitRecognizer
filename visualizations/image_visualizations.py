import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def visualize(img_path, predictions):
    image = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f'Predicted values = {predictions}')
    plt.show()


def visualize_image(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    plt.show()


def visualize_image_with_bounding_box(prediction_info):
    img_path, rectangle, prediction = prediction_info
    image = plt.imread(os.path.join('test_samples', img_path))
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    for rect in rectangle:
        x,y,w,h = rect
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f'Image: {img_path} | Prediction: {prediction}')
    fig.savefig(os.path.join('test_samples', 'predicted_samples', f'{img_path}'))
    plt.show()


if __name__ == '__main__':
    img_path = 'Example1.png'
    rectangle = [[[93, 80, 22, 43],
                [35, 71, 26, 36],
                [76, 35, 25, 33],
                [15, 15, 24, 41]],
                [2],
                [2]]

    prediction = '1348'
    prediction_info = img_path, rectangle, prediction
    visualize_image_with_bounding_box(prediction_info)

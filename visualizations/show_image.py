import matplotlib.pyplot as plt

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

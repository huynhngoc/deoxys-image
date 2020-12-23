from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from deoxys_image import ImageAugmentation


def load_image(path, target_size=(224, 224), normalize=True):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    # x = np.expand_dims(x, axis=0)
    if normalize:
        x = (x/255).clip(0, 1)
    else:
        x = x.astype(int)

    return x


if __name__ == "__main__":
    images = np.array([load_image('imgs/cat.jpg') for _ in range(48)])
    targets = np.array([load_image('imgs/cat.jpg', normalize=False)
                        for _ in range(48)])
    aug = ImageAugmentation(
        rank=3, rotation_range=15, rotation_axis=2,
        zoom_range=(0.7, 1.3), shift_range=(20, 10), flip_axis=1,
        brightness_range=(0.7, 1.3), contrast_range=(0.5, 1.5),
        noise_variance=(0.5, 0.7), blur_range=(1, 4))

    res = aug.transform(images, targets)

    i = 1
    row, col = 8, 12
    for img, target in zip(*res):
        plt.subplot(row, col, i)
        plt.imshow(img)
        plt.axis('off')
        i += 1
        plt.subplot(row, col, i)
        plt.imshow(target)
        plt.axis('off')
        i += 1
    plt.show()

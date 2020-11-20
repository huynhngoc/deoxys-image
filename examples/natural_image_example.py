from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from deoxys_image import apply_affine_transform
from deoxys_image import apply_random_gaussian_noise, apply_gaussian_noise
from deoxys_image import apply_random_brightness, apply_random_contrast
from deoxys_image import apply_random_gaussian_blur


def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    # x = np.expand_dims(x, axis=0)
    x = (x/255).clip(0, 1)

    return x


def affine_transform():
    img = load_image('imgs/cat.jpg')
    nrow = 6
    ncol = 10

    rotate = [0, -30, 15]
    zoom = [1, 0.5, 0.8, 1.2, 2]
    shift = [(0, 0), (0, 50), (70, 0), (70, 50)]

    mode = 'constant'
    cval = 0.0

    s = 0
    i = 1
    pos = 1
    for row, r in enumerate(rotate):
        for col, z in enumerate(zoom):
            transformed = apply_affine_transform(img,
                                                 theta=r,
                                                 zoom_factor=z,
                                                 shift=shift[s],
                                                 mode=mode, cval=cval)
            plt.subplot(nrow, ncol, pos+row*10+col)
            plt.axis('off')
            plt.imshow(transformed)
            plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

    s = 1
    i = 1
    pos = 6
    for row, r in enumerate(rotate):
        for col, z in enumerate(zoom):
            transformed = apply_affine_transform(img,
                                                 theta=r,
                                                 zoom_factor=z,
                                                 shift=shift[s],
                                                 mode=mode, cval=cval)
            plt.subplot(nrow, ncol, pos+row*10+col)
            plt.axis('off')
            plt.imshow(transformed)
            plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

    s = 2
    i = 1
    pos = 31
    for row, r in enumerate(rotate):
        for col, z in enumerate(zoom):
            transformed = apply_affine_transform(img,
                                                 theta=r,
                                                 zoom_factor=z,
                                                 shift=shift[s],
                                                 mode=mode, cval=cval)
            plt.subplot(nrow, ncol, pos+row*10+col)
            plt.axis('off')
            plt.imshow(transformed)
            plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

    s = 3
    i = 1
    pos = 36
    for row, r in enumerate(rotate):
        for col, z in enumerate(zoom):
            transformed = apply_affine_transform(img,
                                                 theta=r,
                                                 zoom_factor=z,
                                                 shift=shift[s],
                                                 mode=mode, cval=cval)

            plt.subplot(nrow, ncol, pos+row*10+col)
            plt.axis('off')
            plt.imshow(transformed)
            plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

    plt.show()


def noise_transform():
    images = np.array([load_image('imgs/cat.jpg'),
                       load_image('imgs/cat_dog.jpg'),
                       load_image('imgs/elephant.jpg'),
                       load_image('imgs/husky.jpg')])

    nrow = 5
    ncol = 4
    noise_var = [0.05, 0.1, 0.25, 0.5]

    pos = 1

    for img in images:
        plt.subplot(nrow, ncol, pos)
        plt.axis('off')
        plt.imshow(img)
        # plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

        pos += 1

    for var in noise_var:
        noised = apply_gaussian_noise(images, var)

        for img in noised:
            plt.subplot(nrow, ncol, pos)
            plt.axis('off')
            plt.imshow(img)
            plt.title(f'Var={var}', fontsize=6)

            pos += 1

    plt.show()


def random_noise_transform():
    images = np.array([load_image('imgs/cat.jpg'),
                       load_image('imgs/cat_dog.jpg'),
                       load_image('imgs/elephant.jpg'),
                       load_image('imgs/husky.jpg')])

    images = np.concatenate([images, images])

    nrow = 5
    ncol = 8
    random_noise_var = [(0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0, 0.5)]

    pos = 1

    for img in images:
        plt.subplot(nrow, ncol, pos)
        plt.axis('off')
        plt.imshow(img)
        # plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

        pos += 1

    for var in random_noise_var:
        noised = apply_random_gaussian_noise(images, var)

        for img in noised:
            plt.subplot(nrow, ncol, pos)
            plt.axis('off')
            plt.imshow(img)
            plt.title(f'Var={var}', fontsize=6)

            pos += 1

    plt.show()


def brightness_transform():
    images = np.array([load_image('imgs/cat.jpg'),
                       load_image('imgs/cat_dog.jpg'),
                       load_image('imgs/elephant.jpg'),
                       load_image('imgs/husky.jpg')])

    nrow = 8
    ncol = 4
    brightness_range = [(1.7, 1.7), (1.5, 1.5), (0.3, 0.3), (0.7, 0.7),
                        (1.3, 1.5), (0.5, 0.7), (0.7, 1.3)]

    pos = 1

    for img in images:
        plt.subplot(nrow, ncol, pos)
        plt.axis('off')
        plt.imshow(img)
        # plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

        pos += 1

    for low, high in brightness_range:
        transformed = apply_random_brightness(images, low, high)

        for img in transformed:
            plt.subplot(nrow, ncol, pos)
            plt.axis('off')
            plt.imshow(img)
            plt.title(f'Low={low} high={high}', fontsize=6)

            pos += 1

    plt.show()


def contranst_transform():
    images = np.array([load_image('imgs/cat.jpg'),
                       load_image('imgs/cat_dog.jpg'),
                       load_image('imgs/elephant.jpg'),
                       load_image('imgs/husky.jpg')])

    nrow = 8
    ncol = 4
    contranst_range = [(2, 2), (1.5, 1.5), (0.3, 0.3), (0.7, 0.7),
                       (1.3, 1.5), (0.5, 0.7), (0.7, 1.3)]

    pos = 1

    for img in images:
        plt.subplot(nrow, ncol, pos)
        plt.axis('off')
        plt.imshow(img)
        # plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

        pos += 1

    for low, high in contranst_range:
        transformed = apply_random_contrast(images, low, high)

        for img in transformed:
            plt.subplot(nrow, ncol, pos)
            plt.axis('off')
            plt.imshow(img)
            plt.title(f'Low={low} high={high}', fontsize=6)

            pos += 1

    plt.show()


def blur_transform():
    images = np.array([load_image('imgs/cat.jpg'),
                       load_image('imgs/cat_dog.jpg'),
                       load_image('imgs/elephant.jpg'),
                       load_image('imgs/husky.jpg')])

    # images = np.concatenate([images, images])

    # plt.figure(figsize=(80, 80))

    nrow = 8
    ncol = 4
    contranst_range = [(4, 4), (3, 3), (2, 2), (1.5, 1.5), (1, 1),
                       (0.5, 0.5), (0.5, 1.5)]

    pos = 1

    for img in images:
        plt.subplot(nrow, ncol, pos)
        plt.axis('off')
        plt.imshow(img)
        # plt.title(f'theta={r}, z={z},s={shift[s]}', fontsize=6)

        pos += 1

    for low, high in contranst_range:
        transformed = apply_random_gaussian_blur(images, low, high)

        for img in transformed:
            plt.subplot(nrow, ncol, pos)
            plt.axis('off')
            plt.imshow(img)
            plt.title(f'Low={low} high={high}', fontsize=6)

            pos += 1

    plt.show()
    # plt.savefig('../../blur.png')


if __name__ == '__main__':
    # affine_transform()
    # noise_transform()
    # random_noise_transform()
    # brightness_transform()
    contranst_transform()
    blur_transform()

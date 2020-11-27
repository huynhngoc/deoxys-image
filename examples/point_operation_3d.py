import matplotlib.pyplot as plt
import h5py
from deoxys_image import normalize, gaussian_noise, change_brightness
from deoxys_image import change_contrast, gaussian_blur


def load_images(index=0):
    with h5py.File(
            '../../hn_perf/3d_unet_32/prediction/prediction.030.h5', 'r') as f:
        image = f['x'][index]
        target = f['y'][index]

    return normalize(image), target


if __name__ == "__main__":
    brightness = 1
    contrast = 1
    noise = 0.1
    sigma = 0

    image, target = load_images()
    shape = image.shape[:-1]

    transformed = image.copy()

    if brightness != 1:
        transformed = change_brightness(transformed, brightness, channel=None)
    if contrast != 1:
        transformed = change_contrast(transformed, contrast, channel=None)
    if noise > 0:
        transformed = gaussian_noise(transformed, noise, channel=None)
    if sigma > 0:
        transformed = gaussian_blur(transformed, sigma=sigma, channel=None)

    fig, axes = plt.subplots(2, 2)

    for ax in axes.flatten():
        ax.axis('off')

    initialize = False
    mask_levels = [0.5]

    pause_time = 0.1

    while(True):
        for i in range(shape[0]):
            if not initialize:
                plt.suptitle(
                    f'Slice {i}, brightness {brightness}, contrast {contrast}, noise {noise}, signma {sigma}')

                im_ax_ct = axes[0][0].imshow(
                    image[i][..., 0], cmap='gray', vmin=0, vmax=1)
                im_ax_pet = axes[0][1].imshow(
                    image[i][..., 1], cmap='gray', vmin=0, vmax=1)
                label_ax_ct = axes[0][0].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')
                label_ax_pet = axes[0][1].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')

                transform_ax_ct = axes[1][0].imshow(
                    transformed[i][..., 0], cmap='gray', vmin=0, vmax=1)
                transform_ax_pet = axes[1][1].imshow(
                    transformed[i][..., 1], cmap='gray', vmin=0, vmax=1)
                new_label_ax_ct = axes[1][0].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')
                plt.pause(pause_time)
                initialize = True
            else:
                im_ax_ct.set_data(image[i][..., 0])
                im_ax_pet.set_data(image[i][..., 1])
                # label_ax_ct.set_data(target[i][..., 0])
                # label_ax_pet.set_data(target[i][..., 0])
                for c in label_ax_ct.collections:
                    c.remove()
                for c in label_ax_pet.collections:
                    c.remove()
                label_ax_ct = axes[0][0].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')
                label_ax_pet = axes[0][1].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')

                transform_ax_ct.set_data(transformed[i][..., 0])
                transform_ax_pet.set_data(transformed[i][..., 1])
                # new_label_ax_ct.set_data(
                #     target[i][..., 0])
                # new_label_ax_pet.set_data(
                #     target[i][..., 0])
                for c in new_label_ax_ct.collections:
                    c.remove()
                for c in new_label_ax_pet.collections:
                    c.remove()

                new_label_ax_ct = axes[1][0].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    target[i][..., 0], 1, levels=mask_levels, colors='yellow')

                plt.suptitle(
                    f'Slice {i}, brightness {brightness}, contrast {contrast}, noise {noise}, signma {sigma}')
                plt.pause(pause_time)

        if input('Press ENTER to continue...') == 'exit':
            break
    plt.show()

    fig, axes = plt.subplots(2, 2)

    for ax in axes.flatten():
        ax.axis('off')

    initialize = False
    mask_levels = [0.5]

    pause_time = 0.1

    while(True):
        for i in range(shape[0]):
            if not initialize:
                plt.suptitle(
                    f'Slice {i}, brightness {brightness}, contrast {contrast}, noise {noise}, signma {sigma}')

                im_ax_ct = axes[0][0].imshow(
                    image[:, i, :, 0], cmap='gray', vmin=0, vmax=1, origin='lower')
                im_ax_pet = axes[0][1].imshow(
                    image[:, i, :, 1], cmap='gray', vmin=0, vmax=1, origin='lower')
                label_ax_ct = axes[0][0].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
                label_ax_pet = axes[0][1].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')

                transform_ax_ct = axes[1][0].imshow(
                    transformed[:, i, :, 0], cmap='gray', vmin=0, vmax=1, origin='lower')
                transform_ax_pet = axes[1][1].imshow(
                    transformed[:, i, :, 1], cmap='gray', vmin=0, vmax=1, origin='lower')
                new_label_ax_ct = axes[1][0].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
                plt.pause(pause_time)
                initialize = True
            else:
                im_ax_ct.set_data(image[:, i, :, 0])
                im_ax_pet.set_data(image[:, i, :, 1])
                # label_ax_ct.set_data(target[:, i, :, 0])
                # label_ax_pet.set_data(target[:, i, :, 0])
                for c in label_ax_ct.collections:
                    c.remove()
                for c in label_ax_pet.collections:
                    c.remove()
                label_ax_ct = axes[0][0].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
                label_ax_pet = axes[0][1].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')

                transform_ax_ct.set_data(transformed[:, i, :, 0])
                transform_ax_pet.set_data(transformed[:, i, :, 1])
                # new_label_ax_ct.set_data(
                #     target[:, i, :, 0])
                # new_label_ax_pet.set_data(
                #     target[:, i, :, 0])
                for c in new_label_ax_ct.collections:
                    c.remove()
                for c in new_label_ax_pet.collections:
                    c.remove()

                new_label_ax_ct = axes[1][0].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    target[:, i, :, 0], 1, levels=mask_levels, colors='yellow')

                plt.suptitle(
                    f'Slice {i}, brightness {brightness}, contrast {contrast}, noise {noise}, signma {sigma}')
                plt.pause(pause_time)

        if input('Press ENTER to continue...') == 'exit':
            break
    plt.show()

    fig, axes = plt.subplots(2, 2)

    for ax in axes.flatten():
        ax.axis('off')

    initialize = False
    mask_levels = [0.5]

    pause_time = 0.1

    while(True):
        for i in range(shape[0]):
            if not initialize:
                plt.suptitle(
                    f'Slice {i}, brightness {brightness}, contrast {contrast}, noise {noise}, signma {sigma}')

                im_ax_ct = axes[0][0].imshow(
                    image[:, :, i, 0], cmap='gray', vmin=0, vmax=1, origin='lower')
                im_ax_pet = axes[0][1].imshow(
                    image[:, :, i, 1], cmap='gray', vmin=0, vmax=1, origin='lower')
                label_ax_ct = axes[0][0].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
                label_ax_pet = axes[0][1].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')

                transform_ax_ct = axes[1][0].imshow(
                    transformed[:, :, i, 0], cmap='gray', vmin=0, vmax=1, origin='lower')
                transform_ax_pet = axes[1][1].imshow(
                    transformed[:, :, i, 1], cmap='gray', vmin=0, vmax=1, origin='lower')
                new_label_ax_ct = axes[1][0].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
                plt.pause(pause_time)
                initialize = True
            else:
                im_ax_ct.set_data(image[:, :, i, 0])
                im_ax_pet.set_data(image[:, :, i, 1])
                # label_ax_ct.set_data(target[:, :, i, 0])
                # label_ax_pet.set_data(target[:, :, i, 0])
                for c in label_ax_ct.collections:
                    c.remove()
                for c in label_ax_pet.collections:
                    c.remove()
                label_ax_ct = axes[0][0].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
                label_ax_pet = axes[0][1].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')

                transform_ax_ct.set_data(transformed[:, :, i, 0])
                transform_ax_pet.set_data(transformed[:, :, i, 1])
                # new_label_ax_ct.set_data(
                #     target[:, :, i, 0])
                # new_label_ax_pet.set_data(
                #     target[:, :, i, 0])
                for c in new_label_ax_ct.collections:
                    c.remove()
                for c in new_label_ax_pet.collections:
                    c.remove()

                new_label_ax_ct = axes[1][0].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    target[:, :, i, 0], 1, levels=mask_levels, colors='yellow')

                plt.suptitle(
                    f'Slice {i}, brightness {brightness}, contrast {contrast}, noise {noise}, signma {sigma}')
                plt.pause(pause_time)

        if input('Press ENTER to continue...') == 'exit':
            break
    plt.show()

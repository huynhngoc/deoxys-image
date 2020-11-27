import matplotlib.pyplot as plt
import h5py
from deoxys_image import apply_affine_transform, normalize


def load_images(index=0):
    with h5py.File(
            '../../hn_perf/3d_unet_32/prediction/prediction.030.h5', 'r') as f:
        image = f['x'][index][:128, 32:160, -128:]
        target = f['y'][index][:128, 32:160, -128:]

    return normalize(image), target


if __name__ == "__main__":
    theta = 30
    zoom = 1
    rotation_axis = 0
    shift = (0, 0, 0)

    image, target = load_images()
    shape = image.shape[:-1]

    transformed = apply_affine_transform(image, mode='constant',
                                         rotation_axis=rotation_axis,
                                         theta=theta, zoom_factor=zoom,
                                         shift=shift).clip(0, 1)

    transformed_label = (apply_affine_transform(target, mode='constant',
                                                rotation_axis=rotation_axis,
                                                theta=theta, zoom_factor=zoom,
                                                shift=shift).clip(0, 1) > 0.5).astype(int)

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
                    f'Slice {i}, Theta {theta}, zoom {zoom}, shift {shift}')

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
                    transformed_label[i][..., 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    transformed_label[i][..., 0], 1, levels=mask_levels, colors='yellow')
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
                #     transformed_label[i][..., 0])
                # new_label_ax_pet.set_data(
                #     transformed_label[i][..., 0])
                for c in new_label_ax_ct.collections:
                    c.remove()
                for c in new_label_ax_pet.collections:
                    c.remove()

                new_label_ax_ct = axes[1][0].contour(
                    transformed_label[i][..., 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    transformed_label[i][..., 0], 1, levels=mask_levels, colors='yellow')

                plt.suptitle(
                    f'Slice {i}, Theta {theta}, zoom {zoom}, shift {shift}')
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
                    f'Slice {i}, Theta {theta}, zoom {zoom}, shift {shift}')

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
                    transformed_label[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    transformed_label[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
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
                #     transformed_label[:, i, :, 0])
                # new_label_ax_pet.set_data(
                #     transformed_label[:, i, :, 0])
                for c in new_label_ax_ct.collections:
                    c.remove()
                for c in new_label_ax_pet.collections:
                    c.remove()

                new_label_ax_ct = axes[1][0].contour(
                    transformed_label[:, i, :, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    transformed_label[:, i, :, 0], 1, levels=mask_levels, colors='yellow')

                plt.suptitle(
                    f'Slice {i}, Theta {theta}, zoom {zoom}, shift {shift}')
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
                    f'Slice {i}, Theta {theta}, zoom {zoom}, shift {shift}')

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
                    transformed_label[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    transformed_label[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
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
                #     transformed_label[:, :, i, 0])
                # new_label_ax_pet.set_data(
                #     transformed_label[:, :, i, 0])
                for c in new_label_ax_ct.collections:
                    c.remove()
                for c in new_label_ax_pet.collections:
                    c.remove()

                new_label_ax_ct = axes[1][0].contour(
                    transformed_label[:, :, i, 0], 1, levels=mask_levels, colors='yellow')
                new_label_ax_pet = axes[1][1].contour(
                    transformed_label[:, :, i, 0], 1, levels=mask_levels, colors='yellow')

                plt.suptitle(
                    f'Slice {i}, Theta {theta}, zoom {zoom}, shift {shift}')
                plt.pause(pause_time)

        if input('Press ENTER to continue...') == 'exit':
            break
    plt.show()

import matplotlib.pyplot as plt
import h5py
from deoxys_image import normalize, get_patch_indice, get_patches


def load_images(index=2):
    with h5py.File(
            '../../hn_perf/3d_unet_32/prediction/prediction.030.h5', 'r') as f:
        images = f['x'][:index]
        targets = f['y'][:index]

    return normalize(images), targets


if __name__ == "__main__":
    images, targets = load_images()
    patch_size = (128, 128, 128)
    overlap = 0.5
    stratified = True

    batch_size = 8
    drop_fraction = 0.1
    check_drop_channel = 0

    indice = get_patch_indice(images.shape[1:-1], patch_size, overlap)
    # print(indice, len(indice))

    patches, labels = get_patches(images, targets, indice, patch_size,
                                  stratified=stratified,
                                  batch_size=batch_size,
                                  drop_fraction=drop_fraction,
                                  check_drop_channel=check_drop_channel)

    print(patches.shape)

    fig, ax = plt.subplots(1, 1)
    ax.axis('off')

    initialize = False
    mask_levels = [0.5]

    pause_time = 1e-5

    for p_i, patch in enumerate(patches):
        # if p_i < 100:
        #     continue

        for i in range(patch_size[0]):
            if not initialize:
                plt.suptitle(
                    f'Slice {i}, patch {p_i}')

                im_ax_ct = ax.imshow(
                    patch[i][..., 0], cmap='gray', )  # vmin=0, vmax=1)
                im_ax_pet = ax.imshow(
                    patch[i][..., 1], cmap='magma', vmin=0, vmax=1, alpha=0.5)
                label_ax_ct = ax.contour(
                    labels[p_i][i][..., 0], 1, levels=mask_levels,
                    colors='yellow')

                plt.pause(pause_time)
                initialize = True
            else:
                im_ax_ct.set_data(patch[i][..., 0])
                im_ax_pet.set_data(patch[i][..., 1])

                for c in label_ax_ct.collections:
                    c.remove()

                label_ax_ct = ax.contour(
                    labels[p_i][i][..., 0], 1, levels=mask_levels,
                    colors='yellow')

                plt.suptitle(
                    f'Slice {i}, patch {p_i}')

                plt.pause(pause_time)

        if input('Press ENTER to continue...') == 'exit':
            break
    plt.show()

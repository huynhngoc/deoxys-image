import matplotlib.pyplot as plt
import h5py
from deoxys_image import apply_affine_transform


if __name__ == '__main__':
    with h5py.File(
            '../../hn_perf/3d_unet_32/prediction/prediction.030.h5', 'r') as f:
        image = f['x'][0]
        target = f['y'][0]
        pred = f['predicted'][0]

    # img_slice = image[:, 100]
    img_slice = image[0]

    nrow = 5
    ncol = 8
    i = 0

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(img_slice[..., 0])
    plt.title('Original')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(img_slice[..., 1])

    transformed = apply_affine_transform(img_slice, theta=-16, rotation_axis=2)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate -16')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=30, rotation_axis=2,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate +30')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=0, rotation_axis=2, shift=(50, 0),
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Translate axis_0 50')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=0.5, rotation_axis=2)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Zoom 0.5')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=0.5, theta=-16, rotation_axis=2)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate -16, Zoom 0.5')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=30, rotation_axis=2, zoom_factor=0.5,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate +30, Zoom 0.5')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=0, rotation_axis=2, shift=(50, 0), zoom_factor=0.5,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Translate axis_0 50, Zoom 0.5')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=2, rotation_axis=2)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Zoom 2')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=2, theta=-16, rotation_axis=2)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate -16, Zoom 2')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=30, rotation_axis=2, zoom_factor=2,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate +30, Zoom 2')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=0, rotation_axis=2, shift=(50, 0), zoom_factor=2,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Translate axis_0 50, Zoom 2')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=1.2, rotation_axis=2, shift=(0, 70))

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Zoom 1.2, Translate axis_1 70')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=1.2, theta=-16, rotation_axis=2,  shift=(0, 70))

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate -16, Zoom 1.2, Translate axis_1 70')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=30, rotation_axis=2, shift=(0, 70), zoom_factor=1.2,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate +30, Zoom 1.2, Translate axis_1 70')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=0, rotation_axis=2, shift=(50, 70), zoom_factor=1.2,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Translate (50, 70), Zoom 1.2')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=0.8, rotation_axis=2, shift=(0, -70))

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Zoom 0.8, Translate axis_1 -70')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, zoom_factor=0.8, theta=-16,
        rotation_axis=2,  shift=(0, -70))

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate -16, Zoom 0.8, Translate axis_1 -70')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=30, rotation_axis=2, shift=(0, -70), zoom_factor=0.8,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Rotate +30, Zoom 0.8, Translate axis_1 -70')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    transformed = apply_affine_transform(
        img_slice, theta=30, rotation_axis=2, shift=(50, -70), zoom_factor=0.8,
        mode='constant', cval=5)

    transformed[..., 0] = transformed[..., 0].clip(-100, 100)
    transformed[..., 1] = transformed[..., 1].clip(0, img_slice[..., 1].max())

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 0])
    plt.title('Translate (50, -70), Zoom 0.8, rotate +30')

    i += 1
    plt.subplot(nrow, ncol, i)
    plt.axis('off')
    plt.imshow(transformed[..., 1])

    plt.show()

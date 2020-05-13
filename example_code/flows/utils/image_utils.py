"""
Source: https://github.com/ISosnovik/few/blob/master/few/image.py
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def viz_array_grid(array, rows, cols, padding=0, channels_last=False, normalize=False, **kwargs):
    # normalization
    '''
    Args:
        array: (N_images, N_channels, H, W) or (N_images, H, W, N_channels)
        rows, cols: rows and columns of the plot. rows * cols == array.shape[0]
        padding: padding between cells of plot
        channels_last: for Tensorflow = True, for PyTorch = False
        normalize: `False`, `mean_std`, or `min_max`
    Kwargs:
        if normalize == 'mean_std':
            mean: mean of the distribution. Default 0.5
            std: std of the distribution. Default 0.5
        if normalize == 'min_max':
            min: min of the distribution. Default array.min()
            max: max if the distribution. Default array.max()
    '''
    if not channels_last:
        array = np.transpose(array, (0, 2, 3, 1))

    array = array.astype('float32')

    if normalize:
        if normalize == 'mean_std':
            mean = kwargs.get('mean', 0.5)
            mean = np.array(mean).reshape((1, 1, 1, -1))
            std = kwargs.get('std', 0.5)
            std = np.array(std).reshape((1, 1, 1, -1))
            array = array * std + mean
        elif normalize == 'min_max':
            min_ = kwargs.get('min', array.min())
            min_ = np.array(min_).reshape((1, 1, 1, -1))
            max_ = kwargs.get('max', array.max())
            max_ = np.array(max_).reshape((1, 1, 1, -1))
            array -= min_
            array /= max_ + 1e-9

    batch_size, H, W, channels = array.shape
    assert rows * cols == batch_size

    if channels == 1:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1)))
        array = array[:, :, :, 0]
    elif channels == 3:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1),
                          3))
    else:
        raise TypeError('number of channels is either 1 of 3')

    for i in range(rows):
        for j in range(cols):
            img = array[i * cols + j]
            start_h = i * padding + i * H
            start_w = j * padding + j * W
            canvas[start_h: start_h + H, start_w: start_w + W] = img

    canvas = np.clip(canvas, 0, 1)
    canvas *= 255.0
    canvas = canvas.astype('uint8')
    return canvas


def viz_array_set_grid(arrays, rows, cols, padding=0, channels_last=False, normalize=False, **kwargs):
    '''
    Args:
        arrays: [Arr1, Arr2, ...] or [[Arr1, Arr2], [Arr3, Arr4], ...]
        rows, cols: rows and columns of the plot. rows * cols == array.shape[0]
        padding: padding between cells of plot
        channels_last: for Tensorflow = True, for PyTorch = False
        normalize: `False`, `mean_std`, or `min_max`
    Kwargs:
        if normalize == 'mean_std':
            mean: mean of the distribution. Default 0.5
            std: std of the distribution. Default 0.5
        if normalize == 'min_max':
            min: min of the distribution. Default array.min()
            max: max if the distribution. Default array.max()
    '''
    arr = np.array(arrays)
    if np.ndim(arr) == 5:
        arr = np.expand_dims(arr, 0)

    assert np.ndim(arr) == 6

    if not channels_last:
        arr = np.transpose(arr, (0, 1, 2, 4, 5, 3))

    arr = np.transpose(arr, [0, 3, 1, 2, 4, 5])
    arr = np.concatenate(arr)
    arr = np.transpose(arr, [1, 3, 0, 2, 4])
    arr = np.concatenate(arr)
    arr = np.transpose(arr, [2, 1, 0, 3])

    return viz_array_grid(arr, rows=rows, cols=cols, padding=padding, normalize=normalize, channels_last=True, **kwargs)


def save_image(img, path, size):
    plt.figure(figsize=size)
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

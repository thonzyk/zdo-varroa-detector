import numpy as np
from skimage import io
from pathlib import Path
import cv2
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

object_path = Path("D:/ML-Data/varroa/segMask/SegmentationObject/")
images_path = Path("D:/ML-Data/varroa/segMask/JPEGImages/")


def get_distribution_mask():
    return np.load('tmp.npy')


def get_necessary_heatmap(image):
    """
    :param image: raw image
    :return: heatmap of TP and and FP
    """

    image = 1.0 - image

    distribution_mask = get_distribution_mask() / 255.0
    # distribution_mask = get_circle_mask()

    heat_map = signal.convolve2d(image, distribution_mask, mode='same')
    heat_map = heat_map / (distribution_mask.shape[0] * distribution_mask.shape[1])

    return heat_map

    # plt.figure()
    # plt.imshow(threshold_heat_map)
    #
    # plt.figure()
    # plt.imshow(heat_map)
    # plt.show()


if __name__ == '__main__':
    image = io.imread(images_path / "Original_1305_image.jpg", as_gray=True)

    necessary_heatmap = get_necessary_heatmap(image)

    print()

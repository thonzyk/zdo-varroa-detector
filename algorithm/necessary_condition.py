import numpy as np
from scipy import signal


def get_distribution_mask():
    return np.load('nec_cond_maxk.npy')


def get_necessary_heatmap(image):
    """Compute heatmap of potential locations of targets"""
    image = 1.0 - image
    distribution_mask = get_distribution_mask() / 255.0
    heat_map = signal.convolve2d(image, distribution_mask, mode='same')
    heat_map = heat_map / (distribution_mask.shape[0] * distribution_mask.shape[1])

    return heat_map

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# AREA_THRESHOLD = 100


def remove_large_regions(mask, area_threshold):
    regions = regionprops(label(mask))

    remove_x_list = []
    remove_y_list = []

    for region in regions:
        if region.area > area_threshold:
            remove_x_list.extend(region.coords[:, 0].tolist())
            remove_y_list.extend(region.coords[:, 1].tolist())

    print()

    mask[remove_x_list, remove_y_list] = False

    return mask

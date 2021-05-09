import cv2
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# AREA_THRESHOLD = 100


def remove_regions_by_size(mask, lower_limit, upper_limit):
    mask_c = mask.copy()
    regions = regionprops(label(mask_c))

    remove_x_list = []
    remove_y_list = []

    for region in regions:
        if region.area > upper_limit or region.area < lower_limit:
            remove_x_list.extend(region.coords[:, 0].tolist())
            remove_y_list.extend(region.coords[:, 1].tolist())

    print()

    mask_c[remove_x_list, remove_y_list] = False

    return mask_c

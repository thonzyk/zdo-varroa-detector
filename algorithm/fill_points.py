import morphsnakes as ms
from skimage.measure import label, regionprops
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

REGION_SIZE = 24

INITIAL_MASK = np.ones((24, 24)).astype('bool')

# INITIAL_MASK[1:-1,1:-1] = True

INITIAL_MASK[12, 12] = False


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def fill_points(image, mask, threshold):
    regions = regionprops(label(mask))

    morph_regions_coord = []

    # find coordinates of image cuts
    for region in regions:
        centr_x = int(region.centroid[0])
        centr_y = int(region.centroid[1])
        morph_reg_coord = [[centr_x - REGION_SIZE // 2, centr_x + REGION_SIZE // 2],
                           [centr_y - REGION_SIZE // 2, centr_y + REGION_SIZE // 2]]
        morph_regions_coord.append(morph_reg_coord)

    # create the image cuts
    cuts_images = []
    for region in morph_regions_coord:
        x_from = region[0][0]
        x_to = region[0][1]
        y_from = region[1][0]
        y_to = region[1][1]

        cuts_images.append(image[x_from:x_to, y_from:y_to])

    # apply morph snakes on image cuts

    for i in range(len(cuts_images)):
        cut = cuts_images[i]
        region = morph_regions_coord[i]
        x_from = region[0][0]
        x_to = region[0][1]
        y_from = region[1][0]
        y_to = region[1][1]

        mask[x_from:x_to, y_from:y_to] = cut < threshold

    return mask

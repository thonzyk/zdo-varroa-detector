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

THRESHOLD = 0.85


def load_reference_shapes():
    regerence_shapes = np.load('maska_tenzor.npy').astype('float32')

    regerence_shapes -= 0.5

    regerence_shapes = np.sign(regerence_shapes)

    return regerence_shapes


def get_cut_by_centroid(source, centroid, sizes):
    cut = source[int(centroid[0]) - sizes[0] // 2:int(centroid[0]) + sizes[0] // 2,
          int(centroid[1]) - sizes[1] // 2:int(centroid[1]) + sizes[1] // 2]

    cut = cut.astype('float32') - 0.5
    cut = np.sign(cut)

    return cut


def filter_by_shape(mask):
    regions = regionprops(label(mask))

    reference_shapes = load_reference_shapes()

    cut_shape = (reference_shapes.shape[1], reference_shapes.shape[2])

    remove_x_list = []
    remove_y_list = []

    for region in regions:
        cut = get_cut_by_centroid(mask, region.centroid, cut_shape)
        if cut.size < cut_shape[0]*cut_shape[1]:
            remove_x_list.extend(region.coords[:, 0].tolist())
            remove_y_list.extend(region.coords[:, 1].tolist())
            continue
        cut = np.repeat(np.expand_dims(cut, axis=0), reference_shapes.shape[0], axis=0)

        scores = np.multiply(cut, reference_shapes)
        scores = np.sum(scores, axis=-1)
        scores = np.sum(scores, axis=-1)

        score = np.max(scores) / (cut_shape[0] * cut_shape[1])

        if score < THRESHOLD:
            remove_x_list.extend(region.coords[:, 0].tolist())
            remove_y_list.extend(region.coords[:, 1].tolist())

    mask[remove_x_list, remove_y_list] = False

    # plt.imshow(mask)
    # plt.show()

    return mask

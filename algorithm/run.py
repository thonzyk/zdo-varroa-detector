import numpy as np
import argparse
from skimage import io
import cv2

from algorithm.necessary_condition import get_necessary_heatmap
from algorithm.filter_large import remove_regions_by_size
from algorithm.fill_points import fill_points
from algorithm.shape_filtering import filter_by_shape
from preprocessing.preprocessing_threshold import create_hue_mask
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

parser = argparse.ArgumentParser(description='Generates sequences for labeling task')
parser.add_argument('input_file', metavar='INPUT', type=str, help='Path to .npy file containing input image.')
parser.add_argument('pred_file', metavar='PRED', type=str, help='Path to output .npy file containing prediction mask.')
parser.add_argument('label_file', metavar='LBL', type=str, help='Path to .npy file containing label mask.', nargs='?',
                    default=None)


def plot_comparison(y_true, y_pred):
    y_true = y_true[:, :, None].astype('float32')
    y_pred = y_pred[:, :, None].astype('float32')
    view = np.concatenate([y_true, y_pred, np.zeros(y_true.shape)], axis=-1)
    plt.figure()
    plt.imshow(view)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    image = io.imread(args.input_file, as_gray=False)
    image_gs = io.imread(args.input_file, as_gray=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if args.label_file:
        label_mask = io.imread(args.label_file, as_gray=False)
        label_mask = label_mask[:, :, 0].astype('float32')
        label_mask /= 255.0
        label_mask = label_mask > 0.3
    hue_mask = create_hue_mask(image).astype('float32')
    hue_mask /= 255.0
    image = image_gs * hue_mask + (hue_mask < 0.5).astype('float32')

    heat_map = get_necessary_heatmap(image)

    mask = heat_map > 0.2

    mask = remove_regions_by_size(mask, 60, 500)

    mask = fill_points(image, mask, 0.3)

    mask = remove_regions_by_size(mask, 60, 500)

    mask = filter_by_shape(mask, 0.85)

    np.save(args.pred_file, mask)

import numpy as np
import argparse
from skimage import io
import cv2

from algorithm.necessary_condition import get_necessary_heatmap
from algorithm.filter_large import remove_large_regions
from algorithm.fill_points import fill_points
from algorithm.shape_filtering import filter_by_shape
from preprocessing.preprocessing_threshold import create_hue_mask
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

parser = argparse.ArgumentParser(description='Generates sequences for labeling task')
parser.add_argument('input_file', metavar='INPUT', type=str, help='Path to .npy file containing label mask.')
parser.add_argument('pred_file', metavar='PRED', type=str, help='Path to .npy file containing prediction mask.')

if __name__ == '__main__':
    args = parser.parse_args()
    image = io.imread(args.input_file, as_gray=False)
    image_gs = io.imread(args.input_file, as_gray=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = create_hue_mask(image).astype('float32')
    image /= 255.0
    image = image_gs * image + (image < 0.5).astype('float32')

    heat_map = get_necessary_heatmap(image)

    mask = heat_map > 0.15

    mask = remove_large_regions(mask, 10000)

    mask = fill_points(image, mask, 0.3)

    mask = filter_by_shape(mask)

    np.save('../evaluation/out.npy', mask)

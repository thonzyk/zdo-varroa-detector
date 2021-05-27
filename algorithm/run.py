import numpy as np
import argparse
from skimage import io
import cv2
import os

from necessary_condition import get_necessary_heatmap
from filter_large import remove_regions_by_size
from fill_points import fill_points
from shape_filtering import filter_by_shape
from preprocessing.preprocessing_threshold import create_hue_mask

parser = argparse.ArgumentParser(description='Generates sequences for labeling task')
parser.add_argument('input_folder', metavar='INPUT', type=str, help='Path to .npy file containing input image.')
parser.add_argument('pred_folder', metavar='PRED', type=str, help='Path to output .npy file containing prediction mask.')


def get_hsv_filter(input_file):
    image = io.imread(input_file, as_gray=False)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue_mask = create_hue_mask(image).astype('float32')
    hue_mask /= 255.0
    return hue_mask


if __name__ == '__main__':
    # Load
    args = parser.parse_args()

    image_names = [fn for fn in os.listdir(args.input_folder)
                  if fn.endswith('jpg')]
    for img_name in image_names:

        image_gs = io.imread(args.input_folder+img_name, as_gray=True)
        hue_mask = get_hsv_filter(args.input_folder+img_name)

        # Filter by hue
        image = image_gs * hue_mask + (hue_mask < 0.5).astype('float32')

        # Apply necessary condition
        heat_map = get_necessary_heatmap(image)
        mask = heat_map > 0.2

        # Remove regions by size + fill points
        mask = remove_regions_by_size(mask, 60, 500)
        mask = fill_points(image, mask, 0.3)
        mask = remove_regions_by_size(mask, 60, 500)

        # Filter by shape
        mask = filter_by_shape(mask, 0.85)

        np.save(args.pred_folder+img_name[:-4], mask)


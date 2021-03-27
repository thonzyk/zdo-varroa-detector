import numpy as np
import argparse
from skimage import io

from algorithm.necessary_condition import get_necessary_heatmap
from algorithm.filter_large import remove_large_regions
from algorithm.fill_points import fill_points


parser = argparse.ArgumentParser(description='Generates sequences for labeling task')
parser.add_argument('input_file', metavar='INPUT', type=str, help='Path to .npy file containing label mask.')
parser.add_argument('pred_file', metavar='PRED', type=str, help='Path to .npy file containing prediction mask.')

if __name__ == '__main__':
    args = parser.parse_args()
    image = io.imread(args.input_file, as_gray=True)
    heat_map = get_necessary_heatmap(image)

    mask = heat_map > 0.25

    mask = remove_large_regions(mask)

    mask = fill_points(image, mask)


    np.save('../evaluation/out.npy', mask)
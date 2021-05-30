import cv2
import numpy as np

from fill_points import fill_points
from filter_large import remove_regions_by_size
from necessary_condition import get_necessary_heatmap
from preprocessing.preprocessing_threshold import create_hue_mask
from shape_filtering import filter_by_shape


def get_hsv_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue_mask = create_hue_mask(image).astype('float32')
    hue_mask /= 255.0
    return hue_mask


def predict(img):
    image_gs = np.mean(img, axis=-1) / 255.0
    hue_mask = get_hsv_filter(img)

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

    return mask

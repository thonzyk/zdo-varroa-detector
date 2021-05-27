from skimage.measure import label, regionprops


def remove_regions_by_size(mask, lower_limit, upper_limit):
    """Removes elements which does not met condition on upper and lower size limit from the given binary picture."""
    mask_c = mask.copy()
    regions = regionprops(label(mask_c))
    remove_x_list = []
    remove_y_list = []

    for region in regions:
        if region.area > upper_limit or region.area < lower_limit:
            remove_x_list.extend(region.coords[:, 0].tolist())
            remove_y_list.extend(region.coords[:, 1].tolist())

    mask_c[remove_x_list, remove_y_list] = False

    return mask_c

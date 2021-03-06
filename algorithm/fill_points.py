from skimage.measure import label, regionprops

REGION_SIZE = 24


def fill_points(image, mask, threshold):
    """Takes each element of binary mask image, look at the according place to the non-binary image and applies
    brightness thresholding. The result binary mask is returned."""
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

    # fill the dark regions by simple thresholding
    for i in range(len(cuts_images)):
        cut = cuts_images[i]
        region = morph_regions_coord[i]
        x_from = region[0][0]
        x_to = region[0][1]
        y_from = region[1][0]
        y_to = region[1][1]

        mask[x_from:x_to, y_from:y_to] = cut < threshold

    return mask

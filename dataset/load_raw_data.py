from skimage import io
from pathlib import Path
import cv2
from matplotlib import pyplot as plt

object_path = Path("D:/ML-Data/varroa/segmentation maskl/SegmentationObject/")
images_path = Path("D:/ML-Data/varroa/segmentation maskl/JPEGImages/")

if __name__ == '__main__':
    mask = io.imread(object_path / "Original_1305_image.png", as_gray=True)
    image = io.imread(images_path / "Original_1305_image.jpg", as_gray=True)

    # mask = mask > 0.5
    # mast = mask.astype('int32')

    mask = cv2.threshold(mask, thresh=0.5, maxval=1, type=cv2.THRESH_BINARY)

    result = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)


    print()

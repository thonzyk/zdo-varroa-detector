from skimage import io
from pathlib import Path
import cv2
from matplotlib import pyplot as plt

object_path = Path("C:/Users/poker/Downloads/MASK-sieberm/SegmentationClass/")
images_path = Path("C:/Users/poker/Downloads/MASK-sieberm/JPEGImages/")

if __name__ == '__main__':
    #mask = io.imread(object_path / "Original_1305_image.png", as_gray=True)
    #image = io.imread(images_path / "Original_1305_image.jpg", as_gray=True)

    im = cv2.imread('C:/Users/poker/Downloads/MASK-sieberm/SegmentationClass/Original_1305_image.png')

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,20,255,cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    #cv2.imshow('Contours', image)
    #cv2.waitKey(0)
    print()


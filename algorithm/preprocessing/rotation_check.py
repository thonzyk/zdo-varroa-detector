import cv2
import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np


if __name__== '__main__':

    relevant_path_mask = "../MASK-sieberm2/SegmentationClass"
    relevant_path_img = "../MASK-sieberm2/JPEGImages"
    extension_jpg = ['jpg']
    extension_png = ['png']

    img_names = [fn for fn in os.listdir(relevant_path_img)
                 if any(fn.endswith(ext) for ext in extension_jpg)]
    mask_names = [fn for fn in os.listdir(relevant_path_mask)
                  if any(fn.endswith(ext) for ext in extension_png)]

    for index in range(0, len(mask_names)):
        im = cv2.imread(relevant_path_mask+'/'+mask_names[index])         # mask
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #grey mask

        im2_unrot = cv2.imread(relevant_path_img + '/' + img_names[index])  # img
        im2 = cv2.rotate(im2_unrot, cv2.ROTATE_90_COUNTERCLOCKWISE)


        print(img_names[index])
        print('image'+str(im2.shape))
        print('mask' + str(im.shape))

        ret, thresh = cv2.threshold(imgray, 20, 255, cv2.THRESH_BINARY)  # klestici v masce
        contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(im2.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, -1, (255), -1)
        masked = cv2.bitwise_and(im2, im2, mask=mask)

        plt.figure()
        plt.imshow(masked)
        plt.show()
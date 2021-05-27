from skimage import io
from pathlib import Path
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')

if __name__ == '__main__':

    im = cv2.imread('C:/Users/poker/Downloads/MASK-sieberm/SegmentationClass/Original_1305_image.png')
    orig = cv2.imread('C:/Users/poker/Downloads/MASK-sieberm/JPEGImages/Original_1305_image.jpg')

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,20,255,cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

# KLESTICI MASKA Z ANOT DAT
    mat_list=[]
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        mat = image[cY - 12:cY + 12, cX - 12:cX + 12].astype("int32")
        mat_list.append(mat)

    final_mat=sum(mat_list)

    template=(final_mat.astype("float32")/19/255)
    np.save('nec_cond_maxk.npy',template)



    # template=template/ template.max()
    # template=template*255
    # template = template.astype(np.uint8)
    #
    #
    # im_in = cv2.imread('C:/Users/poker/Downloads/MASK-sieberm/JPEGImages/Original_1305_image.jpg')
    # imgray_in =cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
    #
    # res = cv2.matchTemplate(imgray_in,template,cv2.TM_CCORR_NORMED)
    # #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    # cv2.imshow("a", res)
    # cv2.waitKey(0)


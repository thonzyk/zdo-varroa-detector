from skimage import io
from pathlib import Path
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')

object_path = Path("C:/Users/poker/Downloads/MASK-sieberm/SegmentationClass/")
images_path = Path("C:/Users/poker/Downloads/MASK-sieberm/JPEGImages/")

if __name__ == '__main__':
    #mask = io.imread(object_path / "Original_1305_image.png", as_gray=True)
    orig = io.imread(images_path / "Original_1305_image.jpg", as_gray=True)

    im = cv2.imread('C:/Users/poker/Downloads/MASK-sieberm/SegmentationClass/Original_1305_image.png')

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,20,255,cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))


    # mat_list=[]
    # for c in contours:
    #     # compute the center of the contour
    #     M = cv2.moments(c)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #
    #     mat=orig[cY-20:cY+20,cX-20:cX+20]
    #     mat_list.append(mat)
    #
    # print(mat_list[0])
    #
    # for i in range(0,len(mat_list)):
    #     name="klest_"+str(i)+".png"
    #     matplotlib.image.imsave(name, mat_list[i])

    mat_list=[]
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        mat = image[cY - 20:cY + 20, cX - 20:cX + 20].astype("int32")
        mat_list.append(mat)


    #cv2.imshow('a',mat)
    #cv2.waitKey(0)

    final_mat=sum(mat_list)

    template=(final_mat.astype("float32")/19/255)



    im_in = cv2.imread('C:/Users/poker/Downloads/MASK-sieberm/JPEGImages/Original_1305_image.jpg')
    imgray_in =1-cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("a", imgray_in)
    #cv2.waitKey(0)

    #res = cv2.matchTemplate(imgray_in,template,cv2.TM_CCORR_NORMED)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #cv2.imshow("a",res)
    #cv2.waitKey(0)

    #res=res>0.5

    #plt.imshow(res)
    #plt.show()

    print()

    np.save('tmp',template)
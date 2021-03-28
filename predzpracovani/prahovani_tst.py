import numpy as np
import cv2
import matplotlib

if __name__== '__main__':
    im_mask=cv2.imread('tenzor_mask.jpg')
    im_klestici=cv2.imread('tenzor_klestici.jpg')

    # ret, thresh = cv2.threshold(im_mask, 20, 255, cv2.THRESH_BINARY)
    #
    # th=thresh[:,:,1]
    # # cv2.imshow('a',th)
    # # cv2.waitKey(0)
    # np.save('bin_mask',th)

    blurred_img=cv2.medianBlur(im_klestici,5)
    cv2.imshow('a',blurred_img)
    cv2.waitKey(0)
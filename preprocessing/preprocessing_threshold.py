import numpy as np
import cv2
import matplotlib

setup = [0, 33, 0, 255, 255, 201] # [min H, min S, min V, max H, max S, max V]
setup_invert = [12, 0, 0, 121, 231, 253]  #min H highly sensitive

size_of_blur = 11

def template_match_tst(img):

    template = cv2.imread('tst_vzor2a.jpg')

    res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    cv2.imshow('ss',res)
    cv2.waitKey(0)


def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)

    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    mask_invert = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8) #todo definice nova
    mask_invert_e = cv2.erode(mask_invert, kernel, 1)
    mask_invert_d = cv2.dilate(mask_invert_e, kernel, 1)
    output_image = cv2.bitwise_and(image, image, mask=mask_invert_d)
    return output_image

if __name__== '__main__':
    im_mask=cv2.imread('tenzor_mask.jpg')
    im_klestici=cv2.imread('tenzor_klestici.jpg')

    blur_image = cv2.medianBlur(im_klestici, size_of_blur)
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)


    masked_klestici = create_hue_mask(hsv_image, setup_invert[0:3], setup_invert[3:6])
    masked_image = cv2.cvtColor(masked_klestici, cv2.COLOR_HSV2BGR)

    #template_match_tst(masked_image)

    cv2.imshow('12',masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

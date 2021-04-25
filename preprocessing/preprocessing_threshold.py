import numpy as np
import cv2
import matplotlib

setup = [0, 33, 0, 255, 255, 201] # [min H, min S, min V, max H, max S, max V]
setup_invert = [12, 0, 0, 121, 231, 253]  #min H highly sensitive

size_of_blur = 11

def template_match_tst(img):

    template = cv2.imread('tst_vzor5.png')

    res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #res[res < 0.85] = 0

    # cv2.imshow('ss',res)
    # cv2.waitKey(0)
    out = res > 0.85

    return out


def create_hue_mask(image, lower_color=(12, 0, 0), upper_color=(121, 231, 253)):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)

    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    mask_invert = cv2.bitwise_not(mask)


    kernel = np.ones((3, 3), np.uint8) #todo definice nova
    mask_invert_e = cv2.erode(mask_invert, kernel, 1)
    mask_invert_d = cv2.dilate(mask_invert_e, kernel, 1)
    #output_image = cv2.bitwise_and(image, image, mask=mask_invert_d)
    return mask_invert_d

if __name__== '__main__':
    #im_klestici=cv2.imread('tenzor_klestici.jpg')
    im_klestici = cv2.imread('../MASK-sieberm/JPEGImages/Original_1305_image.jpg')

    blur_image = cv2.medianBlur(im_klestici, size_of_blur)
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

    masked_klestici = create_hue_mask(hsv_image)#, setup_invert[0:3], setup_invert[3:6])
    masked_image = cv2.cvtColor(masked_klestici, cv2.COLOR_HSV2BGR)

    h1, w1 = im_klestici.shape[:2]
    out=np.zeros([h1,w1])
    res= template_match_tst(im_klestici)# * 255
    h2, w2 = res.shape[:2]
    out[int((h1-h2)/2):int(h2+(h1-h2)/2), int((w1-w2)/2):int(w2+(w1-w2)/2)] = res
    out=np.rot90(out)
    np.save('../evaluation/out',out)
    # cv2.imshow('12', res)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

import numpy as np
import cv2
import matplotlib

setup = [0, 33, 0, 255, 255, 201] # [min H, min S, min V, max H, max S, max V]

size_of_blur = 5

def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)

    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    output_image = cv2.bitwise_and(image, image, mask=mask)
    return output_image

if __name__== '__main__':
    im_mask=cv2.imread('tenzor_mask.jpg')
    im_klestici=cv2.imread('tenzor_klestici.jpg')

    blur_image = cv2.medianBlur(im_klestici, size_of_blur)
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)


    masked_klestici = create_hue_mask(hsv_image, setup[0:3], setup[3:6])
    masked_image = cv2.cvtColor(masked_klestici, cv2.COLOR_HSV2BGR)

    cv2.imshow('a',masked_image)
    cv2.waitKey(0)

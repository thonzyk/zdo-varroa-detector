import cv2
import numpy as np

calibrate = 1;  #enable calibration trackbars for color and scene adjustments; disable when setup done!!!

# setup
setup = [21, 17, 0, 255, 255, 169, 100, 500, 100, 400]  # [min H, min S, min V, max H, max S, max V, x1, x2, y1, y2]


def nothing(*arg):
    pass

cap=cv2.imread('tenzor_klestici.jpg')
frame_width = cap.shape[0]
frame_height = cap.shape[1]


if (calibrate):
    cv2.namedWindow('control', 0)
    cv2.namedWindow('controlHSV', 0)

    cv2.createTrackbar('min H', 'controlHSV', 0, 255, nothing)
    cv2.createTrackbar('min S', 'controlHSV', 0, 255, nothing)
    cv2.createTrackbar('min V', 'controlHSV', 0, 255, nothing)

    cv2.createTrackbar('max H', 'controlHSV', 0, 255, nothing)
    cv2.createTrackbar('max S', 'controlHSV', 0, 255, nothing)
    cv2.createTrackbar('max V', 'controlHSV', 0, 255, nothing)
    cv2.setTrackbarPos('max H', 'controlHSV', 255)
    cv2.setTrackbarPos('max S', 'controlHSV', 255)
    cv2.setTrackbarPos('max V', 'controlHSV', 255)

    cv2.createTrackbar('x1', 'control', 0, frame_width, nothing)
    cv2.createTrackbar('x2', 'control', 0, frame_width, nothing)
    cv2.setTrackbarPos('x2', 'control', frame_width)

    cv2.createTrackbar('y1', 'control', 0, frame_height, nothing)
    cv2.createTrackbar('y2', 'control', 0, frame_height, nothing)
    cv2.setTrackbarPos('y2', 'control', frame_height)


def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)

    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    output_image = cv2.bitwise_and(image, image, mask=mask)
    return output_image


while (True):

    frame = cv2.imread('tenzor_klestici.jpg')
    #ret, frame = cap.read();
    cv2.imshow("orig",frame)

    if (calibrate):
        minH = cv2.getTrackbarPos('min H', 'controlHSV')
        minS = cv2.getTrackbarPos('min S', 'controlHSV')
        minV = cv2.getTrackbarPos('min V', 'controlHSV')

        maxH = cv2.getTrackbarPos('max H', 'controlHSV')
        maxS = cv2.getTrackbarPos('max S', 'controlHSV')
        maxV = cv2.getTrackbarPos('max V', 'controlHSV')

        x1 = cv2.getTrackbarPos('x1', 'control')
        x2 = cv2.getTrackbarPos('x2', 'control')
        y1 = cv2.getTrackbarPos('y1', 'control')
        y2 = cv2.getTrackbarPos('y2', 'control')

        if (x1 < x2 and y1 < y2):
            frame = frame[y1:y2, x1:x2]
    else:
        frame = frame[setup[8]:setup[9], setup[6]:setup[7]]

    blur_image = cv2.medianBlur(frame, 3)
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

    if (calibrate):
        red_hue = create_hue_mask(hsv_image, [minH, minS, minV], [maxH, maxS, maxV])
        masked_image = cv2.cvtColor(red_hue, cv2.COLOR_HSV2BGR)
        cv2.imshow("threshold", masked_image)
    else:
        red_hue = create_hue_mask(hsv_image, [setup[0], setup[1], setup[2]], [setup[3], setup[4], setup[5]])

    if cv2.waitKey(33) == ord('a'):
        break
        cv2.destroyAllWindows()
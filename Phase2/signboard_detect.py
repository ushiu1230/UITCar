import re
from sre_parse import State
import numpy as np
import cv2

# red = np.array([0,0,255])
# green = np.array([0,255,0])
# blue = np.array([255,0,0])
# violet = np.array([255,0,255])
# yellow = np.array([0,255,255])
def detect_sign(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    gray_blurred = cv2.blur(gray, (3,3))
    cv2.imshow("b_gray",gray_blurred)
    detected_circles = cv2.HoughCircles(gray_blurred,
            cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
        param2 = 30, minRadius = 5, maxRadius = 20)
        # Draw circles that are detected.
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2] 
            return a,b
    return 0 , 0

def detect_color(img, state):
    x, y = detect_sign(img)
    if (x and y != 0 ):
        b,g,r = img[y,x]
        print(b,g,r)
        if b == 0 and g == 0 and r == 255:
            state = 1
        elif b == 0 and g == 255 and r == 0:
            state = 2
        elif b == 255 and g == 0 and r == 0:
            state = 3
        elif b == 255 and g == 0 and r == 255:
            state = 4
        elif b == 0 and g == 255 and r == 255:
            state = 5
    return state

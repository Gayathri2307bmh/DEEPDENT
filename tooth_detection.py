import cv2
import numpy as np

def detect_teeth(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _,thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY_INV)

    # Morphological operations for better segmentation
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    teeth = []

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        if w>30 and h>30:

            tooth = img[y:y+h,x:x+w]

            teeth.append((tooth,(x,y,w,h)))

    return img, teeth
"""
Assignment 1 Test program takes input pic pup keep in color
generates an elliptical portrait with gradient grayscale exterior
"""


import numpy
import cv2
import math

windowname = 'me'
img = cv2.imread('prof pic.jpg')
LENGTH, WIDTH, COLORS = img.shape


def ellipseFunction(x, y):
    h = LENGTH/2
    k = WIDTH/3
    a = ((x-k)**2)/((WIDTH/3)**2)
    b = ((y-h)**2)/(((7*h)/8)**2)
    return a+b


for i in range(LENGTH):
    for j in range(WIDTH):
        val = ellipseFunction(j, i)
        if val > 1.0:
            img[i, j, 0] = 255 - ((val - 1) * 50)
            img[i, j, 1] = 255 - ((val - 1) * 50)
            img[i, j, 2] = 255 - ((val - 1) * 50)


cv2.imshow(windowname, img)
print(LENGTH, WIDTH)
cv2.waitKey()
cv2.destroyAllWindows()



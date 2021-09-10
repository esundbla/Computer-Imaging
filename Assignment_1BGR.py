"""
Assignment 1 Test program takes input pic pup keep in color
generates an elliptical portrait with gradient grayscale exterior
"""


import numpy
import cv2
import math

windowname = 'pup'
img = cv2.imread('pup.jpg')
LENGTH, WIDTH, COLORS = img.shape


def ellipseFunction(x, y):
    h = LENGTH//2
    k = WIDTH//2
    a = ((x-h)**2)/((h/2)**2)
    b = ((y-k)**2)/(((3*k)/4)**2)
    return a+b


for i in range(LENGTH):
    for j in range(WIDTH):
        val = ellipseFunction(j, i)
        if val > 1.0:
            img[i:, j:, 0] = 255 - ((val - 1) * 17)
            img[i:, j:, 1] = 255 - ((val - 1) * 17)
            img[i:, j:, 2] = 255 - ((val - 1) * 17)


cv2.imshow(windowname, img)
print(LENGTH, WIDTH)



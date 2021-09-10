"""
Assignment 1 Test program takes input pic in grayscale
generates an elliptical portrait with gradient grayscale exterior
"""


import numpy
import cv2
import math

windowname = 'pup'
img = cv2.imread('prof pic.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
LENGTH, WIDTH = img.shape   # global variables of imported pic


def ellipseFunction(x, y):
    """ helper function takes a pixel location as a co-ordinate and returns
    the value of the equation (x-h)^2/a^2 + (y-k)^2/b^2 the ellipse equation"""
    h = LENGTH//2
    k = WIDTH//2
    a = ((x-h)**2)/((h/2)**2)
    b = ((y-k)**2)/(((3*k)/4)**2)
    return a+b


for i in range(LENGTH):
    for j in range(WIDTH):
        val = ellipseFunction(j, i)
        print(val)
        if val > 1.0:            # if return value of a given pixel is >1 the pixel isnt inside the ellipse
            img[i, j] = 255 - ((val - 1) * 50)  # using 0-max val given by ellipse we can gradient out
            img[i, j] = 255 - ((val - 1) * 50)  # the exterior of the ellipse to created a fake grayscale
            img[i, j] = 255 - ((val - 1) * 50)  # 50 hard coded in/could be found by entering 0,0 pixel in ellipseFunc
                                                # and dividing 255 by given val


cv2.imshow(windowname, img)
print(LENGTH, WIDTH)
cv2.waitKey()
cv2.destroyAllWindows()

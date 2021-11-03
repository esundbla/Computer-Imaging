

import cv2
import math
import numpy as np


#Given color thresholds
green_high = (160, 210, 120)
green_low = (80, 130, 30)
yellow_high = (255, 255, 75)
yellow_low = (155, 120, 0)
orange_high = (255, 90, 75)
orange_low = (115, 50, 30)
pink_high = (255, 95, 160)
pink_low = (160, 40, 70)
blue_high = (100, 155, 255)
blue_low = (25, 45, 120)
purple_high = (140, 70, 120)
purple_low = (75, 40, 60)
white_high = (200, 200, 200)
white_low = (150, 150, 150)

#Dictionary for simple organization
color_dictionary = {
    "green"  : (green_high, green_low),
    "yellow" : (yellow_high, yellow_low),
    "orange" : (orange_high, orange_low),
    "pink"   : (pink_high, pink_low),
    "blue"   : (blue_high, blue_low),
    "purple" : (purple_high, purple_low),
}

def show(image, color):
    """ Helper method to display a single image
    with pyplot """
    if (color == "gray"):
        pyplot.imshow(image, cmap="gray")
    else:
        pyplot.imshow(image)
    pyplot.show()


if __name__ == "__main__":
    source = cv.imread('test2-1.JPG')
    source = cv.cvtColor(wall, cv2.COLOR_BGR2RGB)
    show(source, 'RGB')

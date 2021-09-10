#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:08:39 2020

"""

import numpy as np
import cv2
import math

# from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow


windowname = 'image'
im2 = cv2.imread('prof pic.jpg')

#im2[:,:,1]=255  #G try this
#im2[:,:,0]=255  #B try this
#im2[:, :, 2] = 255  # R
print(im2[0, 0, 0])  # check the first pixel on top left corner
print(im2[0, 0, 1])  # check the first pixel on top left corner
print(im2[0, 0, 2])  # check the first pixel on top left corner
cv2.imshow(windowname, im2)  # what can you see? IS the image more green&bluish or more red?




# cv2.imshow(windowname,im1)


cv2.waitKey()
cv2.destroyAllWindows()


import numpy as np
import cv2
import math
from matplotlib import pyplot as plt



#testarray
im1=cv2.imread('prof pic.jpg')
img = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)


length, width = img.shape
radius = 0.5*min(length,width)
radius
for i in range(length):
    for j in range(width):
        if math.sqrt((i-length/2)**2 + (j-width/2)**2) > radius:
            img[i,j] = 128
plt.figure()
#plt.imshow(img)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:08:39 2020

"""

import numpy as np
import cv2 
import math
import random
from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow

# =============================================================================
# #1  Negative
# =============================================================================
im=cv2.imread('./1_test.jpg')
#im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#plt.figure()
#plt.imshow(im_rgb)
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
length, width = img.shape
plt.figure()
plt.imshow(img,cmap='gray', vmin=0, vmax=255) 

im1=np.zeros((length, width))
for i in range(length):     # this is the row
    for j in range(width):  # this is the column      
             im1[i,j] = 255-img[i,j]    # we don't have to loop    
plt.figure()
plt.imshow(im1,cmap='gray', vmin=0, vmax=255)  

# =============================================================================
###2 contrast stretching
#mx = max(max(im));
#mn = min(min(im));
#im2  = (im-mn)/(mx-mn)*255;
# =============================================================================
im2=np.zeros((length, width))
mx=np.amax(img)
mn=np.amin(img)
im2=(img-mn)/(mx-mn)*255
#for i in range(length):     # this is the row
#    for j in range(width):  # this is the column      
#             im2[i,j] = (img[i,j]-mn)/(mx-mn)*255   # we don't need to loop   
plt.figure()
plt.title('contrast stretching')
plt.imshow(im2,cmap='gray', vmin=0, vmax=255)  

# =============================================================================
#3 Gamma / power law
# =============================================================================
im3=np.zeros((length, width))
mx=np.amax(img)
mn=np.amin(img)
imgN=img/255
im3=imgN**2.2   
plt.figure()
plt.imshow(im3,cmap='gray', vmin=0, vmax=1) # why we need to normalize the image?

im4=np.zeros((length, width))
im4=imgN**0.5   
plt.figure()
plt.imshow(im4,cmap='gray', vmin=0, vmax=1)

# =============================================================================
# 4 histogram equalization
# =============================================================================
im=cv2.imread('./veg.jpg')
img2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
length2, width2 = img2.shape
######### Histogram by Numpy ###
#hist,bins = np.histogram(img.flatten(),256,[0,256]) #numpy histogram
######### Histogram by Feng ###
hist=np.zeros(256)
for i in range(length2):     # this is the row
    for j in range(width2):  # this is the column  
        hist[img2[i,j]] = hist[img2[i,j]]+1   

hist_n=hist*255/hist.max()
cdf = hist.cumsum()
cdf_n = cdf * 255/ cdf.max()
plt.figure()
plt.plot(cdf_n, color = 'b')
plt.plot(hist_n,color='r')
#plt.hist(img.flatten(),256,[0,256], color = 'r')
######### Histogram equalization by feng ###
myeq=np.zeros((length2, width2))
for i in range(length2):
    for j in range(width2):
        pixel = img2[i][j]
        myeq[i][j] = cdf_n[pixel]

plt.figure()
plt.title('Feng equalize Hist')
plt.imshow(myeq,cmap='gray', vmin=0, vmax=255)

######### Histogram equalization by opnecv ###
equ = cv2.equalizeHist(img2)
res = np.hstack((img2,equ)) #stacking images side-by-side
plt.figure()
plt.title('cv2.equalizeHist')
plt.imshow(res,cmap='gray', vmin=0, vmax=255)


#what will happen if I don't change back to BGR and show it?        
#windowname = 'image'
#cv2.imshow(windowname,imrgb) #this should work but not in Colab
#cv2.waitKey()  
#cv2.destroyAllWindows() 
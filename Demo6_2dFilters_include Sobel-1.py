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
from scipy.signal import convolve2d
#from google.colab.patches import cv2_imshow

  
# =============================================================================
# #1   average filter
# =============================================================================
im=cv2.imread('./1_test.jpg')
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
length, width = img.shape
plt.figure()
plt.imshow(img,cmap='gray', vmin=0, vmax=255) 
# averaging filter
box = np.array(
    [[0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04]]
)
box2 = np.array(
    [[1/9,1/9,1/9],
    [1/9,1/9,1/9],
    [1/9,1/9,1/9]]
)
average = cv2.filter2D(img,-1,box2)    
plt.figure()
plt.imshow(average,cmap='gray', vmin=0, vmax=255)  

# =============================================================================
###2  sobel filter
# =============================================================================
sobel_vert = np.array([
         [-1.0, 0.0, 1.0]
        ,[-2.0, 0.0, 2.0]
        ,[-1.0, 0.0, 1.0]
        ])
sobel_horiz = sobel_vert.T

d_horiz = convolve2d(img, sobel_horiz, mode='same', boundary = 'symm', fillvalue=0)
d_vert = convolve2d(img, sobel_vert, mode='same', boundary = 'symm', fillvalue=0)
grad=np.sqrt(np.square(d_horiz) + np.square(d_vert))
grad *= 255.0 / np.max(grad)
plt.figure()
plt.title('Vert_by 2d Convolve')
plt.imshow(d_vert,cmap='gray', vmin=0, vmax=255) 
plt.figure()
plt.title('Horiz_ by 2d Convolve')
plt.imshow(d_horiz,cmap='gray', vmin=0, vmax=255) 
plt.figure()
plt.title('Gradient Edge by 2d Conv')
plt.imshow(grad,cmap='gray', vmin=0, vmax=255) 

# use OpenCV functions
dst_vert = cv2.filter2D(img, -1, sobel_vert) 
dst_horiz = cv2.filter2D(img, -1, sobel_horiz) 
plt.figure()
plt.title('Vert_by opencv')
plt.imshow(dst_vert,cmap='gray', vmin=0, vmax=255) 
plt.figure()
plt.title('Horiz_ by opencv')
plt.imshow(dst_horiz,cmap='gray', vmin=0, vmax=255) 

#try to use sqrt of vert and horiz, check what is your output and why
#grad2 = np.sqrt(np.square(dst_vert) + np.square(dst_horiz)).astype(int)
#plt.figure()
#plt.title('WHy it is so black?')
#plt.imshow(grad2,cmap='gray', vmin=0, vmax=255) 
grad3=np.maximum(dst_vert, dst_horiz) 
plt.figure()
plt.title('Max Edge by opencv')
plt.imshow(grad3,cmap='gray', vmin=0, vmax=255) 

# =============================================================================
#3  Laplacian
# =============================================================================
l_kern = np.array([
         [0.0,  1.0, 0.0]
        ,[1.0, -4.0, 1.0]
        ,[0.0,  1.0, 0.0]
        ])
#// alternate form uses diagonals 
l_kern2 = np.array([
         [1.0,  1.0, 1.0]
        ,[1.0, -8.0, 1.0]
        ,[1.0,  1.0, 1.0]
        ])
Edge_l = cv2.filter2D(img, -1, l_kern2) 
plt.figure()
plt.title('Laplacian Edge')
plt.imshow(Edge_l,cmap='gray', vmin=0, vmax=255) 
# =============================================================================
# 4  median filter
# also check    cv.medianBlur(src, dst, 5);
# =============================================================================
height,width = np.shape(img)
median = np.zeros((height,width),dtype=float)
for i in range(1,height-2):
    for j in range(1,width-2):
        sorted_pixels = sorted(np.ndarray.flatten(img[i-1:i+2,j-1:j+2]))
        median[i][j] = sorted_pixels[4] 
plt.figure()
plt.title('median filter')
plt.imshow(median,cmap='gray', vmin=0, vmax=255) 

# =============================================================================
# 5  gaussian filter
# =============================================================================
#from scipy.ndimage import gaussian_filter
#gaussian_filter(a, sigma=1)
G_ker=np.array([
       [ 4,  6,  8,  9, 11],
       [10, 12, 14, 15, 17],
       [20, 22, 24, 25, 27],
       [29, 31, 33, 34, 36],
       [35, 37, 39, 40, 42]])
G_ker =G_ker/np.sum(G_ker)
Gaussian = cv2.filter2D(img, -1, G_ker) 
plt.figure()
plt.title('Gaussian blur')
plt.imshow(Gaussian,cmap='gray', vmin=0, vmax=255) 
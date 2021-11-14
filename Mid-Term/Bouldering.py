"""
Erik Sundblad
CS3150
11/12/2021
Program to isolate a selected bouldering path off given sample image then highlight said path
Added additional average path line to define start and finnish of selected path
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

global_filters = {          # Kernels for gradient edge filter
    "Sobel(V)": np.array([
         [-1.0, 0.0, 1.0]
        ,[-2.0, 0.0, 2.0]
        ,[-1.0, 0.0, 1.0]
        ]),
    "Sobel(H)": np.array([
         [-1.0, -2.0, -1.0]
        ,[0.0, 0.0, 0.0]
        ,[1.0, 2.0, 1.0]
        ])}


# Given color thresholds
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

# Dictionary for simple organization
color_dictionary = {
    # "green"  : (green_high, green_low), Bad color thresholding results in poor isolation
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
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.show()

def findPath(img, color):
    # Read in, extract specific path color, morph path
    high, low = color_dictionary[color]
    color_mask = cv2.inRange(img, low, high)
    path_raw = cv2.bitwise_and(img, img, mask=color_mask)
    morphed = morphTool(path_raw)
    # Convert to LUV to isolate the luminance and apply a gradient edge filter
    luv = cv2.cvtColor(morphed, cv2.COLOR_BGR2LUV)
    l = luv[:, :, 0]
    edge = gradientEdge(l)
    # Using gradient edge image, clean image if lum > 128 set to 128 otherwise set to 0
    # Then isolate top and bottom gradient lines and reserve to calculate average path line
    length, width = edge.shape
    x_bottom_pos = []
    x_top_pos = []
    y_top = 0
    y_bottom = 0
    for i in range(length):
        for j in range(width):
            if edge[i, j] > 128:
                edge[i, j] = 128
            else:
                edge[i, j] = 0
            if edge[i, j] == 128:
                x_bottom_pos.append(j)
                if len(x_top_pos) == 0:
                    y_top = (-i)
                    x_top_pos.append(j)
                elif i == abs(y_top):
                    x_top_pos.append(j)
                if i > abs(y_bottom):
                    y_bottom = (-i)
                    x_bottom_pos.clear()
                    x_bottom_pos.append(j)
                elif i == abs(y_bottom):
                    x_bottom_pos.append(j)
    # use average x_pos of top and bottom to get x,y co-ordinates for average path line
    x_top = sum(x_top_pos) // len(x_top_pos)
    x_bottom = sum(x_bottom_pos) // len(x_bottom_pos)
    # Slope and y intercept calculations note y pos are negative to compensate for orientation
    if x_top > x_bottom:
        slope = ((y_top - y_bottom)/(x_top - x_bottom))
        start = x_bottom
        stop = x_top
    else:
        slope = ((y_bottom - y_top)/(x_bottom - x_top))
        start = x_top
        stop = x_bottom
    intercept = y_bottom - (slope * x_bottom)
    # Draw average line on edge extract
    for x in range(start, stop):
        y = int((slope * x) + intercept)
        edge[(-y), x] = 128
    # Use dilation morph to better highlight
    path_edge = morphTool2(edge)
    # Combine edge with original file to "highlight" selected path
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    img_luv[:,:, 0] = img_luv[:,:, 0] - (path_edge)
    path_final = cv2.cvtColor(img_luv, cv2.COLOR_LUV2BGR)
    # Print result
    show(path_final, 'RGB')


def morphTool(img):
    """ First morph to clean color extracton then dialate to outline path sections"""
    kernel_1 = np.ones((35, 35), np.uint8)  # square kernal of 1
    kernel_2 = np.ones((19, 19), np.uint8)  # square kernal of 1
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_2)
    dilation = cv2.dilate(opening, kernel_1, iterations = 4)
    return dilation


def gradientEdge(pic):
    """ Gradient edge function combines the verticle and horizontal sobel filters """
    ver = convolve2d(pic, global_filters["Sobel(V)"], mode='same', boundary='symm', fillvalue=0)
    hor = convolve2d(pic, global_filters["Sobel(H)"] , mode='same', boundary='symm', fillvalue=0)
    gradient = np.sqrt(np.square(ver) + np.square(hor))
    gradient *= 255.0/np.max(gradient)
    return gradient


def morphTool2(img):
    """ Second morph tool specifically for dilation of gradient edges + path line """
    kernel = np.ones((35, 35), np.uint8)  # square kernal of 1
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation


if __name__ == "__main__":

    while(True):
        # Loop structure to allow multiple path selection
        source = cv2.imread('test2-1.JPG')
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        show(source, 'RGB')

        print('Routes: yelllow, orange, pink, blue, purple')
        selection = input("Desired route: ").lower()
        if selection == "":
            break
        if selection in color_dictionary:
            # Run find path to extract results
            findPath(source, selection)
        else:
            print("Invalid path color")
        print("Select different path or return to exit")

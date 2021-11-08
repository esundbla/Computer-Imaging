

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

global_filters = {          #Utilizing a python Dict. to store multiple kerels and there respective names
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
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.show()

def findPath(img, color):
    high, low = color_dictionary[color]
    color_mask = cv2.inRange(img, low, high)
    path_raw = cv2.bitwise_and(img, img, mask=color_mask)
    #show(path_raw, 'RGB')
    morphed = morphTool(path_raw)
    luv = cv2.cvtColor(morphed, cv2.COLOR_BGR2LUV)
    l = luv[:, :, 0]
    edge = gradientEdge(l)
    dial_edge = morphTool2(edge)
    length, width = dial_edge.shape
    for i in range(length):
        for j in range(width):
            if dial_edge[i,j] > 100:
                dial_edge[i,j] = 128
            else:
                dial_edge[i,j] = 0
    show(dial_edge, 'gray')
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    img_luv[:,:, 0] = img_luv[:,:, 0] - (dial_edge)
    path_final = cv2.cvtColor(img_luv, cv2.COLOR_LUV2BGR)
    show(path_final, 'RGB')

def morphTool(img):
    kernel_1 = np.ones((35, 35), np.uint8)  # square kernal of 1
    kernel_2 = np.ones((19, 19), np.uint8)  # square kernal of 1
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_2)
    #show(opening, 'RGB')
    dilation = cv2.dilate(opening, kernel_1, iterations = 4)
    #show(dilation, 'RGB')
    return dilation

def gradientEdge(pic):
    """ Gradient edge function combines the verticle and horizontal sobel filters """
    ver = convolve2d(pic, global_filters["Sobel(V)"], mode='same', boundary='symm', fillvalue=0)
    hor = convolve2d(pic, global_filters["Sobel(H)"] , mode='same', boundary='symm', fillvalue=0)
    gradient = np.sqrt(np.square(ver) + np.square(hor))
    gradient *= 255.0/np.max(gradient)
    plt.figure()
    plt.title('Gradient Edge')
    plt.imshow(gradient, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return gradient

def morphTool2(img):
    kernel = np.ones((35, 35), np.uint8)  # square kernal of 1
    dilation = cv2.dilate(img, kernel, iterations=1)
    show(dilation, 'gray')
    return dilation




if __name__ == "__main__":

    while(True):
        source = cv2.imread('test2-1.JPG')
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        #show(source, 'RGB')

        print('Routes: yelllow, orange, pink, blue, purple')
        selection = input("Desired route: ").lower()
        if selection == "":
            break
        if selection in color_dictionary:
            findPath(source, selection)
        else:
            print("Invalid path color")
        print("Select different path or return to exit")
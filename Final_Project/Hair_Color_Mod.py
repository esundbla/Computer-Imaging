import random
import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


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


def pixel_perc_filter(perc, hair_space):
    length, width = hair_space.shape
    percent_filt = np.ones((length, width), np.uint8)
    for i in range(length):
        for j in range(width):
            if hair_space[i, j] > 0 and i < 1200:
                percent_filt[i, j] = (0.5+ (perc/100))
    return percent_filt


def morphTool(img):

    kernel_1 = np.ones((17, 17), np.uint8)
    kernel_2 = np.ones((25, 25), np.uint8)
    kernel_3 = np.ones((37, 37), np.uint8)
    kernel_4 = np.ones((3, 3), np.uint8)

    dilation = cv.dilate(img, kernel_1, iterations=1)
    open = cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel_2)
    erosion = cv.erode(open, kernel_3, iterations=2)
    open2 = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel_2)
    d2 = cv.dilate(open2, kernel_4, iterations=33)
    return d2


def gradientEdge(pic):
    """ Gradient edge function combines the verticle and horizontal sobel filters """
    ver = convolve2d(pic, global_filters["Sobel(V)"], mode='same', boundary='symm', fillvalue=0)
    hor = convolve2d(pic, global_filters["Sobel(H)"] , mode='same', boundary='symm', fillvalue=0)
    gradient = np.sqrt(np.square(ver) + np.square(hor))
    gradient *= 255.0/np.max(gradient)
    return gradient

if __name__ == "__main__":
    initial_img = cv.imread('Charlotte(Crop).jpg')
    length, width, extra = initial_img.shape
    rgb_img = cv.cvtColor(initial_img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.title('original')
    plt.imshow(rgb_img)


    red_img = rgb_img[:, :, 0]
    blue_img = rgb_img[:, :, 2]
    green_img = rgb_img[:, :, 1]

    luv = cv.cvtColor(initial_img, cv.COLOR_BGR2LUV)
    lum = luv[:, :, 0]

    grad = gradientEdge(lum)
    for i in range(length):
        for j in range(width):
            if grad[i, j] > 11:
                grad[i, j] = 255
            else:
                grad[i, j] = 0


    morph_grad = morphTool(grad)
    #plt.figure()
    #plt.title('luv grad morph')
    #plt.imshow(morph_grad)
    #plt.show()


    red_per = float(input("Red Percent: "))
    new_red = pixel_perc_filter(red_per, morph_grad)
    grn_per = float(input("Green Percent: "))
    new_green = pixel_perc_filter(grn_per, morph_grad)
    blu_per = float(input("Blue Percent: "))
    new_blue = pixel_perc_filter(blu_per, morph_grad)
    mod_img = rgb_img
    mod_img[:, :, 0] = (mod_img[:, :, 0] * new_red)
    mod_img[:, :, 1] = (mod_img[:, :, 1] * new_green)
    mod_img[:, :, 2] = (mod_img[:, :, 2] * new_blue)

    plt.figure()
    plt.title('modified')
    plt.imshow(mod_img)
    plt.show()
    exit()








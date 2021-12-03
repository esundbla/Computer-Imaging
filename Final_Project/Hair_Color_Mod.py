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


def morphTool(img):
    """"""
    kernel_1 = np.ones((11, 11), np.uint8)  # square kernal of 1
    kernel_2 = np.ones((10, 10), np.uint8)  # square kernal of 2

    open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_2)
    close = cv.morphologyEx(open, cv.MORPH_CLOSE, kernel_1)
    return close


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
    plt.title('initial color')
    plt.imshow(rgb_img)
    #rgb_img[:, :, 0] = 0 #Red
    rgb_img[:, :, 1] = 0 #Green
    rgb_img[:, :, 2] = 0 #Blue
    plt.figure()
    plt.title('rgb')
    plt.imshow(rgb_img)
    plt.show()

    gray_img = cv.cvtColor(initial_img, cv.COLOR_BGR2GRAY)
    gradient = gradientEdge(gray_img)
    plt.figure()
    plt.title('gradient')
    plt.imshow(gradient)
    plt.show()

    morph2 = morphTool(gradient)
    plt.figure()
    plt.title('gradient')
    plt.imshow(morph2)
    plt.show()

    """red_img = rgb_img[:, :, 0]
    morph = morphTool(red_img)
    #morph = morphTool(morph)
    #morph = morphTool(morph)
    plt.figure()
    plt.title('gradient')
    plt.imshow(morph)
    plt.show()

    for i in range(length):
        for j in range(width):
            if red_img[i, j] > 150 or red_img[i, j] < 70:
                red_img[i, j] = 0
    plt.figure()
    plt.title('rgb altered space')
    plt.imshow(red_img)
    plt.show()
    


    luv = cv.cvtColor(initial_img, cv.COLOR_BGR2LUV)
    lum = luv[:, :, 0]
    plt.figure()
    plt.title('initial color')
    plt.imshow(lum)

    for i in range(length):
        for j in range(width):
            if lum[i,j] > 100:
                lum[i,j] = 0
    plt.figure()
    plt.title('modified')
    plt.imshow(lum)
    plt.show()"""



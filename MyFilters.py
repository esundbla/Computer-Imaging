import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

global_filters = {
    "Average": np.array([[(1/49), (1/49), (1/49), (1/49), (1/49), (1/49), (1/49)],
                        [(1/49), (1/49), (1/49), (1/49), (1/49), (1/49), (1/49)],
                        [(1/49), (1/49), (1/49), (1/49), (1/49), (1/49), (1/49)],
                        [(1/49), (1/49), (1/49), (1/49), (1/49), (1/49), (1/49)],
                        [(1/49), (1/49), (1/49), (1/49), (1/49), (1/49), (1/49)],
                        [(1/49), (1/49), (1/49), (1/49), (1/49), (1/49), (1/49)],
                        [(1/49), (1/49), (1/49), (1/49), (1/49), (1/49), (1/49)]]),
    "Sobel(V)": np.array([
         [-1.0, 0.0, 1.0]
        ,[-2.0, 0.0, 2.0]
        ,[-1.0, 0.0, 1.0]
        ]),
    "Sobel(H)": np.array([
         [-1.0, -2.0, -1.0]
        ,[0.0, 0.0, 0.0]
        ,[1.0, 2.0, 1.0]
        ]),
    "Laplacian": np.array([
        [1.0, 1.0, 1.0],
        [1.0, -8.0, 1.0],
        [1.0, 1.0, 1.0]]),
    "Gaussian": np.array([
        [1/26, 3/26, 1/26],
        [3/26, 10/26, 3/36],
        [1/26, 3/26, 1/26]]),
    "Self_Def_Gaus": np.array([
        [1/44, 5/44, 1/44],
        [5/44, 20/44, 5/44],
        [1/44, 5/44, 1/44]]),
    "Spacial-large": np.array([[(1/382), (2/382), (3/382), (4/382), (3/382), (2/382), (1/382)],
                        [(2/382), (4/382), (7/382), (11/382), (7/382), (4/382), (2/382)],
                        [(3/382), (7/382), (14/382), (25/382), (14/382), (7/382), (3/382)],
                        [(4/382), (11/382), (25/382), (50/382), (25/382), (11/382), (4/382)],
                        [(3/382), (7/382), (14/382), (25/382), (14/382), (7/382), (3/382)],
                        [(2/382), (4/382), (7/382), (11/382), (7/382), (4/382), (2/382)],
                        [(1/382), (2/382), (3/382), (4/382), (3/382), (2/382), (1/382)]]),}


def filter(pic, kernel):
    """ Generic Filter func takes picture to process, filter kernel, and filter name"""
    filtered = cv2.filter2D(pic, -1, global_filters[kernel])
    windowname = kernel
    plt.figure()
    plt.title(kernel)
    plt.imshow(filtered, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return filtered

def gradientEdge(pic):
    """ Gradient edge function combines the verticle and horizontal sobel filters """
    ver = cv2.filter2D(pic, -1, global_filters["Sobel(V)"])
    hor = cv2.filter2D(pic, -1, global_filters["Sobel(H)"])
    #v2 = np.square(ver)
    #h2 = np.square(hor)
    #g2 = np.add(v2, h2)
    #gradient = np.sqrt(g2)
    gradient = ver + hor
    plt.figure()
    plt.title('Gradient Edge')
    plt.imshow(gradient, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return gradient


def median(pic):
    """ Median filter need special implementation """
    height, width, something = np.shape(pic)
    med = np.ones((height, width), dtype=float)
    for i in range(1, height - 2):
        for j in range(1, width - 2):
            pixels_in_array = sorted(np.ndarray.flatten(pic[i-1:i+2,j-1:j+2]))
            med[i][j] = pixels_in_array[4]
    plt.figure()
    plt.title('median filter')
    plt.imshow(med, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return med

if __name__ == "__main__":
    im = cv2.imread('kuma2.jpg')
    pic = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.title('Source Pic')
    plt.imshow(pic, cmap='gray', vmin=0, vmax=255)
    plt.show()
    #filter(pic, "Average")
    #filter(pic, "Sobel(V)")
    #filter(pic, "Sobel(H)")
    #gradientEdge(pic)
    #median(pic)
    #filter(pic, "Laplacian")
    #filter(pic, "Gaussian")
    #filter(pic, "Self_Def_Gaus")
    #filter(pic, "Spacial-large")

    pic2 = cv2.imread('Iris.bmp')
    detect = filter((filter((filter(filter(pic2, 'Gaussian'), 'Laplacian') * pic2), 'Average') - pic2), 'Laplacian')
    plt.figure()
    plt.title('Original')
    plt.imshow(pic2, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.figure()
    plt.title('Twice filtered')
    plt.imshow(detect, cmap='gray', vmin=0, vmax=255)
    plt.show()










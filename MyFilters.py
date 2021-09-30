import numpy
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


def filter(pic, kernal):
    """ Generic Filter func takes picture to process, filter kernal, and filter name"""
    average = cv2.filter2D(pic, -1, global_filters[kernal])
    windowname = kernal
    cv2.imshow(windowname, average)
    cv2.waitKey()
    #cv2.destroyAllWindows()

def gradientEdge(pic):
    """ Gradient edge function """
    ver = cv2.filter2D(pic, -1, global_filters["Sobel(V)"])
    hor = cv2.filter2D(pic, -1, global_filters["Sobel(H)"])
    #v2 = np.square(ver)
    #h2 = np.square(hor)
    #g2 = v2 + h2
    #gradient = numpy.sqrt(g2)
    grad = ver + hor
    windowname = "Gradient Edge"
    cv2.imshow(windowname, grad)


def Median(pic):
    """ Median filter need special implementation """


if __name__ == "__main__":
    im = cv2.imread('kuma2.jpg')
    pic = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    length, width = pic.shape
    windowname = 'Inital Pic'
    cv2.imshow(windowname, pic)
    cv2.waitKey()

    #cv2.destroyAllWindows()

    filter(pic, "Average")
    filter(pic, "Sobel(V)")
    filter(pic, "Sobel(H)")
    gradientEdge(pic)
    filter(pic, "Laplacian")
    filter(pic, "Gaussian")
    filter(pic, "Self_Def_Gaus")
    filter(pic, "Spacial-large")




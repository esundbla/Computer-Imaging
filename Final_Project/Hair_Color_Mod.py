import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt






if __name__ == "__main__":
    intial_img = cv.imread('Charlotte(Crop).jpg')
    rgb_img = cv.cvtColor(intial_img, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.title('initial color')
    plt.imshow(rgb_img)
    #rgb_img[:, :, 0] = 0 #Red
    rgb_img[:, :, 1] = 0 #Green
    rgb_img[:, :, 2] = 0 #Blue
    plt.figure()
    plt.title('initial color')
    plt.imshow(rgb_img)

    plt.show()



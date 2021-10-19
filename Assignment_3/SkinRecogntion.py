""" Skin recognition software utilizing RGB and LUV color-spaces """


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def skin_detection( src ):

    b = src[:,:,0].astype(np.int16)
    g = src[:,:,1].astype(np.int16)
    r = src[:,:,2].astype(np.int16)

    skin_mask =                                    \
          (r > 96) & (g > 40) & (b > 10)           \
        & ((src.max() - src.min()) > 15)           \
        & (np.abs(r-g) > 15) & (r > g) & (r > b)

    return src * skin_mask.reshape(skin_mask.shape[0], skin_mask.shape[1], 1)

def histogram_normalizer(img):
    length, width = img.shape
    num_pix = length * width
    pixel_dict = {}
    #populate pixel dictionary to then calculate histogram equilization
    for i in range(length):
        for j in range(width):
            if img[i,j] in pixel_dict:
                pixel_dict[img[i,j]] += 1
            else:
                pixel_dict.update({img[i,j] :1})
    for i in range(length):
        for j in range(width):
            img[i,j] = (pixel_dict[img[i,j]]/num_pix) * 255

    return img










if __name__ == "__main__":
    face_g = cv.imread('face_good-1.bmp', cv.IMREAD_COLOR)
    plt.figure()
    plt.title('good face')
    plt.imshow(cv.cvtColor(face_g, cv.COLOR_BGR2RGB))
    plt.show()

    skin_g = skin_detection(face_g)
    plt.figure()
    plt.title('good skin detect')
    plt.imshow(cv.cvtColor(skin_g, cv.COLOR_BGR2RGB))
    plt.show()

    face_b = cv.imread('face_dark-1.bmp', cv.IMREAD_COLOR)
    plt.figure()
    plt.title('bad face')
    plt.imshow(cv.cvtColor(face_b, cv.COLOR_BGR2RGB))
    plt.show()

    skin_b = skin_detection(face_b)
    plt.figure()
    plt.title('bad skin detect')
    plt.imshow(cv.cvtColor(skin_b, cv.COLOR_BGR2RGB))
    plt.show()

    luv = cv.cvtColor(face_b, cv.COLOR_BGR2Luv)
    l = luv[:, :, 0]
    luv[:, :, 0] = histogram_normalizer(l)
    hist = cv.cvtColor(luv, cv.COLOR_LUV2BGR)
    plt.figure()
    plt.title('histogram results')
    plt.imshow(cv.cvtColor(hist, cv.COLOR_BGR2RGB))
    plt.show()








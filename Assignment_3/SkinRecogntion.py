"""
Erik Sundblad
CS3150
10/19/2021
Skin recognition software utilizing RGB and LUV color-spaces in combination with
histographical analysis and thresholding
"""


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def skin_detection( src ):
    """ Skin detection in RGB colorspace, using known pixel value minimmums for skin
    as provided by A Survey on Pixel-Based Skin Color Detection Techniques """

    b = src[:,:,0].astype(np.int16)
    g = src[:,:,1].astype(np.int16)
    r = src[:,:,2].astype(np.int16)

    #clever boolean style array
    skin_mask =                                    \
          (r > 96) & (g > 40) & (b > 10)           \
        & ((src.max() - src.min()) > 15)           \
        & (np.abs(r-g) > 15) & (r > g) & (r > b)

    return src * skin_mask.reshape(skin_mask.shape[0], skin_mask.shape[1], 1)

def histogram_normalizer(img):
    """ Function to extract the histogram in list file and pass to local_min
    function to extract desired threshold(seperate spikes in luminocity frequency)"""
    length, width = img.shape
    hist = [0] * 256
    num_pix = length * width
    # loop through and aggregate histogram
    for i in range(length):
        for j in range(width):
            hist[img[i,j]] += 1

    #pass hist to local min function and reserve returned threshold
    thresh, deriv = find_local_min(hist)
    #print(thresh) -> Print out threshold for checking function result

    for i in range(length):
        for j in range(width):
            if img[i, j] > (thresh):
                img[i,j] = 0
    #Mask out any values greater than thresh value
    return img

def find_local_min( hist ):
    """ Function to return threshold of localized minimum
    (largest changes in count of pixels at specific lum values """

    kern = np.array(
            [2,0,0,0,
             2,0,0,0,
             2,0,0,0,
             2,0,0,0,
             1,0,0,0,
             1,0,0,0,
             1,0,0,0,
             1,0,0,0,
             -3,-3,-3,-3
             -3,-3,-3,-3
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,2
             ,0,0,0,2
             ,0,0,0,2
             ,0,0,0,2])
    #// theres a lot of 0's in there what will throw off
    #// the convolution
    hist[0] = 0
    deriv = np.convolve(hist, kern, mode='same')
    threshold = deriv.argmax()
    return threshold, deriv



if __name__ == "__main__":
    """ main code segment for facial recognition"""

    #Read in Good face display then apply skind detect and display result
    face_g = cv.imread('face_good-1.bmp', cv.IMREAD_COLOR)
    plt.figure()
    plt.title('good face')
    plt.imshow(cv.cvtColor(face_g, cv.COLOR_BGR2RGB))
    #plt.show()

    skin_g = skin_detection(face_g)
    plt.figure()
    plt.title('good skin detect')
    plt.imshow(cv.cvtColor(skin_g, cv.COLOR_BGR2RGB))
    #plt.show()

    #Read in Dark Face, display, apply skin filter, and display
    face_b = cv.imread('face_dark-1.bmp', cv.IMREAD_COLOR)
    plt.figure()
    plt.title('bad face')
    plt.imshow(cv.cvtColor(face_b, cv.COLOR_BGR2RGB))
    #plt.show()

    skin_b = skin_detection(face_b)
    plt.figure()
    plt.title('bad skin detect')
    plt.imshow(cv.cvtColor(skin_b, cv.COLOR_BGR2RGB))
    #plt.show()

    #convert filtered dark face to LUV color space and extract Luminance component
    luv = cv.cvtColor(skin_b, cv.COLOR_BGR2LUV)
    l = luv[:,:,0]
    #pass luminance component to histogram and assign the edited luminance back to luv
    luv[:,:,0] = histogram_normalizer(l).astype(np.uint8)

    #plt.figure()   #personal interest in what plotted hist would show
    #plt.title('histogram')
    #plt.imshow(luv)

    #Convert modified hist back to BRG then display as converted RGB image with
    #more accurate skin detection
    hist = cv.cvtColor(luv, cv.COLOR_LUV2BGR)
    plt.figure()
    plt.title('histogram results')
    plt.imshow(cv.cvtColor(hist, cv.COLOR_BGR2RGB))
    plt.show()









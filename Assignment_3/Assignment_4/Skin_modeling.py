"""
Erik Sundblad
CS3150
10/27/2021
Extending Skin detection utilizing morphology to improve results
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

def filter(pic):
    """ Generic Filter func takes picture to process, filter kernel, and filter name"""
    avg = np.full((11, 11), (1/(11**2)))
    filtered = cv.filter2D(pic, -1, avg)
    plt.figure()
    plt.title("Average Filter")
    plt.imshow(filtered, cmap='gray', vmin=0, vmax=255)
    return filtered

def face_morph(img):
    """Utilizes morphology to better define skin region"""
    kernel_square_open = np.ones((5, 5), np.uint8)  # square kernal of 1
    kernel_square_close = np.ones((15,15), np.uint8) #square kernal of 1

    kernel_cross = np.array(    # For testing out results with different shaped structs
        [[0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0]], np.uint8
    )
    kernel_circle = np.array(
        [[0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0]], np.uint8
    )
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_square_open)
    plt.figure()
    plt.title('opening morph')
    plt.imshow(cv.cvtColor(opening, cv.COLOR_BGR2RGB))

    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_square_close)
    plt.figure()
    plt.title('closing morph')
    plt.imshow(cv.cvtColor(closing, cv.COLOR_BGR2RGB))
    return closing




if __name__ == "__main__":
    """Main codeblock to perform dilation"""

    # Read in Dark Face, display, apply skin filter, and display
    face_b = cv.imread('face_dark.bmp', cv.IMREAD_COLOR)
    skin_b = skin_detection(face_b)

    # convert filtered dark face to LUV color space and extract Luminance component
    luv = cv.cvtColor(skin_b, cv.COLOR_BGR2LUV)
    l = luv[:, :, 0]

    # pass luminance component to histogram and assign the edited luminance back to luv
    luv[:, :, 0] = histogram_normalizer(l).astype(np.uint8)
    hist = cv.cvtColor(luv, cv.COLOR_LUV2BGR)
    plt.figure()
    plt.title('histogram results')
    plt.imshow(cv.cvtColor(hist, cv.COLOR_BGR2RGB))
    #created a filtered img after histographical correction to skin detection
    avg = filter(hist)

    #morph both the original and filtered images
    double = filter(face_morph(hist))
    face_morph(avg)
    plt.figure()
    plt.title('filter final')
    plt.imshow(cv.cvtColor(double, cv.COLOR_BGR2RGB))


    plt.show()
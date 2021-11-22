import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import random
 
#from skimage.measure import structural_similarity as ssim
#from sckit-image.measure import compare_ssim as ssim
#from skimage.metrics import mean_squared_error
#from skimage.metrics import peak_signal_noise_ratio
 
def add_sp_noise( src, percent):
    flat = src.ravel()
    length = len(flat)
    for ii in range(int(length * percent / 2)):
        index = int(random.random() * length)
        flat[index] = random.randint(0, 1)
    return flat.reshape(src.shape)

 
def make_g_noise( height, width , variance):
    sigmas = math.sqrt(variance)
    #noise = variance * np.random.randn(height,width)
    noise = sigmas * np.random.randn(height,width)
    return noise

def box_blur( src , k_size ):
    box_kern = np.ones((k_size,k_size))
    box_kern /= box_kern.sum()
    return cv.filter2D(src, -1, box_kern)

def median_blur( src , size ):
    src_padded = cv.copyMakeBorder(src
            , size, size, size, size, cv.BORDER_REPLICATE)
    rows = src.shape[0]
    cols = src.shape[1]
    if size % 2 == 0:
        size += 1
    dst = np.zeros(src.shape, dtype='float64')
    for ii in range(rows):
        for jj in range(cols):
            x = int(size//2) + jj
            y = int(size//2) + ii
            sub = src_padded[y:y+size, x:x+size]
            sub = np.sort(sub,0)
            sub = np.sort(sub,1)
            dst[ii,jj] = sub[size//2, size//2]

    return dst

def fft_lowpass( src ):
    #// make a mask that has a gaussian curve
    g_image = make_g_kern(src.shape[0]//2, 10)
    #// added a tiny offset so it doesn't overflow
    g_image = g_image * 1.0 / (g_image.max() +.0000001)

    #// save mask to disk for display  
    #cv.imwrite( 'g_image.jpg', (g_image * 255).astype(np.uint8))

    #// do a fourier transform on the source image
    fourier = np.fft.fft2(src)
    fourier_shifted = np.fft.fftshift(fourier)

    blurred = fourier_shifted * g_image

    blurred_unshifted = np.fft.ifftshift(blurred)
    inv_fourier = np.fft.ifft2(blurred_unshifted)
    return inv_fourier 


def SSIM(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def PSNR(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def MSE(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
     
    return mse
##############################################################################
#                           Main Program                                     #
##############################################################################
imA =cv.imread("./lena_g.bmp")
imA = cv.cvtColor(imA, cv.COLOR_BGR2GRAY)
length = imA.shape[0]
width = imA.shape[1]


mseV= MSE(imA,imA)
psnrV = PSNR(imA, imA)
ssimV = SSIM(imA,imA)

plt.figure()
plt.imshow(imA,cmap='gray')
plt.title("MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))


## Gaussian noise
g_noise = make_g_noise(length, width, 100)
imB= imA + g_noise  
mseV= MSE(imA,imB)
psnrV = PSNR(imA, imB)
ssimV = SSIM(imA,imB)
plt.figure()
plt.imshow(imB,cmap='gray')
plt.title("Gaussian: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))

## Uniform noise
u_noise = np.random.uniform(-20,20, (length, width))
imB= imA + u_noise  
mseV= MSE(imA,imB)
psnrV = PSNR(imA, imB)
ssimV = SSIM(imA,imB)
plt.figure()
plt.imshow(imB,cmap='gray')
plt.title("Uniform: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))

## contrast stregthing
mx=np.amax(imA)
mn=np.amin(imA)
imB=(imA-mn)/(mx-mn)*255
mseV= MSE(imA,imB)
psnrV = PSNR(imA, imB)
ssimV = SSIM(imA,imB)
plt.figure()
plt.imshow(imB,cmap='gray')
plt.title("Contrast enhance: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
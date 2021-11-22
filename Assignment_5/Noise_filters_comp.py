import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def SSIM(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def PSNR(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def MSE(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)

    return mse

def contrast_stretching(img, length, width):
    im2 = np.zeros((length, width))
    mx = np.amax(img)
    mn = np.amin(img)
    im2 = (img - mn) / (mx - mn) * 255
    return im2

def gamma_fix(img, length, width):
    im3 = np.zeros((length, width))
    mx = np.amax(img)
    mn = np.amin(img)
    imgN = img / 255
    im3 = imgN ** 2.2
    return im3


def add_sp_noise( img, percent):
    flat = img.ravel()
    length = len(flat)
    for ii in range(int(length * percent / 2)):
        index = int(random.random() * length)
        flat[index] = random.randint(0, 1)
    return flat.reshape(img.shape)


def median_blur( img , size ):
    img_padded = cv.copyMakeBorder(img
            , size, size, size, size, cv.BORDER_REPLICATE)
    rows = img.shape[0]
    cols = img.shape[1]
    if size % 2 == 0:
        size += 1
    dst = np.zeros(img.shape, dtype='float64')
    for ii in range(rows):
        for jj in range(cols):
            x = int(size//2) + jj
            y = int(size//2) + ii
            sub = img_padded[y:y+size, x:x+size]
            sub = np.sort(sub,0)
            sub = np.sort(sub,1)
            dst[ii,jj] = sub[size//2, size//2]

    return dst


def make_g_noise( height, width , variance):
    sigmas = math.sqrt(variance)
    noise = sigmas * np.random.randn(height, width)
    return noise


if __name__ == "__main__":
    origin = cv.imread("lena_g-1.bmp")
    origin = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
    length = origin.shape[0]
    width = origin.shape[1]
    plt.title('Original')
    plt.imshow(origin, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Median Blur Noise
    #medBlur = median_blur(origin, 21)
    #plt.title('Median Blur')
    #plt.imshow(medBlur, cmap='gray', vmin=0, vmax=255)
    #plt.show()

# Gaussian Noise
    gaus_noise = make_g_noise(length, width, 400)
    gauss = origin + gaus_noise
    plt.title('Gaussian Noise')
    plt.imshow(gauss, cmap='gray', vmin=0, vmax=255)
    plt.show()

# SnP Noise
    snp = add_sp_noise(origin, 0.35)
    plt.title('Salt n pepper')
    plt.imshow(snp, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Gamma Filter
    guass_gamma = gamma_fix(gauss, length, width)
    plt.title('Guass/Gamma')
    plt.imshow(guass_gamma, cmap='gray', vmin=0, vmax=1)
    plt.show()
    mseV = MSE(origin, guass_gamma)
    psnrV = PSNR(origin, guass_gamma)
    ssimV = SSIM(origin, guass_gamma)
    print("Gaussian: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
    snp_gamma = gamma_fix(snp, length, width)
    plt.title('SnP/Gamma')
    plt.imshow(snp_gamma, cmap='gray', vmin=0, vmax=1)
    plt.show()
    mseV = MSE(origin, snp_gamma)
    psnrV = PSNR(origin, snp_gamma)
    ssimV = SSIM(origin, snp_gamma)
    print("S and P: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))



# Contrast Stretching
    guass_contr = contrast_stretching(gauss, length, width)
    plt.title('Guass/Contrast')
    plt.imshow(guass_contr, cmap='gray', vmin=0, vmax=255)
    plt.show()
    mseV = MSE(origin, guass_contr)
    psnrV = PSNR(origin, guass_contr)
    ssimV = SSIM(origin, guass_contr)
    print("Gaussian: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
    snp_contr = contrast_stretching(snp, length, width)
    plt.title('SnP/Contrast')
    plt.imshow(snp_contr, cmap='gray', vmin=0, vmax=255)
    plt.show()
    mseV = MSE(origin, snp_contr)
    psnrV = PSNR(origin, snp_contr)
    ssimV = SSIM(origin, snp_contr)
    print("S and P: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))

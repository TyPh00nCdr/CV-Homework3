import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.signal import convolve2d
from scipy.stats import norm
from skimage.transform import pyramid_laplacian, pyramid_gaussian

I = imageio.imread('data/woman.png')
K_gauss = norm(loc=2).pdf(np.arange(5))  # loc = mean, offset to the left b/c of different origin
K_gauss = np.outer(K_gauss, K_gauss)
K_gauss = K_gauss / K_gauss.sum()


def gauss_subsample(im, depth):
    pyramid = [im]
    for _ in range(depth - 1):
        conv = convolve2d(pyramid[-1], K_gauss, mode='same')  # .astype('uint8')
        pyramid.append(conv[::2, ::2])
    return pyramid


def laplace_subsample(im, depth):
    return [(img - convolve2d(img, K_gauss, mode='same')) for img in
            gauss_subsample(im, depth)]


fix, axs = plt.subplots(4, 4)
for idx, im in enumerate(zip(gauss_subsample(I, 4),
                             pyramid_gaussian(I, max_layer=3, sigma=1),
                             laplace_subsample(I, 4),
                             pyramid_laplacian(I, max_layer=3, sigma=1))):
    axs[0, idx].imshow(im[0], cmap='gray')
    axs[1, idx].imshow(im[1], cmap='gray')
    axs[2, idx].imshow(im[2], cmap='gray')
    axs[3, idx].imshow(im[3], cmap='gray')
axs[0, 0].set_title('Own Gaussian')
axs[1, 0].set_title('skimage Gaussian')
axs[2, 0].set_title('Own Laplacian')
axs[3, 0].set_title('skimage Laplacian')
plt.show()

import cv2
import numpy as np


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel = sobely if orient is 'y' else sobelx
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return mag_binary


def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)

    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return dir_binary


def hls_thresh(image, thresh=(0, 255)):
    # HLS
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary


def get_binary(image):
    # Apply S-channel thresholding using HLS color space
    s_binary = hls_thresh(image, thresh=(160, 255))

    gray = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)

    # Choose a Sobel kernel size
    ksize = 25  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(40, 250))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(40, 250))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(50, 250))
    dir_binary = dir_thresh(gray, sobel_kernel=ksize, thresh=(np.pi / 4 * 0.6, np.pi / 4 * 1.5))

    combined = np.zeros_like(gray)
    combined_grad = np.zeros_like(gray)
    combined_grad[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[(s_binary == 1) | (combined_grad == 1)] = 1

    return combined


def main():
    pass


if __name__ == "__main__":
    main()

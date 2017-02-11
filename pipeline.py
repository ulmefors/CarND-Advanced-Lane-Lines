import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import camera_calibrator
import image_undistorter
import image_binary_thresholder
import perspective_transformer


def pipeline(img_file_name):

    # Camera calibration
    mtx, dist = camera_calibrator.get_camera_calibration_matrix()

    # Get undistorted image
    undistorted_gray = image_undistorter.get_undistorted_image(img_file_name, mtx, dist, gray_scale=True)

    # Thresholded image
    binary_thresholded = image_binary_thresholder.get_binary(undistorted_gray)

    # Perspective transform
    top_view = perspective_transformer.get_transformed_perspective(binary_thresholded)

    plt.imshow(top_view, cmap='gray')
    plt.show()

    return None


def main():
    fname = 'test_images/straight_lines2.jpg'
    pipeline(fname)


if __name__ == "__main__":
    main()

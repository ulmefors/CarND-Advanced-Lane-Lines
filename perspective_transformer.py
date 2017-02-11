import cv2
import numpy as np


def get_transformed_perspective(image):

    y, x = image.shape # Image size
    x_mid = x/2

    # Trapezoid shape
    x_top = 0.12 * x  # Width at top border
    x_bottom = 0.68 * x  # Width at bottom border
    y_top = 0.64 * y  # Top border position
    y_bottom = 0.92 * y  # Bottom border position

    top_left = [x_mid - x_top/2, y_top]
    top_right = [x_mid + x_top/2, y_top]
    bottom_left = [x_mid - x_bottom/2, y_bottom]
    bottom_right = [x_mid + x_bottom/2*1.02, y_bottom]

    src = np.float32([top_left, top_right, bottom_left, bottom_right])
    offset = 0.25 * min(y, x)
    dst = np.float32([[offset, offset], [x - offset, offset], [offset, y], [x - offset, y]])

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Calculate top-down image using perspective transform matrix
    warped = cv2.warpPerspective(image, M, (x, y), flags=cv2.INTER_LINEAR)

    return warped


def main():
    pass


if __name__ == "__main__":
    main()

import cv2
import numpy as np


def get_transformed_perspective(image):

    y, x = image.shape # Image size
    x_mid = x/2

    # Trapezoid shape
    x_top = 0.12 * x  # Width at top border
    x_bottom = 0.54 * x  # Width at bottom border
    y_top = 0.66 * y  # Top border position
    y_bottom = 0.94 * y  # Bottom border position

    top_left = [x_mid - x_top / 2, y_top]
    top_right = [x_mid + x_top / 2, y_top]
    bottom_left = [x_mid - x_bottom / 2, y_bottom]
    bottom_right = [x_mid + x_bottom / 2, y_bottom]

    src = np.float32([top_left, top_right, bottom_left, bottom_right])
    offset = 0.40 * min(y, x)
    dst = np.float32([[offset, offset], [x - offset, offset], [offset, y], [x - offset, y]])

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)

    # Calculate top-down image using perspective transform matrix
    warped = cv2.warpPerspective(image, M, (x, y), flags=cv2.INTER_LINEAR)
    #unwarped = cv2.warpPerspective(warped, Minv, (x, y), flags=cv2.INTER_LINEAR)

    return warped


def main():
    pass


if __name__ == "__main__":
    main()

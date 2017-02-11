import cv2


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


# Assumes RGB
def get_undistorted_image(image, mtx, dist, gray_scale=False):
    img_undistorted = undistort(image, mtx, dist)
    if gray_scale:
        return cv2.cvtColor(img_undistorted, cv2.COLOR_RGB2GRAY)
    else:
        return img_undistorted


def get_undistorted_image_from_file(fname, mtx, dist, gray_scale=False):
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return get_undistorted_image(image, mtx, dist, gray_scale=gray_scale)


def main():
    pass


if __name__ == "__main__":
    main()

import cv2


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def get_undistorted_image(fname, mtx, dist, gray_scale=False):
    img_orig = cv2.imread(fname)
    img_undist = undistort(img_orig, mtx, dist)
    color_profile = cv2.COLOR_BGR2GRAY if gray_scale else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(img_undist, color_profile)


def main():
    pass


if __name__ == "__main__":
    main()

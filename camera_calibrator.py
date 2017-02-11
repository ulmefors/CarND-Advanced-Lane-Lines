import numpy as np
import cv2
import glob
import pickle

camera_cal_fname = "camera_cal/camera_cal.p"


def calibrate():
    print('Starting camera calibration')
    images = []

    img, gray, img_corners, img_original = None, None, None, None

    objpoints, imgpoints = [], []

    image_files = glob.glob('camera_cal/calibration*.jpg')

    for fname in image_files:

        nx, ny = 9, 6

        if 'calibration1.jpg' in fname:
            nx, ny = 9, 5
        if 'calibration4.jpg' in fname:
            nx, ny = 6, 5
        if 'calibration5.jpg' in fname:
            nx, ny = 7, 6

        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # Save example image and corresponding identified corners
        if 'calibration2.jpg' in fname:
            img_original = img.copy()
            img_corners = img.copy()
            img_corners = cv2.drawChessboardCorners(img_corners, (nx, ny), corners, ret)
            cv2.imwrite('output_images/chessboard_original.jpg', img_original)
            cv2.imwrite('output_images/chessboard_corners.jpg', img_corners)

        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            images.append(img)
        else:
            print('Did not find corners', fname, (nx, ny))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    camera_calibration = {'mtx': mtx, 'dist': dist}
    pickle.dump(camera_calibration, open(camera_cal_fname, "wb"))

    print('Camera calibrated using {0} images'.format(np.array(images).shape[0]))


def get_camera_calibration_matrix():
    try:
        camera_calibration = pickle.load(open(camera_cal_fname, 'rb'))
    except FileNotFoundError:
        calibrate()
        camera_calibration = pickle.load(open(camera_cal_fname, 'rb'))

    mtx = camera_calibration['mtx']
    dist = camera_calibration['dist']
    return mtx, dist


def main():
    calibrate()


if __name__ == "__main__":
    main()

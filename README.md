Advanced Lane Finding
---

This aim of this project is to identify road lanes using a suite of computer vision techniques.
The steps undertaken are:
* Calculate camera calibration matrix
* Correct image distortion
* Create thresholded binary image
* Apply perspective transform to obtain "top view" ("birds-eye view")
* Detect lanes
* Determine curvature and vehicle position
* Apply lane pixels on original input image
* Visualize image with corresponding lanes and curvature/position values


[//]: # (Image References)

[image1]: ./output_images/chessboard_original.jpg "Chessboard"
[image2]: ./output_images/chessboard_corners.jpg "Chessboard corners"
[image3]: ./output_images/chessboard_undistort.jpg "Chessboard undistorted"
[image4]: ./test_images/test6.jpg 
[image5]: ./output_images/test6.jpg
[image6]: ./output_images/calibration12_original.jpg
[image7]: ./output_images/calibration12_perspective_transform.jpg

### Camera calibration
The first step is to compute the camera calibration matrix. This is done by identifying corners in chessboard images. The grid of chessboard corners must consist of parallell lines in the real world object.
The camera distortion can thus be calculated by comparing real world straight lines to the curves in the image.
The corners are found using OpenCV taking a gray-scale image and the corner grid shape as inputs.
    
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
The identified corners are visualized using `cv2.drawChessboardCorners(img, (nx, ny), corners, ret)`.

Original     | Corners        | Undistorted
:----------------------:|:-------------------------:|:------:
![alt text][image1]     |  ![alt text][image2]      | ![alt text][image3] 

20 images were used to obtain calibration data. The corner grid coordinates and the corresponding pixel positions were saved for use during calibration. 
 
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objpoints.append(objp)
    imgpoints.append(corners)

### Distortion correction
With `objpoints` and `imgpoints` defined thanks to the calibration images, it is possible to calculate the camera calibration matrix.
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
 The camera calibration matrix `mtx` is used to undistort images.
    
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

A comparison is shown with original and resulting image after distortion correction. Changes are easily noticed at the top left corner.

Original                | Distortion correction
:----------------------:|:-------------------------:
![alt text][image4]     |  ![alt text][image5]

### Thresholded binary image

### Perspective transform
All identified chessboard corners are stored in `corners` and the four outer corners (top-left, top-right, bottom-left, bottom-right) are selected.

    src = np.float32([corners[0][0], corners[nx-1][0], corners[nx*(ny-1)][0], corners[nx*ny-1][0]])

The destination positions are chosen as corners positioned with a distance `[offset, offset]` from image frame corners where `[x, y]` are image width, height respectively.   

    dst = np.float32([[offset, offset],[x-offset, offset],[offset, y-offset],[x-offset, y-offset]])

The perspective transform matrix `M` is calculated and used to transform camera perspective.

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_undistorted, M, (x, y), flags=cv2.INTER_LINEAR)


Original                | Perspective transform
:----------------------:|:-------------------------:
![alt text][image6]     |  ![alt text][image7]

### Lane detection

### Curvature radius and vehicle position
 
### Lane pixel overlay on original image

### Visualization of lanes, curvature and position


The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

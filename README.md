Advanced Lane Finding
---

This aim of this project is to identify road lanes using a suite of computer vision techniques.
The steps undertaken are:
1. Calculate camera calibration matrix
2. Correct image distortion
3. Create thresholded binary image
4. Apply perspective transform to obtain "top view" ("birds-eye view")
5. Detect lanes
6. Determine curvature
7. Determin vehicle position
8. Apply lane pixels on original input image
9. Visualize image with corresponding lanes and curvature/position values

Main script is `pipeline.py`.

Output video is `project_video_lane_detection.mp4`.

[//]: # (Image References)
[image1]: ./output_images/chessboard_original.jpg "Chessboard"
[image2]: ./output_images/chessboard_corners.jpg "Chessboard corners"
[image3]: ./output_images/chessboard_undistort.jpg "Chessboard undistorted"
[image4]: ./test_images/test6.jpg 
[image5]: ./output_images/undistorted_test6.jpg
[image6]: ./output_images/undistorted_straight_lines1.jpg
[image7]: ./output_images/undistorted_straight_lines1_binary_thresholded.jpg
[image8]: ./output_images/calibration12_original.jpg
[image9]: ./output_images/calibration12_perspective_transform.jpg
[image10]: ./test_images/straight_lines2.jpg
[image11]: ./output_images/straight_lines2_masked.jpg
[image12]: ./output_images/straight_lines2_masked_transformed.jpg
[image13]: ./output_images/top_view_histogram.png
[image14]: ./output_images/top_view_lane_curves.png
[image15]: ./output_images/color_zone_warped.jpg
[image16]: ./output_images/color_zone_unwarped.jpg
[image17]: ./output_images/color_zone_lane_overlay.jpg
[image18]: ./output_images/curvature_vehicle_position.jpg
[video1]:  ./project_video_lane_detection.mp4 "Video"

### 1. Camera calibration
The first step is to compute the camera calibration matrix. This is done by identifying corners in chessboard images. The grid of chessboard corners must consist of parallell lines in the real world object.
The camera distortion can thus be calculated by comparing real world straight lines to the curves in the image.
The corners are found using OpenCV taking a gray-scale image `gray` and the corner grid shape `(nx, ny)` as inputs.
    
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
The identified corners are visualized using `cv2.drawChessboardCorners(img, (nx, ny), corners, ret)`. The final outcome of camera calibration will be an undistorted image with straight lines as presented below to the right.

Original     | Corners        | Undistorted
:----------------------:|:-------------------------:|:------:
![alt text][image1]     |  ![alt text][image2]      | ![alt text][image3] 

20 images were used to obtain calibration data. The corner grid coordinates and the corresponding pixel positions were saved for use during calibration. 
 
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objpoints.append(objp)
    imgpoints.append(corners)

File: `camera_calibration.py`

### 2. Distortion correction
With `objpoints` and `imgpoints` defined thanks to the calibration images, it is possible to calculate the camera calibration matrix.
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
 The camera calibration matrix `mtx` is used to undistort images.
    
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

A comparison is shown with original and resulting image after distortion correction. Changes are easily noticed at the top left corner.

Original                | Distortion correction
:----------------------:|:-------------------------:
![alt text][image4]     |  ![alt text][image5]

File: `image_undistorter.py`

### 3. Thresholded binary image
In order to efficiently detect lane lines we employ a combination of gradient filters based on the Sobel operator using a gray-scale image as input.
The Sobel operators output gradients in the x-direction and y-direction respectively.
The `ksize` parameter specifies the kernel size with larger values smoothing the image and thereby filtering out small high contrast features that are not lane lines.
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
The x- and y- gradients are also combined to find gradient magnitude (vector sum, length of hypotenuse) and gradient direction.
 
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_direction = np.arctan2(sobely, sobelx)
    
Binary thresholded images for the four individual gradients are created by evaluating whether the gradient values fall between defined minimum and maximum bounds.

    grad_binary = np.zeros_like(sobel)
    grad_binary[(sobel >= thresh_min) & (sobel <= thresh_max)] = 1
    
In the final stage we combine all four gradients by activating pixels that have been identified in the separate filters (x-gradient, y-gradient, magnitude, direction).

    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

Color filtering was performed after converting the color image into HLS color space and applying thresholds to the S-channel.

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    
The Sobel gradient thresholded image was combined with the color filter to create the final result. The resulting binary thresholded image successfully identifies the lane lines and filters out the majority of unwanted edges.

Undistorted                | Thresholded binary
:----------------------:|:-------------------------:
![alt text][image6]     |  ![alt text][image7]

File: `image_binary_thresholder.py`

### 4. Perspective transform
All identified chessboard corners are stored in `corners` and the four outer corners (top-left, top-right, bottom-left, bottom-right) are selected.

    src = np.float32([corners[0][0], corners[nx-1][0], corners[nx*(ny-1)][0], corners[nx*ny-1][0]])

The destination positions are chosen as corners positioned with a distance `[offset, offset]` from image frame corners where `[x, y]` are image width, height respectively.   

    dst = np.float32([[offset, offset],[x-offset, offset],[offset, y-offset],[x-offset, y-offset]])

The perspective transform matrix `M` is calculated and used to transform camera perspective.

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_undistorted, M, (x, y), flags=cv2.INTER_LINEAR)


Original                | Perspective transform
:----------------------:|:-------------------------:
![alt text][image8]     |  ![alt text][image9]

Using an input image with straight lanes we define a trapezoid that encompasses the lanes of interest but ignores the surroundings.
The area is transformed into a rectangle that will be used for lane detection.

Original                | Masked                    | Transformed
:----------------------:|:-------------------------:|:------:
![alt text][image10]    |  ![alt text][image11]     | ![alt text][image12] 

File: `perspective_transformer.py`

### 5. Lane detection
The binary thresholded top-view image is processed in the lane detection algorithm.
A histogram is drawn at the image base counting the frequency of lane pixels for each column (y-direction).
Two peaks are identified, corresponding to the two lanes. 

Frequency histogram     | Fit quadratic curve 
:----------------------:|:-------------------------:
![alt text][image13]    |  ![alt text][image14]

Starting from these two peaks a window rectangle is drawn with a pre-defined height, splitting the height in nine rows.
The position of the non-zero (lane pixels) within this window will determine the horizontal (x) position for the following window.
Windows are stacked vertically until reaching the top of the image and a quadratic curve is fit through all non-zero pixels (identified as components of the lane) that lie within the window bounds.

The x- and y- values of the non-zero pixels for each lane (left: red and right: blue) are used to determine the coefficients of the polynomial fit. We use a quadratic (2-order) curve.
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
The polynomial functions are defined as _x_ = _f(y)_ since _x_ can be uniquely defined for each value of _y_ in `ploty`.
The general formulae thus become _x_ = _f(y)_ = _Ay<sup>2</sup> + By + C_ with A, B, C corresponding to the values in `left_fit` and `right_fit` for each lane.
    
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

File: `lane_detector.py`

### 6. Curvature radius
We fit a second degree polynomial to the lane lines _x_ = _f(y)_ = _Ay<sup>2</sup> + By + C_. The curvature radius can be calculated as R<sub>curve</sub> = (1 + (2Ay + B)<sup>2</sup>)<sup>3/2</sup> / |2A| .
In order to obtain correct curvature values we translate pixel distance to real world coordinates using coefficients.
The full image height (720 px) corresponds to roughly 30 m whereas the 3.7 m lane width is accommodated in 840 px width. Corresponding _A, B, C_ coefficients are calculated for the real-world values.

    ym_per_pix = 30  / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 840  # meters per pixel in x dimension
    left_fit_m = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_m = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

The curvature is calculated using the real-world coefficients for _A_ and _B_. The expression is evaluated at the vehicle position which is at 720 px, corresponding to 30 m.
Each lane results in its own curvature. The average of the two is used for the visualization.

File: `lane_detector.py`

### 7. Vehicle position
The vehicle position with respect to the lane lines was found by comparing the image center to the lane positions.

    image_centerx = image_width / 2
    lane_centerx = np.mean((leftx[-1], rightx[-1]))
    vehicle_position_px = image_centerx - lane_centerx
    
If the image center has a larger x-value than lane center we know that the vehicle position is closer to the right lane.
This offset position can be converted into cm real-world distance.
    
    vehicle_position_cm = np.int32(vehicle_position_px * xm_per_pix * 100)
    
File: `lane_detector.py`

### 8. Lane pixel overlay on original image
The zone between the two fitted polynomial curves is painted before being unwarped to the original perspective.
The original perspective is obtained by using the matrix of the inverse transformation which is found by the same process as described above but with swapped order of `src` and `dst`.

    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped_color_zone = cv2.warpPerspective(warped_color_zone, Minv, (x, y), flags=cv2.INTER_LINEAR)

The unwarped color zone is added to the original image.

    overlay_result = cv2.addWeighted(image, 1, unwarped_color_zone, 0.3, 0)

Warped color zone       | Unwarped color zone       | Overlay result
:----------------------:|:-------------------------:|:------:
![alt text][image15]    |  ![alt text][image16]     | ![alt text][image17] 

File: `perspective_transformer.py`, `pipeline.py`

### 9. Visualization of curvature and position
The values for curvature radius and vehicle position were averaged over a number of frames and printed on top of the image. 

    radii.append(curvature)
    if len(radii) > smooth_no:
        radii.pop(0)
    curvature = np.int(np.mean(radii))
    cv2.putText(image, 'Curvature radius: {0} m'.format(curvature), (10, 60),
                font, fontScale, color_white, thickness, cv2.LINE_AA)
                
File: `text_writer.py`

## Final result

[![alt_text][image18]](./project_video_lane_detection.mp4)

## Discussion
Lane detection works reasonable well on `project_video.mp4` even if there are frames where the polynomial fit outputs somewhat too strong curvature.
These instances can happen when high gradient areas (shadows or marks) are found in the lane which can result in erroneous end-point of the fitted curve.
Performance on more challenging videos is unsatisfactory. In order to achieve better results it will be necessary to implement smoothing using previous frames instead of relying solely on a single frame for detection.
 
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
from moviepy.editor import VideoFileClip


def pipeline(image):

    # Camera calibration
    mtx, dist = camera_calibrator.get_camera_calibration_matrix()

    # Get undistorted image
    undistorted_color = image_undistorter.get_undistorted_image(image, mtx, dist, gray_scale=False)

    # Thresholded image
    binary_thresholded = image_binary_thresholder.get_binary(undistorted_color)

    # Perspective transform
    top_view = perspective_transformer.get_transformed_perspective(binary_thresholded)

    # Detect lanes




    result = top_view
    result *= 255

    # If gray scale convert to triple channel format
    if len(result.shape) == 2:
        result = np.dstack((result,)*3)

    return result


def main():
    #fname = 'test_images/test1.jpg'
    #image = cv2.imread(fname)
    #result = pipeline(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #plt.imshow(result)
    #plt.show()



    #TODO: Create output folder

    file = 'project_video_super_short'
    output = 'output_videos/' + file + '.mp4'
    clip1 = VideoFileClip(file + '.mp4')
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(output, audio=False)




if __name__ == "__main__":
    main()

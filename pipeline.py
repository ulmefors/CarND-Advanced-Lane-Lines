import numpy as np
import cv2
import matplotlib.pyplot as plt
import camera_calibrator
import image_undistorter
import image_binary_thresholder
import perspective_transformer
import lane_detector
import text_writer
import os
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
    color_zone_warp, curvature, vehicle_pos_cm = lane_detector.detect_lanes(top_view)

    # Warp the color zone back to original image space using inverse perspective matrix (Minv)
    color_zone = perspective_transformer.get_original_perspective(color_zone_warp)

    # Add the color zone to the original image
    result = cv2.addWeighted(image, 1, color_zone, 0.3, 0)

    # Add text with curvature and vehicle position
    result = text_writer.add_text(result, curvature, vehicle_pos_cm)

    # If gray scale convert to triple channel format
    if len(result.shape) == 2:
        result = np.dstack((result,)*3)

    # If binary image, scale to full 8-bit values
    if np.max(result) <= 1:
        result *= 255

    return result


def main():

    # Run video or single image
    video = True

    # Specify inputs and outputs
    image_file = 'test_images/test5.jpg'
    video_file = 'project_video_short'
    video_output_dir = 'output_videos/'

    # Create output folder if missing
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    if video:
        # Create video with lane zone overlay
        output = video_output_dir + video_file + '.mp4'
        input_clip = VideoFileClip(video_file + '.mp4')
        output_clip = input_clip.fl_image(pipeline)
        output_clip.write_videofile(output, audio=False)
    else:
        # Plot image with detected lanes
        image = cv2.imread(image_file)
        result = pipeline(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.imshow(result)
        plt.show()


if __name__ == "__main__":
    main()

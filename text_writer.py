import numpy as np
import cv2

radii = []
positions = []
smooth_no = 50
font = cv2.FONT_HERSHEY_SIMPLEX
color_white = (255, 255, 255)
fontScale = 2
thickness = 4


def add_text(image, curvature, vehicle_pos_cm):

    # Maintain a list of latest values in order to create moving averages
    radii.append(curvature)
    if len(radii) > smooth_no:
        radii.pop(0)

    positions.append(vehicle_pos_cm)
    if len(positions) > smooth_no:
        positions.pop(0)

    # Calculate moving averages for curvature radius and offset position
    curvature = np.int(np.mean(radii))
    vehicle_pos_cm = np.int(np.mean(positions))

    # Print curvature radius and vehicle right/left offset position in lane
    cv2.putText(image, 'Curvature radius: {0} m'.format(curvature), (20, 70),
                font, fontScale, color_white, thickness, cv2.LINE_AA)
    cv2.putText(image, 'Vehicle position: {0} cm'.format(vehicle_pos_cm), (20, 140),
                font, fontScale, color_white, thickness, cv2.LINE_AA)

    return image


def main():
    pass


if __name__ == "__main__":
    main()

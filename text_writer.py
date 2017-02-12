import numpy as np
import cv2

radii = []
positions = []
smooth_no = 50


def add_text(image, curvature, vehicle_pos_cm):

    radii.append(curvature)
    if len(radii) > smooth_no:
        radii.pop(0)

    positions.append(vehicle_pos_cm)
    if len(positions) > smooth_no:
        positions.pop(0)

    curvature = np.int(np.mean(radii))
    vehicle_pos_cm = np.int(np.mean(positions))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Curvature radius: {0} m'.format(curvature), (10, 60), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'Vehicle position: {0} cm'.format(vehicle_pos_cm), (10, 160), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def main():
    pass


if __name__ == "__main__":
    main()

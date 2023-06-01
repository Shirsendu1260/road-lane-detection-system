import cv2
import numpy as np


# Function to display lane lines in original image
def display_lines(img, lines, thickness):
    # Create a mask image of all black pixels of same dimension as the road image
    line_image = np.zeros_like(img)

    # 'lines' is 3-D array
    # Check if it had detected any lines
    if lines is not None:
        # Take each line we are iterating through and draw it onto the blank image
        for x1, y1, x2, y2 in lines:
            # print(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), thickness)

    return line_image

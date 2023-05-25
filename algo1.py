import sys
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
from vehicle_detector import vehicle_detector as vd


# Function that takes in an image. It then converts it to grayscale
# Then converts that grayscale image to blurred image to remove noise
# Finally edges are detected using Canny Edge detector on that blurred image
def edge_detector(img, low_threshold, high_threshold):
    if img is None:
        # Release the resources
        capture.release()
        output.release()
        cv2.destroyAllWindows()

    # Grayscale conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    kernel = 5
    img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), 0)

    # Edge detection using Canny Edge detector
    img_canny = cv2.Canny(img_blur, low_threshold, high_threshold)

    return img_canny


# Function that masks every portion of the image except the portion where the lane is visible
def region_of_interest(img):
    height = img.shape[0]

    # The region where the lane is visible
    vertices = np.array([
        [[145, height], [436, 329], [527, 329], [913, height]]
    ])
    # print(pts)

    # Create a mask image of all black pixels of same dimension as the road image
    mask = np.zeros_like(img)

    # Fill the mask with the polygon
    cv2.fillPoly(mask, vertices, 255)

    # Mask the original image (canny image) with previously created mask
    # Ultimately masking the image to only show the region of interest traced by the polygonal contour of the mask
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img


# Function to display lane lines in original image
def display_lines(img, lines):
    # Create a mask image of all black pixels of same dimension as the road image
    line_image = np.zeros_like(img)

    # 'lines' is 3-D array
    # Check if it had detected any lines
    if lines is not None:
        # Take each line we are iterating through and draw it onto the blank image
        for x1, y1, x2, y2 in lines:
            # print(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 7)

    return line_image


# Function to return coordinates of a straight line
def get_coordinates(img, line_parameters):
    m = line_parameters[0]  # Slope
    b = line_parameters[1]  # y-intersect value

    # print(image.shape)

    y1 = img.shape[0]
    y2 = int(y1 * (3.25/5))

    # Since, y = mx + b
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)

    # Return the derived coordinates
    return np.array([x1, y1, x2, y2])


# Function to average multiple lines into a single line for each side of road lane
def average_lane_lines(img, lines):
    # Will contain slope and y-intersect value of the lines on the left
    left_fit = []
    # Will contain slope and y-intersect value of the lines on the right
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # polyfit() will fit a 1st degree polynomial (which y = mx + b) to x and y points
        # And return a vector of coefficients
        # Which describes the slope (m) and y intersect value (b) for the line
        # Returns an array for each line (which is [slope, y-intersect value])
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)

        slope = parameters[0]
        y_intersect = parameters[1]

        # Check slope of that line which corresponds to a line on left side or right side
        # Lines on the left and the right have a negative and a positive slope respectively
        if slope < 0:
            left_fit.append((slope, y_intersect))
        else:
            right_fit.append((slope, y_intersect))

    # print(left_fit)
    # print(right_fit)

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    # print("Left Fit Average:", left_fit_avg)
    # print("Right Fit Average:", right_fit_avg)

    # Constructing the left and the right line
    left_line = get_coordinates(img, left_fit_avg)
    right_line = get_coordinates(img, right_fit_avg)

    # Return the lines
    return np.array([left_line, right_line])


# The main algorithm code
def detect_lanes(frame):
    # Edge detection
    image_canny = edge_detector(frame, 50, 150)

    # # Display the image
    # plt.imshow(image_canny)
    # plt.show()

    # Define region of interest
    image_roi = region_of_interest(image_canny)

    # Detect lines in cropped gradient image and display that lane lines in the original image
    rho = 1
    theta = 1 * (np.pi/180)
    lines = cv2.HoughLinesP(image_roi, rho, theta, 30, np.array(
        []), minLineLength=90, maxLineGap=200)

    # Average multiple lines into a single line for each side of road lane
    # Ultimately there will be only 2 lines for 2 lane lines
    avg_lines = average_lane_lines(frame, lines)
    image_lines = display_lines(frame, avg_lines)
    # image_lines = display_lines(image_copy, lines)

    # Blend 'image_lines' with the original image
    alpha = 0.8
    beta = 1
    gamma = 0
    image_blend = cv2.addWeighted(frame, alpha, image_lines, beta, gamma)

    return image_blend


# # Test code
# if __name__ == "__main__":
#     # Read a footage
#     image = cv2.imread('test_files\\test_images\\test8.jpg')

#     # Copy of original footage
#     image_copy = np.copy(image)

#     # Edge detection
#     image_canny = edge_detector(image_copy, 50, 150)

#     # # Display the image
#     # plt.imshow(image_canny)
#     # plt.show()

#     # Define region of interest
#     image_roi = region_of_interest(image_canny)

#     # Detect lines in cropped gradient image and display that lane lines in the original image
#     rho = 1  # Precision of 1 pixels
#     # Precision of 1 degree i.e. 1 degree = pi/180 radians
#     theta = 1 * (np.pi/180)
#     # Threshold = 40
#     lines = cv2.HoughLinesP(image_roi, rho, theta, 40, np.array(
#         []), minLineLength=30, maxLineGap=200)

#     # Average multiple lines into a single line for each side of road lane
#     # Ultimately there will be only 2 lines for 2 lane lines
#     avg_lines = average_lane_lines(image_copy, lines)
#     image_lines = display_lines(image_copy, avg_lines)
#     # image_lines = display_lines(image_copy, lines)

#     # Blend 'image_lines' with the original image
#     alpha = 0.8
#     beta = 1
#     gamma = 0
#     image_blend = cv2.addWeighted(image_copy, alpha, image_lines, beta, gamma)

#     # Display the image
#     cv2.imshow("Result", image_blend)
#     cv2.waitKey(0)


# Driver code
if __name__ == "__main__":
    # Get the input video filename as command line argument
    if len(sys.argv) != 2:
        print("Usage: python algo1.py input.mp4")
        sys.exit(1)

    # Obtain the input path
    input_path = os.path.join("test_videos", sys.argv[1])

    # Extract the input video filename without extension
    in_filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # Create output video filename
    out_filename = in_filename + "_out.mp4"

    # set output folder path
    output_path = os.path.join("rld_pht_out", out_filename)

    # Define desired width and height for resizing
    w = 600
    h = 338

    # Read a video footage with the desired width and height
    capture = cv2.VideoCapture(input_path)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # Create a video writer object and set fps, width, height, codec for it
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, codec, fps, (frame_w, frame_h))

    # Set the window resolution explicitly
    cv2.namedWindow(
        "Road Lane Detection using Probabilistic Hough Transform", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "Road Lane Detection using Probabilistic Hough Transform", w*2, h)

    while True:
        # Read each frame in the video
        success, input_frame = capture.read()

        if not success:
            # If unable to read the frame
            break

        # Apply the main algorithm to the input video frame
        output_frame = detect_lanes(input_frame)

        # Apply Vehicle Detection
        output_frame = vd.vehicle_detector(output_frame)

        # Write the output frame
        output.write(output_frame)

        # Add labels to the frames
        cv2.putText(input_frame, "Before", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(output_frame, "After", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the input and the output frame side by side
        final_frame = cv2.hconcat([input_frame, output_frame])
        cv2.imshow(
            "Road Lane Detection using Probabilistic Hough Transform", final_frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    capture.release()
    output.release()
    cv2.destroyAllWindows()

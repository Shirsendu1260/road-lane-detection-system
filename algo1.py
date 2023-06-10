import sys
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
# import time
from utilities.edge_detector import edge_detector
from utilities.display_lines import display_lines
from utilities.get_lane_lines import average_lane_lines
from vehicle_detector import vehicle_detector as vd


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


# # Display the image
    # plt.imshow(image_canny)
    # plt.show()


# # Test code
# if __name__ == "__main__":
#     # Read a footage
#     image = cv2.imread(r'test_images\test8.jpg')

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
    output_path = os.path.join("algo1_out", out_filename)

    # Define desired width and height for resizing
    w = 500
    h = 282

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
        "Road Lane Detection (Without Distortion Correction & Bird's-eye View)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "Road Lane Detection (Without Distortion Correction & Bird's-eye View)", w*2, h*2)

    while True:
        # Read each frame in the video
        success, input_frame = capture.read()

        if not success:
            # If unable to read the frame
            print("END!")
            break

        # start = time.time()

        ##### MAIN ALGORITHM STARTS #####

        # Edge detection
        image_canny = edge_detector(input_frame, 50, 150)

        # Define region of interest
        image_roi = region_of_interest(image_canny)

        # Detect lines in cropped gradient image and display that lane lines in the original image
        rho = 1
        theta = 1 * (np.pi/180)
        lines = cv2.HoughLinesP(image_roi, rho, theta, 30, np.array(
            []), minLineLength=90, maxLineGap=200)

        # Average multiple lines into a single line for each side of road lane
        # Ultimately there will be only 2 lines for 2 lane lines
        avg_lines = average_lane_lines(input_frame, lines)
        image_lines = display_lines(input_frame, avg_lines, 7)
        # image_lines = display_lines(image_copy, lines)

        # Blend 'image_lines' with the original image
        alpha = 0.8
        beta = 1
        gamma = 0
        output_frame = cv2.addWeighted(
            input_frame, alpha, image_lines, beta, gamma)

        ##### MAIN ALGORITHM ENDS #####

        # total_time = time.time() - start

        # print("{:.2f}".format(total_time * 1000))

        # Apply Vehicle Detection
        output_frame = vd.vehicle_detector(output_frame)

        # Write the output frame
        output.write(output_frame)

        # Add labels to the frames
        cv2.putText(input_frame, "Input Frame", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_canny, "Edge Detected Frame", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_lines, "Cropped Gradient Frame with detected lanes", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(output_frame, "Final Output", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frames
        concate_frame_1 = cv2.hconcat(
            [input_frame, cv2.cvtColor(image_canny, cv2.COLOR_GRAY2BGR)])
        concate_frame_2 = cv2.hconcat(
            [image_lines, output_frame])
        final_frame = cv2.vconcat([concate_frame_1, concate_frame_2])
        cv2.imshow(
            "Road Lane Detection (Without Distortion Correction & Bird's-eye View)", final_frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    capture.release()
    output.release()
    cv2.destroyAllWindows()

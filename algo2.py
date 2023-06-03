import sys
import cv2
import numpy as np
import os
from utilities import camera_calibrator as cc
from utilities.sliding_window import thresh_binary, sliding_window
from vehicle_detector import vehicle_detector as vd


# The points from the region where the lane is visible
source_points = np.float32([(387, 370), (112, 540), (900, 540), (540, 370)])

# The points to display the source region from top-view
destination_points = np.float32([(75, 0), (75, 540), (825, 540), (680, 0)])

# Create perspective region of top-view where lane lines look parallel
top_view = cv2.getPerspectiveTransform(source_points, destination_points)

# Create perspective region of front-view where lane lines are converging
front_view = cv2.getPerspectiveTransform(destination_points, source_points)


# Function that converts front-view image to top-view image
def front_to_top(img, width, height):
    return cv2.warpPerspective(img, top_view, (width, height), flags=cv2.INTER_LINEAR)


# Function that converts top-view image to front-view image
def top_to_front(img, width, height):
    return cv2.warpPerspective(img, front_view, (width, height), flags=cv2.INTER_LINEAR)


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
    output_path = os.path.join("algo2_out", out_filename)

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
        "Road Lane Detection (With Distortion Correction & Bird's-eye View)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "Road Lane Detection (With Distortion Correction & Bird's-eye View)", w*2, h*2)

    while True:
        # Read each frame in the video
        success, input_frame = capture.read()

        if not success:
            # If unable to read the frame
            print("END!")
            break

        ##### MAIN ALGORITHM STARTS #####

        # Get undistorted image
        image_undist = cc.undistort_image(input_frame)

        # Convert the image to Bird's-eye View
        image_bev = front_to_top(image_undist, frame_w, frame_h)

        # Convert the image to thresholded binary image
        image_bin = thresh_binary(image_bev)

        # Perform Sliding Window algorithm
        image_sw = sliding_window(image_bin)

        # Convert the top-view image back to dashcam perspective
        image_front = top_to_front(image_sw, frame_w, frame_h)
        image_final = cv2.cvtColor(image_front, cv2.COLOR_GRAY2BGR)

        # Blend 'image_front' with the original image
        alpha = 0.8
        beta = 1
        gamma = 0
        output_frame = cv2.addWeighted(
            input_frame, alpha, image_final, beta, gamma)

        ##### MAIN ALGORITHM ENDS #####

        # Apply Vehicle Detection
        output_frame = vd.vehicle_detector(output_frame)

        # Write the output frame
        output.write(output_frame)

        # Add labels to the frames
        cv2.putText(input_frame, "Input Frame", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_bev, "Bird's-eye View Frame", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_final, "Bird's-eye View Frame with detected lanes", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(output_frame, "Final Output", (11, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frames
        concate_frame_1 = cv2.hconcat(
            [input_frame, image_bev])
        concate_frame_2 = cv2.hconcat(
            [image_final, output_frame])
        final_frame = cv2.vconcat([concate_frame_1, concate_frame_2])
        cv2.imshow(
            "Road Lane Detection (With Distortion Correction & Bird's-eye View)", final_frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    capture.release()
    output.release()
    cv2.destroyAllWindows()

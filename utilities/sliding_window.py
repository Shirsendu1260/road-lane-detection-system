import cv2
import numpy as np


# def nothing(x):
#     pass


# # Create trackbars
# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("H (Low)", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("H (High)", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("S (Low)", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("S (High)", "Trackbars", 50, 255, nothing)
# cv2.createTrackbar("V (Low)", "Trackbars", 200, 255, nothing)
# cv2.createTrackbar("V (High)", "Trackbars", 255, 255, nothing)


# # The points from the region where the lane is visible
# source_points = np.float32([(412, 345), (112, 540), (900, 540), (570, 345)])

# # The points to display the source region from top-view
# destination_points = np.float32([(75, 0), (75, 540), (825, 540), (825, 0)])

# # Create perspective region of top-view where lane lines look parallel
# top_view = cv2.getPerspectiveTransform(source_points, destination_points)

# # Create perspective region of front-view where lane lines are converging
# front_view = cv2.getPerspectiveTransform(destination_points, source_points)


# # Function that converts front-view image to top-view image
# def front_to_top(img, width, height):
#     return cv2.warpPerspective(img, top_view, (width, height), flags=cv2.INTER_LINEAR)


# # Function that converts top-view image to front-view image
# def top_to_front(img, width, height):
#     return cv2.warpPerspective(img, front_view, (width, height), flags=cv2.INTER_LINEAR)


# Function to convert a RGB image to a thresholded binary image
def thresh_binary(img):
    # Convert the image to HSV color format
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # # Define trackbars
    # low_h = cv2.getTrackbarPos("H (Low)", "Trackbars")
    # high_h = cv2.getTrackbarPos("H (High)", "Trackbars")
    # low_s = cv2.getTrackbarPos("S (Low)", "Trackbars")
    # high_s = cv2.getTrackbarPos("S (High)", "Trackbars")
    # low_v = cv2.getTrackbarPos("V (Low)", "Trackbars")
    # high_v = cv2.getTrackbarPos("V (High)", "Trackbars")

    low_h = 0
    high_h = 255
    low_s = 0
    high_s = 255
    low_v = 215
    high_v = 255

    # Get lower and higher thresholds of every channel
    low = np.array([low_h, low_s, low_v])
    high = np.array([high_h, high_s, high_v])

    # print(low, high)

    # Threshold the HSV image
    mask = cv2.inRange(img_hsv, low, high)

    return mask


# Function to apply Sliding Window algorithm to a thresholded binary image which is in Bird's-eye view form
def sliding_window(img):
    # Get histogram of bottom half of the binary image along x-axis where lane pixel's intensity are high
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    # Separate left and right lanes
    mid_point = int((histogram.shape[0] / 2))

    # When there is any peak before midpoint, store the maximum peak's x-coordinate
    left_base = np.argmax(histogram[:mid_point])

    # When there is any peak after midpoint, store the maximum peak's x-coordinate
    right_base = np.argmax(histogram[mid_point:]) + mid_point

    y = 531

    # List to store x-coordinates of left and right lanes respectively
    lx = []
    rx = []

    window_no = 20  # Number of sliding windows
    window_height = int(y / window_no)

    # Half of the width of a window = 50

    # Until reaching top
    while y > 0:
        # It will contain only the contents of the bottom left window
        img_l = img[y-window_height:y, left_base-50:left_base+50]

        # Detect contours
        contours, _ = cv2.findContours(
            img_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            # Get moments
            M = cv2.moments(cont)

            if M["m00"] != 0:
                # Get center coordinate (x) of the detected lane of that window
                cx = int(M["m10"] / M["m00"])

                # Append 'cx' as the perspective of the window
                # So considering x-origin, append as 'left_base - 50 + cx'
                lx.append(left_base - 50 + cx)

                # Update left_base
                left_base = left_base - 50 + cx

        # It will contain only the contents of the bottom right window
        img_r = img[y-window_height:y, right_base-50:right_base+50]

        # Detect contours
        contours, _ = cv2.findContours(
            img_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            # Get moments
            M = cv2.moments(cont)

            if M["m00"] != 0:
                # Get center coordinate (x) of the detected lane of that window
                cx = int(M["m10"] / M["m00"])

                # Append 'cx' as the perspective of the window
                # So considering x-origin, append as 'right_base - 50 + cx'
                rx.append(right_base - 50 + cx)

                # Update right_base
                right_base = right_base - 50 + cx

        # Draw the windows
        cv2.rectangle(img, (left_base-50, y), (left_base+50,
                      y-window_height), (255, 255, 255), 3)
        cv2.rectangle(img, (right_base-50, y), (right_base +
                      50, y-window_height), (255, 255, 255), 3)

        # Update y-axis value
        y -= window_height

    return img


# # Test code
# vid = cv2.VideoCapture(r"test_videos\test3.mp4")
# while True:
#     _, frame = vid.read()
#     top_frame = front_to_top(frame, 960, 540)
#     bin = thresh_binary(top_frame)
#     sw = sliding_window(bin)
#     sw_f = cv2.cvtColor(top_to_front(sw, 960, 540), cv2.COLOR_GRAY2BGR)
#     alpha = 0.8
#     beta = 1
#     gamma = 0
#     output = cv2.addWeighted(frame, alpha, sw_f, beta, gamma)
#     cv2.imshow("Output", bin)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()

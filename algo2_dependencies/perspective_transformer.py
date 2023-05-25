import cv2
import numpy as np
import matplotlib.pyplot as plt


# img = cv2.imread(r"test_images\test3.jpg")
# plt.imshow(img)
# plt.show()


# The points from the region where the lane is visible
source_points = np.float32([(412, 345), (112, 540), (900, 540), (577, 345)])

# The points to display the source region from top-view
destination_points = np.float32([(75, 0), (75, 540), (825, 540), (825, 0)])

# Create perspective region of top-view where lane lines look parallel
top_view = cv2.getPerspectiveTransform(source_points, destination_points)

# Create perspective region of front-view where lane lines are converging
front_view = cv2.getPerspectiveTransform(destination_points, source_points)


# Function that converts front-view image to top-view image
def front_to_top(img):
    return cv2.warpPerspective(img, top_view, (960, 540), flags=cv2.INTER_LINEAR)


# Function that converts top-view image to front-view image
def top_to_front(img):
    return cv2.warpPerspective(img, front_view, (960, 540), flags=cv2.INTER_LINEAR)


# Test code
img = cv2.imread(r"test_images\test3.jpg")
out_img1 = front_to_top(img)
out_img2 = top_to_front(out_img1)
cv2.imshow("Image1", img)
cv2.imshow("Image2", out_img1)
cv2.imshow("Image3", out_img2)
cv2.waitKey(0)

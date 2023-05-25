# A sample program to understand camera calibration on a chessboard


import cv2


nx = 7  # No. of boxes in horizontal direction in the chessboard (count from 0)
ny = 7  # No. of boxes in vertical direction in the chessboard (count from 0)

img = cv2.imread(r"test_images\chess.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find coordinates of chessboard corners of every boxes
success, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

# Draw the corners in the chessboard image
if success == True:
    cv2.drawChessboardCorners(img, (nx, ny), corners, success)

cv2.imshow("Image", img)
cv2.waitKey(0)

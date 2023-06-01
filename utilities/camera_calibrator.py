import numpy as np
import cv2
import glob
import os
import pickle


# Function that calibrate the camera using chessboard images
def camera_calibrator():
    # No. of boxes in horizontal direction in the chessboard (count from 0)
    nx = 9
    # No. of boxes in vertical direction in the chessboard (count from 0)
    ny = 6

    obj_points = []  # Coordinates of the corners in real life where they should have been equally spaced
    img_points = []  # Coordinates of the corners in image space obtained by OpenCV's findChessboardCorners()

    # Read all the 20 chessboard images at once from 'test_images\camera_cal'
    images = glob.glob("{}/*".format(r"test_images\camera_cal"))
    # print(images)

    # Get the coordinates of chessboard corners in real life 3D space (initialized with 0s)
    obj_p = np.zeros((nx*ny, 3), np.float32)

    # As the image plane is 2D, transform the 3D space derived from real life to 2D space
    obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Loop through every chessboard images
    for image in images:
        img = cv2.imread(image)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find coordinates of chessboard corners of every boxes
        success, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

        # If the corners are detected, fill up the image and object points
        if success:
            # Where the corners should be in real life
            obj_points.append(obj_p)

            # Where the corners are present in the 2D distorted image
            img_points.append(corners)

    # Compare the points of distorted image with real life points by calibrating camera
    # Get camera calibration matrix and distortion coefficients to undistort the image later
    width = img.shape[1]
    height = img.shape[0]
    ret, cam_mtx, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, (width, height), None, None)

    if not ret:
        print("Camera Calibration failed.")
        cv2.destroyAllWindows()

    print("Camera Calibration done successfully.")
    return cam_mtx, dist_coeffs


# The calibration process only needs to be executed once
# Then we can serialize and store the camera calibration matrix and the distortion coefficients to a pickle file
# So that we can use them without executing calibration process each and every time
def load_pickle():
    if os.path.exists(r"utilities\camera_calibration.p"):
        with open(r"utilities\camera_calibration.p", "rb") as file:
            data = pickle.load(file)
            cam_mtx, dist_coeffs = data['cam_mtx'], data['dist_coeffs']
            print(
                "The saved camera calibration matrix and distortion coefficients are loaded.")
    else:
        cam_mtx, dist_coeffs = camera_calibrator()
        with open(r"utilities\camera_calibration.p", "wb") as file:
            pickle.dump({'cam_mtx': cam_mtx, 'dist_coeffs': dist_coeffs}, file)

    return cam_mtx, dist_coeffs


cam_mtx, dist_coeffs = load_pickle()


# Function to undistort the image
def undistort_image(img):
    undist_img = cv2.undistort(img, cam_mtx, dist_coeffs, None, cam_mtx)
    return undist_img


# # Test code
# img = cv2.imread(r"test_images\camera_cal\calibration17.jpg")

# out_img = undistort_image(img)

# cv2.imshow("Image", out_img)
# cv2.waitKey(0)

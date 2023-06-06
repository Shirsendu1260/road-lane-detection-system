# Road Lane Detection System

The repository includes two image processing pipelines for lane and vehicle detection from dashcam footage using Python and OpenCV. Here, you'll find not one, but two distinct pipelines, each with its own set of features. The first algorithm use a naive approach using Canny Edge Detection and Hough Transform to identify lanes. However, it is unable to eliminate distortion that are caused by camera lenses. Additionally, in this case, the drawn lane lines have a tendency to converge at a specific point in the video frame, which is terrible in a real-world scenario. The second algorithm is therefore suggested for that purpose. The drawn lane lines no longer tend to converge at some point thanks to this algorithm, which also eliminates the camera distortion problem. To find lanes, it use a Sliding Window approach. To detect various cars, both algorithms employ the SSD MobileNet v3 architecture.


---
Project 1: Lane Detection using Canny Edge Detection and Probabilistic Hough Transform
---

### Steps:
* Read the frames of the video one by one.
* Convert the frame to grayscale to reduce computational cost.
* Using a 5X5 Gaussian kernel in a Gaussian filter, reduce noise.
* Detect edges using Canny edge detector.
* Set a region of interest (a trapezoid that only shows the area where the lane is visible), and mask off all other pixels except those in the region of interest.
* To get the coordinates of the left and right lane lines, apply Hough Transform.
* Based on their slope and y-intersect value, average the lane lines to obtain coordinates for just one left and one right line.
* Draw these two lines on a black mask image that is the same dimension as the input frame.
* The final frame is created by blending the input frame and the mask image.
* Apply Vehicle Detection on the final frame.

### Requirements:
* Python 3
* Numpy
* Matplotlib
* OpenCV

### Command to run this project:
`python algo1.py input.mp4`

Where `algo1.py` is a Python file containing the project's algorithm and `input.mp4` is an MP4 video on which to test the algorithm. These test videos are placed in the `test_videos` folder.

### Final result:
![1](https://github.com/Shirsendu1260/road-lane-detection-system/assets/102348951/924f6507-5f33-46f2-a3f7-7a76095cccba)


---
Project 2: Lane Detection using Camera Calibration matrix and Sliding Window approach
---

### Steps:
* Read the frames of the video one by one.
* Apply distortion correction to the image after computing the camera calibration matrix and distortion coefficients using a collection of chessboard images.
* Given a region of interest, apply perspective transform to convert the image to a bird's-eye view.
* Utilizing HSV color space, convert the image to a thresholded binary image.
* Apply Sliding Window approach to detect lane pixels on both lanes and convert them to their original perspective.
* The final frame is created by blending the detected lane image and the original undistorted frame.
* Apply Vehicle Detection on the final frame.

### Requirements:
* Python 3
* Numpy
* Matplotlib
* OpenCV

### Command to run this project:
`python algo2.py input.mp4`

Where `algo2.py` is a Python file containing the project's algorithm and `input.mp4` is an MP4 video on which to test the algorithm. These test videos are placed in the `test_videos` folder.

### Final result:
![2](https://github.com/Shirsendu1260/road-lane-detection-system/assets/102348951/c8634f25-57ad-4c09-9518-ed80fe86370d)


---
Dataset credit: Duong Pham, Eddie Forson

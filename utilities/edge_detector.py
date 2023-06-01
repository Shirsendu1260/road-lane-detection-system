import cv2


# Function that takes in an image. It then converts it to grayscale
# Then converts that grayscale image to blurred image to remove noise
# Finally edges are detected using Canny Edge detector on that blurred image
def edge_detector(img, low_threshold, high_threshold):
    # Grayscale conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    kernel = 5
    img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), 0)

    # Edge detection using Canny Edge detector
    img_canny = cv2.Canny(img_blur, low_threshold, high_threshold)

    return img_canny

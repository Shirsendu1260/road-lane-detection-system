import numpy as np


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

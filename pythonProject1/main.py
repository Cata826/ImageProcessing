import tkinter as tk
from tkinter import filedialog

import cv2 as my


def gaussian_kernel(size, sigma=1):
    """Generate a Gaussian kernel."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    # Calculate each element in the kernel matrix
    for x in range(size):
        for y in range(size):
            off_x, off_y = x - center, y - center
            exponent = -(off_x**2 + off_y**2) / (2 * sigma**2)
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)

    return kernel / np.sum(kernel)  # Normalize the kernel

def gaussian_blur(image, kernel_size=5, sigma=1):
    """Apply Gaussian blur to an image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_width = kernel_size // 2
    image_padded = np.pad(image, pad_width, mode='constant', constant_values=0)

    output = np.zeros_like(image)

    # Convolve the kernel with the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the region of interest
            region = image_padded[i:i + kernel_size, j:j + kernel_size]
            # Perform element-wise multiplication followed by the sum
            output[i, j] = np.sum(region * kernel)

    return output



def sobel_filters(image):
    """Apply Sobel filters to detect image gradients."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    Gx = convolve2d_custom(image, Kx)
    Gy = convolve2d_custom(image, Ky)

    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)

    return G, theta

def convolve2d_custom(image, kernel):
    """Manually apply a 2D convolution without padding (valid mode)."""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(image[y:y+kernel_height, x:x+kernel_width] * kernel)

    return output



def non_max_suppression(gradient_mag, gradient_dir):
    """
    Apply non-maximum suppression to thin edges.

    Parameters:
        gradient_mag (numpy.ndarray): The gradient magnitudes of the image.
        gradient_dir (numpy.ndarray): The gradient directions of the image.

    Returns:
        numpy.ndarray: The thinned edges.
    """
    M, N = gradient_mag.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = gradient_dir * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_mag[i, j + 1]
                    r = gradient_mag[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_mag[i + 1, j - 1]
                    r = gradient_mag[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_mag[i + 1, j]
                    r = gradient_mag[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_mag[i - 1, j - 1]
                    r = gradient_mag[i + 1, j + 1]

                if (gradient_mag[i, j] >= q) and (gradient_mag[i, j] >= r):
                    Z[i, j] = gradient_mag[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(image, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    """
    Apply double threshold to determine weak and strong edges.

    Parameters:
        image (numpy.ndarray): The non-max suppressed image.
        low_threshold_ratio (float): Ratio for low threshold.
        high_threshold_ratio (float): Ratio for high threshold.

    Returns:
        tuple: The thresholded image, weak edge value, strong edge value.
    """
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    strong = 255
    weak = 75
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    result = np.zeros(image.shape, dtype=np.uint8)

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return (result, weak, strong)


def hysteresis(image, weak, strong=255):
    """
    Apply hysteresis to track edge connectivity.

    Parameters:
        image (numpy.ndarray): The thresholded image.
        weak (int): The value assigned to weak edges.
        strong (int): The value assigned to strong edges.

    Returns:
        numpy.ndarray: The final edge-detected image.
    """
    M, N = image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (image[i, j] == weak):
                if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                        or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                        or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                image[i - 1, j + 1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image


def canny(image):
    """
    Complete Canny edge detection.

    Parameters:
        image (numpy.ndarray): The input grayscale image.

    Returns:
        numpy.ndarray: The edge-detected image.
    """
    blurred_image = gaussian_blur(image, sigma=1.4)
    gradient_mag, gradient_dir = sobel_filters(blurred_image)
    non_max_img = non_max_suppression(gradient_mag, gradient_dir)
    threshold_img, weak, strong = threshold(non_max_img)
    img_final = hysteresis(threshold_img, weak, strong)
    return img_final


def find_largest_contour_area(image):
    blurred = my.GaussianBlur(image, (5, 5), 0)

    # Detect edges using the Canny edge detector
    # edges = my.Canny(blurred, 50, 150)
    edges = canny(image)
    # edges = canny_edge_detection(image )
    # Find contours in the edged image
    contours, hierarchy = my.findContours(edges.copy(), my.RETR_EXTERNAL, my.CHAIN_APPROX_SIMPLE)

    # If no contours are detected
    if not contours:
        print("No contours found.")
        return image

    # Create a mask to draw the largest contour
    mask = np.zeros_like(image)

    # Find the largest contour by area
    largest_contour = max(contours, key=my.contourArea)

    # Draw the largest contour on the mask (color it white and fill it)
    my.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=my.FILLED)
    my.drawContours(image, [largest_contour], -1, (255, 105, 180), thickness=my.FILLED)

    # Apply the mask to the original image to black out everything except the largest area
    result = my.bitwise_and(image, mask)

    return result


def find_all_min_white_pixels(image, axis):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = my.cvtColor(image, my.COLOR_BGR2GRAY)
    # Count white pixels along the specified axis
    white_pixel_count = np.sum(image == 255, axis=axis).astype(float)
    # Ignore counts ≤ 2 by setting them to infinity
    white_pixel_count[white_pixel_count <= 2] = np.inf
    # Find the minimum count
    min_count = np.min(white_pixel_count)
    if min_count == np.inf:
        return []
    # Find all indices with the minimum count
    min_indices = np.where(white_pixel_count == min_count)[0]
    return min_indices.tolist()


def find_white_pixel_coordinates(image, column_indices):
    column_coordinates = {}
    for col in column_indices:
        # Find the y-coordinates of the white pixels in this column
        white_pixels_y = np.where(image[:, col] == 255)[0]
        # Calculate the arithmetic mean of the y-coordinates
        if white_pixels_y.size > 0:
            avg_y = np.mean(white_pixels_y)
            column_coordinates[col] = (col, avg_y)
        else:
            column_coordinates[col] = (col, None)
    # Return the coordinates grouped by column, with the average y for each column
    coordinates_list = [(col, column_coordinates[col][1]) for col in column_coordinates]
    unique_coordinates_list = remove_duplicates(coordinates_list)
    return unique_coordinates_list


def remove_duplicates(coordinates):
    # Initialize a set to keep track of the unique coordinates
    unique_coordinates = set()
    # Initialize a list to store non-duplicate coordinates
    non_duplicate_coordinates = []

    for x, y in coordinates:
        # Check if either x or y already exists in the set of unique coordinates
        if (x not in unique_coordinates) and (y not in unique_coordinates):
            # Add this coordinate to the list of non-duplicates
            non_duplicate_coordinates.append((x, y))
            # Add both x and y to the set of seen coordinates
            unique_coordinates.add(x)
            unique_coordinates.add(y)

    return non_duplicate_coordinates


def find_white_pixel_coordinates_row(image, row_indices):
    row_coordinates = {}
    for row in row_indices:
        # Find the x-coordinates of the white pixels in this row
        white_pixels_x = np.where(image[row, :] == 255)[0]
        # Calculate the arithmetic mean of the y-coordinates
        if white_pixels_x.size > 0:
            avg_y = np.mean(white_pixels_x)
            row_coordinates[row] = (row, avg_y)
        else:
            row_coordinates[row] = (row, None)

    # Create the coordinates list with the average y for each row
    coordinates_list = [(row_coordinates[x][1], x) for x in row_coordinates]

    # Now call the remove_duplicates function on the list of coordinates
    unique_coordinates_list = remove_duplicates(coordinates_list)
    return unique_coordinates_list
    # # Return the coordinates grouped by row, with the average y for each row
    # return [(row_coordinates[x][1], x) for x in row_coordinates]


def draw_lines(image, indices, color=(0, 0, 255), thickness=1, axis=0):
    for index in indices:
        if axis == 0:  # Vertical line (column)
            my.line(image, (index, 0), (index, image.shape[0]), color, thickness)
            print(f"Drawing vertical line at x = {index}, covering all y.")
        else:  # Horizontal line (row)
            my.line(image, (0, index), (image.shape[1], index), color, thickness)
            print(f"Drawing horizontal line at y = {index}, covering all x.")
    return image


def calculate_distances(image, rows, columns):
    for row_index in rows:
        row = image[row_index, :]
        for col_index in columns:
            active_pixels = np.where(row == 255)[0]
            if active_pixels.size > 0:
                # Calculate the horizontal distances to the red line at col_index
                distances = np.abs(active_pixels - col_index)
                # Find the minimum distance for this row
                min_distance = np.min(distances)
                print(
                    f"Minimum horizontal distance from red line at x = {col_index} to a white pixel in row y = {row_index} is {min_distance} pixels.")


def hough_transform(image, angle_step=1):
    # Define the Hough space (ranges for theta and rho)
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = image.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # Maximal possible rho value
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(image)  # Get indices of non-zero elements (edges)

    # Vote in the accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(thetas)):
            # Calculate rho. diag_len is added for a positive index
            rho = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j])) + diag_len)
            accumulator[rho, j] += 1

    return accumulator, thetas, rhos

def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: output from Hough Transform
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) * 2) + ((x2 - x1) * 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    #
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image (video frame in our case).
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            my.line(line_image, *line, color, thickness)
    return my.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: we pass here the output from canny where we have
        identified edges in the frame
    """
    # create an array of the same size as of the input image
    mask = np.zeros_like(image)
    # if you pass an image with more then one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    # our image only has one channel so it will go under "else"
    else:
        # color of the mask polygon (white)
        ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed
    rows, cols = image.shape[:2]
    # Adjust these points based on where the arrow is located in the image
    bottom_left = [int(cols * 0.22), int(rows * 0.9)]
    top_left = [int(cols * 0.4), int(rows * 0.1)]
    bottom_right = [int(cols * 0.6), int(rows * 0.9)]
    top_right = [int(cols * 0.6), int(rows * 0.05)]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    my.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = my.bitwise_and(image, mask)
    return masked_image


def find_min_white_pixels_row(image):
    # Count the white pixels in each row
    white_pixel_count_per_row = np.sum(image == 255, axis=1)
    # Find the row with the minimum number of white pixels
    min_white_pixels_row = np.argmin(white_pixel_count_per_row)
    min_white_pixels_count = np.min(white_pixel_count_per_row)
    return min_white_pixels_row, min_white_pixels_count


def draw_min_white_pixels_row(image, min_row_index, color=(0, 0, 255), thickness=1):
    """
    Draw a red line over the row that contains the minimum number of white pixels.

    Parameters:
        image: The input image where the line will be drawn.
        min_row_index: The index of the row to draw the line on.
        color: The color of the line to draw. Default is red.
        thickness: The thickness of the line. Default is 1 pixel.
    """
    # Draw a red horizontal line across the width of the image
    my.line(image, (0, min_row_index), (image.shape[1], min_row_index), color, thickness)


import numpy as np


def find_all_min_white_pixels(image, axis):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = my.cvtColor(image, my.COLOR_BGR2GRAY)
    # Count white pixels along the specified axis
    white_pixel_count = np.sum(image == 255, axis=axis).astype(float)
    # Ignore counts ≤ 2 by setting them to infinity
    white_pixel_count[white_pixel_count <= 2] = np.inf
    # Find the minimum count
    min_count = np.min(white_pixel_count)
    if min_count == np.inf:
        return []
    # Find all indices with the minimum count
    min_indices = np.where(white_pixel_count == min_count)[0]
    return min_indices.tolist()


def find_white_pixel_coordinates(image, column_indices):
    column_coordinates = {}
    for col in column_indices:
        # Find the y-coordinates of the white pixels in this column
        white_pixels_y = np.where(image[:, col] == 255)[0]
        # Calculate the arithmetic mean of the y-coordinates
        if white_pixels_y.size > 0:
            avg_y = np.mean(white_pixels_y)
            column_coordinates[col] = (col, avg_y)
        else:
            column_coordinates[col] = (col, None)
    # Return the coordinates grouped by column, with the average y for each column
    coordinates_list = [(col, column_coordinates[col][1]) for col in column_coordinates]
    unique_coordinates_list = remove_duplicates(coordinates_list)
    return unique_coordinates_list


def remove_duplicates(coordinates):
    # Initialize a set to keep track of the unique coordinates
    unique_coordinates = set()
    # Initialize a list to store non-duplicate coordinates
    non_duplicate_coordinates = []

    for x, y in coordinates:
        # Check if either x or y already exists in the set of unique coordinates
        if (x not in unique_coordinates) and (y not in unique_coordinates):
            # Add this coordinate to the list of non-duplicates
            non_duplicate_coordinates.append((x, y))
            # Add both x and y to the set of seen coordinates
            unique_coordinates.add(x)
            unique_coordinates.add(y)

    return non_duplicate_coordinates


def find_white_pixel_coordinates_row(image, row_indices):
    row_coordinates = {}
    for row in row_indices:
        # Find the x-coordinates of the white pixels in this row
        white_pixels_x = np.where(image[row, :] == 255)[0]
        # Calculate the arithmetic mean of the y-coordinates
        if white_pixels_x.size > 0:
            avg_y = np.mean(white_pixels_x)
            row_coordinates[row] = (row, avg_y)
        else:
            row_coordinates[row] = (row, None)

    # Create the coordinates list with the average y for each row
    coordinates_list = [(row_coordinates[x][1], x) for x in row_coordinates]

    # Now call the remove_duplicates function on the list of coordinates
    unique_coordinates_list = remove_duplicates(coordinates_list)
    return unique_coordinates_list
    # # Return the coordinates grouped by row, with the average y for each row
    # return [(row_coordinates[x][1], x) for x in row_coordinates]


def draw_lines(image, indices, color=(0, 0, 255), thickness=1, axis=0):
    for index in indices:
        if axis == 0:  # Vertical line (column)
            my.line(image, (index, 0), (index, image.shape[0]), color, thickness)
            print(f"Drawing vertical line at x = {index}, covering all y.")
        else:  # Horizontal line (row)
            my.line(image, (0, index), (image.shape[1], index), color, thickness)
            print(f"Drawing horizontal line at y = {index}, covering all x.")
    return image


def calculate_distances(image, rows, columns):
    for row_index in rows:
        row = image[row_index, :]
        for col_index in columns:
            active_pixels = np.where(row == 255)[0]
            if active_pixels.size > 0:
                # Calculate the horizontal distances to the red line at col_index
                distances = np.abs(active_pixels - col_index)
                # Find the minimum distance for this row
                min_distance = np.min(distances)
                print(
                    f"Minimum horizontal distance from red line at x = {col_index} to a white pixel in row y = {row_index} is {min_distance} pixels.")


def draw(img):
    # Find the first encountered x pixel with a red component
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if np.all(img[y, x] == 255):
                # Draw a red line along the y-axis at this x coordinate
                my.line(img, (x, 0), (x, img.shape[0]), (0, 0, 255), thickness=2)
                print(f"Coordinates of the red line: ({x}, 0) to ({x}, {img.shape[0] - 1})")
                return

def find_most_white_pixels(img):
    max_white_pixels = 0
    max_y = 0
    for y in range(img.shape[0]):
        white_pixels = np.sum(img[y] == 255)
        if white_pixels > max_white_pixels:
            max_white_pixels = white_pixels
            max_y = y
    return max_y

def draw_blue_line(img, y):
    my.line(img, (0, y), (img.shape[1]-1, y), (255, 0, 0), thickness=2)
# def find_red_lines(img):
#     red_lines_x = set()
#     for y in range(img.shape[0]):
#         for x in range(img.shape[1]):
#             if np.array_equal(img[y, x], [0, 0, 255]):
#                 red_lines_x.add(x)
#     return red_lines_x

def find_last_red_line(img):
    last_red_line_x = None
    for y in range(img.shape[0] - 1, -1, -1):
        for x in range(img.shape[1]):
            if np.array_equal(img[y, x], [0, 0, 255]):
                last_red_line_x = x
    return last_red_line_x

def find_first_red_line(img):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if np.array_equal(img[y, x], [0, 0, 255]):
                return x
    return None
# def find_distances(red_lines_x, white_pixel_rcoords):
#     distances = []
#     for red_x in red_lines_x:
#         for white_x, _ in white_pixel_rcoords:
#             distances.append(abs(red_x - white_x))
#     return distances
def find_distances(red_line_x, white_pixel_rcoords):
    distances = []
    for white_x, _ in white_pixel_rcoords:
        distances.append(abs(red_line_x - white_x))
    return distances
distance=[]

def select_and_process_image(img2_result):
    image_bgr = my.cvtColor(img2_result, my.COLOR_GRAY2BGR)
    min_row_indices = find_all_min_white_pixels(img2_result, axis=1)
    min_col_indices = find_all_min_white_pixels(img2_result, axis=0)
    condition_met1=0
    condition_met2=0

    if min_row_indices:
        white_pixel_rcoords = find_white_pixel_coordinates_row(img2_result, min_row_indices)
        middle_x_coordinate = find_middle_coordinate_x(image_bgr)

        print(f"Coordinates of white pixels that determine the red line(s) width: {white_pixel_rcoords}")
        print(f"Minimum white pixels found at rows: {min_row_indices}")
        draw_lines(image_bgr, min_row_indices, axis=1)

        # # Check if any white pixel coordinate is less than the middle x-coordinate
        # if (x < middle_x_coordinate for x, _ in white_pixel_rcoords):
        #     # print(
        #     #     "At least one white pixel coordinate determining red line width is less than the middle x-coordinate.")
        #     # Set a boolean value to 1 indicating the condition is met
        #     condition_met1 = 1
        #     # condition_met2 = 1


    if min_col_indices:
        print(f"Minimum white pixels found at columns: {min_col_indices}")
        draw_lines(image_bgr, min_col_indices, axis=0)
        white_pixel_coords = find_white_pixel_coordinates(img2_result, min_col_indices)
        print(f"Coordinates of white pixels that determine the red line(s) height : {white_pixel_coords}")
        calculate_distances(img2_result, min_row_indices, min_col_indices)
        middle_x_coordinate = find_middle_coordinate_x(image_bgr)
        # Check if any white pixel coordinate is less than the middle x-coordinate
        # if (x > middle_x_coordinate for x, _ in white_pixel_coords):
        #     # print(
        #     #     "At least one white pixel coordinate determining red line width is less than the middle x-coordinate.")
        #     # Set a boolean value to 1 indicating the condition is met
        #     condition_met1=2
        #
        #     # condition_met1=0

    result_path = 'result_image_with_multiple_red_lines.png'
    draw(image_bgr)
    most_white_y = find_most_white_pixels(image_bgr)
    draw_blue_line(image_bgr, most_white_y)
    first_red_line_x = find_first_red_line(image_bgr)
    if first_red_line_x is not None:
        print("X-coordinate of the first red line:")
        print(first_red_line_x)
        distances = find_distances(first_red_line_x, white_pixel_rcoords)
        distance.append(distances)
        print("Distances between the first red line and white pixel x-coordinates:")
        print(distances)
    else:
        print("No red line found.")
    last_red_line_x=find_last_red_line(image_bgr)
    if last_red_line_x is not None:
        print("X-coordinate of the last red line:")
        print(last_red_line_x)
        distances = find_distances(last_red_line_x, white_pixel_rcoords)
        distance.append(distances)
        print("Distances between the last red line and white pixel x-coordinates:")
        print(distances)
    # distances = find_distances(red_lines_x, white_pixel_rcoords)
    # print("Distances between red lines and white pixel x-coordinates:")
    # print(distances)

    my.imwrite(result_path, image_bgr)
    print(f"Red lines drawn on all rows and columns with the minimum white pixels. Saved to {result_path}.")
    print(distance)
    # sums = []
    # for rx, _ in white_pixel_rcoords:
    #     for cx, _ in white_pixel_coords:
    #         if rx - cx > 0:
    #             sums.append(rx - cx)
    #         elif rx - cx < 0:
    #             sums.append(cx - rx)
    #
    # if len(sums) < 2:
    #     sums.append(sums[0])
    if distance[0][0] < distance[1][0] and distance[0][0] > 20 and distance[0][0] < 60:
        print("STANGA")
        return
    elif distance[0][0] < distance[1][0] and distance[0][0] < 10:
        print("DREAPTA")
        return
    elif distance[0][0] > distance[1][0] and distance[0][0] > 90:
        print("INAINTE SI STANGA")
        return
    elif distance[0][0] < distance[1][0] and distance[1][0] > 100 and distance[0][0]<75:
        print("INAINTE SI DREAPTA")
        return
    elif distance[0][0] > 75 and distance[1][0] > 80:
        print("INAINTE")

    #
    # if sums[0] > 100 and sums[1] > 70:
    #     print("inainte si dreapta si stanga")
    # if sums[0] < 70 and sums[1] > 70:
    #     print("inainte si la dreapta")
    #     return
    #
    # if sums[0] < 45 and sums[1] < 45  :
    #     print("inainte si stanga")
    #     return
    # if (sums[0] < 120 and sums[1] < 120):
    #     if  sums[0]>35 and sums[0] < 65 and sums[1] < 65 :
    #         print("stanga")
    #         return
    #     elif sums[0]>35 and sums[0] < 65 and sums[1] < 65 and condition_met1==1 :
    #         print("dreapta")
    #         return
    #     elif condition_met1 != 0 :
    #         print("inainte")
    #         return

def find_middle_coordinate_x(image):
    # Calculate the middle x-coordinate
    middle_x = image.shape[1] // 2
    # Print the coordinates
    print(f"Middle coordinate along the x-axis: ({middle_x}, 0)")
    return middle_x

def process_image_and_return_img2():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    image_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])

    if not image_path:
        print("No file selected.")
        return None

    image = my.imread(image_path)
    if image is None:
        print("Image not found or unable to load")
        return None

    grayscale = my.cvtColor(image, my.COLOR_BGR2GRAY)
    blur = my.GaussianBlur(grayscale, (5, 5), 0)
    #edges = my.Canny(blur, 50, 150)
    edges = canny(grayscale)
    region = region_selection(edges)
    my.imshow("Edges", edges)
    my.imshow("Region", region)

    img2 = find_largest_contour_area(region)
    my.imshow("Largest Contour Area", img2)
    middle_x_coordinate = find_middle_coordinate_x(img2)

    select_and_process_image(img2)
    my.waitKey(0)
    my.destroyAllWindows()

    return img2

process_image_and_return_img2()


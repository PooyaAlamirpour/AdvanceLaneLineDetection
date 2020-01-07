import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import HTML
from base64 import b64encode
from math import floor

video = cv2.VideoCapture('project_video.mp4')                   # Load the source video
nFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))              # Get the number of frames
fps = video.get(cv2.CAP_PROP_FPS)                               # Calculate fps of the video
waitPerFrameInMillisec = int(1/fps * 1000/1)                    # Calculate waiting time between each frame

success, input_image = video.read()
height, width, layers = input_image.shape
size = (width, height)
out = cv2.VideoWriter('masked_output_project_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
avg_left_point = 50
avg_right_point = 240
avg_left_point_list = []
avg_right_point_list = []

def make_gradient_transform(img, sobel_kernel=3, hsv_thresh=(100, 255), sobel_thresh=(40, 100), sl=0):
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    l_channel = hls[:, :, 1]  # Convert to HLS
    s_channel = hls[:, :, 2]
    if sl == 0:
        s_channel = hls[:, :, 2]
    else:
        s_channel = hls[:, :, 1]

    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold gradient(
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    # Threshold color
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= hsv_thresh[0]) & (s_channel <= hsv_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    return color_binary

def warp(img, src, dst):
  img_size = (img.shape[1], img.shape[0])
  offset = 100

  M = cv2.getPerspectiveTransform(src, dst)
  Minv = cv2.getPerspectiveTransform(dst, src)
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

  return warped, Minv

def make_binary(img, hsv_thresh=(100, 255)):
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    l_channel = hls[:, :, 1]  # Convert to HLS
    s_channel = hls[:, :, 2]

    # Threshold color
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= hsv_thresh[0]) & (s_channel <= hsv_thresh[1])] = 1
    color_binary = np.dstack((np.zeros_like(s_binary), s_binary, s_binary)) * 255

    return s_binary

def get_suitable_left_right_points(img):
    global avg_left_point
    global avg_right_point
    global avg_left_point_list
    global avg_right_point_list

    center_of_rec_x = input_image.shape[1] / 2
    center_of_rec_y = 460
    far_from_center_x = 120
    far_from_center_y = 30

    rec_top_right_x = center_of_rec_x + far_from_center_x
    rec_top_right_y = center_of_rec_y + far_from_center_y
    rec_top_left_x = center_of_rec_x - far_from_center_x
    rec_top_left_y = center_of_rec_y + far_from_center_y
    rec_bottom_right_x = center_of_rec_x + far_from_center_x
    rec_bottom_right_y = center_of_rec_y - 0
    rec_bottom_left_x = center_of_rec_x - far_from_center_x
    rec_bottom_left_y = center_of_rec_y - 0

    cpy_src = np.copy(img)
    y1 = int(rec_bottom_left_y)
    y2 = int(rec_top_left_y)
    x1 = int(rec_top_left_x)
    x2 = int(rec_top_right_x)

    crop_image = cpy_src[y1:y2, x1:x2]
    crop_binary_image = make_binary(crop_image)

    bottom_half = crop_binary_image[crop_binary_image.shape[0] // 2:, :]
    hist_result = np.sum(bottom_half, axis=0)

    thresh = max(hist_result) / 3
    new_hist = hist_result[:] > (thresh)
    ls = [i for i, e in enumerate(new_hist) if e != 0]
    left_cat = []
    right_cat = []
    tmp = 0
    if (len(ls) == 0):
        tmp = 10
    else:
        tmp = ls[0]

    avg_left = 0
    avg_right = 0

    for index in ls:
        if ((index - tmp) > 2):
            if index < 125:
                tmp = index
                if tmp > 45:
                    left_cat.append(tmp)
            else:
                tmp = index
                right_cat.append(tmp)

    if (len(left_cat) == 0):
        avg_left = avg_left_point
    else:
        avg_left = sum(left_cat) / len(left_cat)

    if (len(right_cat) == 0):
        avg_right = avg_right_point
    else:
        avg_right = sum(right_cat) / len(right_cat)

    avg_left_point_list.append(avg_left)
    avg_right_point_list.append(avg_right)

    tmp_left_avg = sum(avg_left_point_list) / len(avg_left_point_list)
    tmp_right_avg = sum(avg_right_point_list) / len(avg_right_point_list)

    return tmp_left_avg, tmp_right_avg

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, ploty

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_find_lane_pixels, lefty_find_lane_pixels, rightx_find_lane_pixels, righty_find_lane_pixels, out_img_find_lane_pixels = find_lane_pixels(binary_warped)
    left_fit = np.polyfit(lefty_find_lane_pixels, leftx_find_lane_pixels, 2)
    right_fit = np.polyfit(righty_find_lane_pixels, rightx_find_lane_pixels, 2)

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, left_fitx, right_fitx, ploty

def generate_data(binary_warped):
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    return ploty, left_fit, right_fit

def measure_curvature_pixels(binary_warped):
    ploty, left_fit, right_fit = generate_data(binary_warped)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad

last_left_fitx = 0
last_right_fitx = 0
last_ploty = 0
number_of_frame = 0;

def getPreHistogram(binary_warped):
    hist_result = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    hist_thresh = max(hist_result) / 2
    new_hist = hist_result[:] > (hist_thresh)
    ls = [i for i, e in enumerate(new_hist) if e != 0]
    return ls

def run():
    global last_left_fitx
    global last_right_fitx
    global last_ploty
    global number_of_frame

    success, input_image = video.read()
    if (success == True):

        center_of_rec_x = input_image.shape[1] / 2
        center_of_rec_y = 460
        far_from_center_x = 120
        far_from_center_y = 20

        rec_top_right_x = center_of_rec_x - far_from_center_x + 220 + 10
        rec_top_right_y = center_of_rec_y + far_from_center_y
        rec_top_left_x = center_of_rec_x - far_from_center_x + 50 - 10
        rec_top_left_y = center_of_rec_y + far_from_center_y
        top_left_x = rec_top_left_x
        top_left_y = rec_top_left_y
        top_right_x = rec_top_right_x
        top_right_y = rec_top_right_y

        bottom_left_x = 180
        bottom_left_y = input_image.shape[0]
        bottom_right_x = 1240
        bottom_right_y = input_image.shape[0]

        newwarp = np.zeros_like(input_image)
        lineThickness = 3

        cv2.line(newwarp, (int(top_left_x), int(top_left_y)), (int(bottom_left_x), int(bottom_left_y)), (0,255,0), lineThickness)
        cv2.line(newwarp, (int(top_right_x), int(top_right_y)), (int(bottom_right_x), int(bottom_right_y)), (0,255,0), lineThickness)
        cv2.line(newwarp, (int(top_left_x), int(top_left_y)), (int(top_right_x), int(top_right_y)), (0,255,0), lineThickness)
        cv2.line(newwarp, (int(bottom_left_x), int(bottom_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0,255,0), lineThickness)

        result = cv2.addWeighted(input_image, 1, newwarp, 0.3, 0)

        out.write(result)
        input_image = None

for i in range(0, nFrames - 1):
    print('Frame >> ', i, '/', nFrames)
    run()

out.release()


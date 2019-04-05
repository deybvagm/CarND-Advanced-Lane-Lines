import numpy as np
import cv2

def find_lane_pixels(binary_warped):
    '''
    Find the pixels that belong to the right and left lane lines given the `binary_waped` image
    Returns left pixels in `leftx`, `lefty` and right pixels in `rightx`, `righty`
    '''
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]//2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    margin = 100
    minpix = 50
    winheight = np.int(binary_warped.shape[0]//nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    
    left_lane_inds = []
    right_lane_inds = []
    
    leftx_current = left_base
    rightx_current = right_base
    
    for window in range(nwindows):
        # Boundaries of current window
        winy_high = binary_warped.shape[0] - (window * winheight)
        winy_low = binary_warped.shape[0] - (window + 1) * winheight
        winx_left_low = leftx_current - margin
        winx_left_high = leftx_current + margin
        winx_right_low = rightx_current - margin
        winx_right_high = rightx_current + margin
        
#         cv2.rectangle(out_img, (winx_left_low, winy_low), (winx_left_high, winy_high), (0, 255, 0), 2)
#         cv2.rectangle(out_img, (winx_right_low, winy_low), (winx_right_high, winy_high), (0, 255, 0), 2)
        
        good_left_ind = ((nonzerox > winx_left_low) & (nonzerox < winx_left_high) & (nonzeroy > winy_low) & (nonzeroy < winy_high)).nonzero()[0]
        good_right_ind = ((nonzerox > winx_right_low) & (nonzerox < winx_right_high) & (nonzeroy > winy_low) & (nonzeroy < winy_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_ind)
        right_lane_inds.append(good_right_ind)
        
        if len(good_left_ind) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_ind]))
        if len(good_right_ind) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_ind]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix):
    '''
    Fit a 2 order polynomial for each of the two lane lines givel the pixels that belog to the left and right lane lines
    Returns the polynomial coefficients in `poly_params` and the coordinates x, y that describe the fitted lines with the variable `fitted_lines_coord`
    '''
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_fit_r = np.polyfit(lefty*ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_r = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        
    poly_params = (left_fit_r, right_fit_r)
    fitted_lines_coord = (left_fitx, right_fitx, ploty)
    
    return poly_params, fitted_lines_coord


def search_around_poly(binary_warped, left_fit, right_fit, xm_per_pix, ym_per_pix):
    '''
    Takes previos values that describe the fitted lines and does a closer search to find the new coefficient params and the coordinates x, y that describe the lines
    Returns the polynomial paramms in `poly_params` and the points that describes the fitting in `lines_coords`
    '''
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###

    x_points_line1 = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
    x_points_line2 = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]
    left_lane_inds = ((nonzerox > (x_points_line1 - margin)) & (nonzerox < (x_points_line1 + margin)))
    right_lane_inds = ((nonzerox > (x_points_line2 - margin)) & (nonzerox < (x_points_line2 + margin)))

    # Extract the left and right lines pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fit new polynomials
    poly_params, lines_coord = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix)
    
    return poly_params, lines_coord


def sliding_window_search(binary_warped, ym_per_pix, xm_per_pix):
    '''
    Makes a sliding window search to find the polynomial parameters
    Returns the polynomial paramms in `poly_params` and the points that describes the fitting in `lines_coords`
    '''
    margin = 100
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    poly_params, fitted_lines_coord = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix)

    window_img = np.zeros_like(out_img)

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]    

    # Draw the polynomial fit
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
    return out_img, poly_params, fitted_lines_coord
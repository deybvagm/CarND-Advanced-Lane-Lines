import numpy as np

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, ym_per_pix, xm_per_pix):
    '''
    Gets the real curvature values in meters with polynmial coefficients and the mapping from pixels to meters
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** (3 / 2) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** (3 / 2) / np.absolute(
        2 * right_fit_cr[0])
    
    return left_curverad, right_curverad


def calculate_car_position(binary_warped, fitted_lines_coord, xm_per_pix):
    '''
    Calculates the car position with respect to center (assuming the camera is on the center of the car)
    '''
    leftx, rightx, ploty = fitted_lines_coord
    car_position = binary_warped.shape[1] / 2
    lane_center = (leftx[-1] + rightx[-1]) / 2
    center_dist = (car_position - lane_center) * xm_per_pix
    return center_dist
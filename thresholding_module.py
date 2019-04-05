import numpy as np
import cv2

def color_threshold(image, s_thresh=(50, 200), v_thresh=(150, 255), l_thresh=(225, 255), b_thresh=(155, 200)):
    # s_thresh=(200, 255), v_thresh=(150, 255), l_thresh=(130, 255), b_thresh=(150, 200)
    '''
    Applies color thresholding on the image `image`. Takes the S and V channel, apply a threshold and return the combination of them 
    Return a binary image
    '''
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    s = hls[:,:,2]
    v = hsv[:,:,2]
    l = luv[:,:,0]
    b = lab[:,:,2]
    
    s_binary = np.zeros_like(s)
    v_binary = np.zeros_like(v)
    l_binary = np.zeros_like(l)
    b_binary = np.zeros_like(b)
    
    s_binary[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    v_binary[(v >= v_thresh[0]) & (v <= v_thresh[1])] = 1
    l_binary[(l >= l_thresh[0]) & (l <= l_thresh[1])] = 1
    b_binary[(b >= b_thresh[0]) & (b <= b_thresh[1])] = 1
    
    combined_binary = np.zeros_like(s)
#     combined_binary[(v_binary == 1) | (l_binary == 1) | (b_binary == 1)] = 1
    combined_binary[(s_binary == 1) & (v_binary == 1) | (l_binary == 1) | (b_binary == 1) ] = 1
#     combined_binary[(s_binary == 1) | (v_binary == 1) | (l_binary == 1) | (b_binary == 1) ] = 1
    return combined_binary


def abs_sobel(image, orient='x', kernel_size=3, thresh=(20, 120)):
    '''
    Applies gradient threshold on x or y
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif orient == 'y':
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    abs_sobel = np.absolute(grad)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel[1] <= thresh[1])] = 1
    return grad_binary


def mag_gradient(img, kernel_size=3, thresh=(0, 255)):
    '''
    Applies gradient magnitude
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    mag_binary = np.zeros_like(scaled_magnitude)
    mag_binary[(scaled_magnitude >= thresh[0]) & (scaled_magnitude[1] <= thresh[1])] = 1
    return mag_binary


def dir_gradient(image, kernel_size=3, thresh=(0, np.pi/2)):
    '''
    Applies thresholding on the directions of gradients
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_output


def combined_threshold(image):
    '''
    to get a better binary image, combine the gradient on x, y, magnitude, direction and color thresholding
    Returns the combined binary image
    '''
    xy_thresh = (50, 255)#20,120
    mag_thresh = (50, 255)
    dir_thresh = (0.8, 1.2)#0.6, 1.2
    thresholded_sx = abs_sobel(image, 'x', kernel_size=7, thresh=xy_thresh)
    thresholded_sy = abs_sobel(image, 'y', kernel_size=7, thresh=xy_thresh)
    thresholded_mag = mag_gradient(image, kernel_size=7, thresh=mag_thresh)
    thresholded_dir = dir_gradient(image, kernel_size=5, thresh=dir_thresh)    
    thresholded_color = color_threshold(image)
    
    combined_gradient = np.zeros_like(thresholded_dir)
    combined_gradient[((thresholded_sx == 1) & (thresholded_sy == 1)) | ((thresholded_mag == 1)&(thresholded_dir==1))] = 1
    
    combined_binary = np.zeros_like(combined_gradient)
    combined_binary[(combined_gradient == 1) | (thresholded_color == 1)] = 1
    return combined_binary
    
    
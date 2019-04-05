import cv2
import numpy as np
import glob
import matplotlib.image as mpimg

def calculate_calibration_objects(path_to_images, nx, ny):
    '''
    Calculates the objects points and image points needed to calibrate the camera
    Receives the folder where the chessboard images are `path_to_images` and the number of corners in x `nx` and y `ny`
    '''
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in imaga plane
    
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    images = glob.glob(path_to_images + '/calibration*.jpg')
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            mpimg.imsave('output_images/'+fname, img)    
    return objpoints, imgpoints


def calibrate_camera(path_to_images, nx, ny):
    '''
    Calibrates the camera calling the helper function `calculate_calibration_objects`
    Receives as parameters the folder where the chessboard images are `path_to_images` and the number of corners in x `nx` and y `ny`
    Returns the camera matrix and the distortion coefficients
    '''
    calibration_image = mpimg.imread(path_to_images+'/calibration1.jpg')
    img_size = (calibration_image.shape[1], calibration_image.shape[0])
    objpoints, imgpoints = calculate_calibration_objects(path_to_images, nx, ny)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist


def undistort(img, mtx, dist):
    '''
    Undistort an image given the image `img`, the camera matrix `mtx` and the distortion coefficients `dist`
    Returns the undistorted image
    '''
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst
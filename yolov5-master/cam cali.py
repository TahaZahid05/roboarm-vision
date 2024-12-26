import cv2
import numpy as np
import glob

def calibrate_camera(images_folder, pattern_size=(8, 6), square_size=0.025, image_size=(640, 640)):
    """
    Calibrate the camera using checkerboard images.
    
    :param images_folder: Path to folder containing checkerboard images.
    :param pattern_size: Number of internal corners in the checkerboard (cols, rows).
    :param square_size: Size of a square in your checkerboard (in meters).
    :param image_size: The resolution of the images for calibration (width, height).
    :return: Camera matrix, distortion coefficients, rotation and translation vectors.
    """
    # Prepare object points for the checkerboard
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Get all images in the folder
    images = glob.glob(f"{images_folder}/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        
        # Resize the image to 640x640
        img = cv2.resize(img, image_size)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Checkerboard', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera Matrix:\n", mtx)
        print("\nDistortion Coefficients:\n", dist)
        return mtx, dist, rvecs, tvecs
    else:
        print("Camera calibration failed.")
        return None, None, None, None

if __name__ == "__main__":
    # Path to folder containing checkerboard images
    images_folder = "calibration_images"  # Change this to your folder path

    # Run calibration with the new image size
    calibrate_camera(images_folder, pattern_size=(8, 6), square_size=0.025, image_size=(640, 640))
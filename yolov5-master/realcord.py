import sys
import cv2
import torch
import numpy as np
import requests
import glob
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up main window
        self.setWindowTitle("YOLOv5 Object Detection")
        self.setGeometry(100, 100, 800, 700)

        # QLabel to display video feed (640 x 640 video feed)
        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 640, 640)
        self.video_label.setStyleSheet("border: 1px solid black")
        # selection_rect = rectangle drawn by the user
        # start_point = initial point from where user starts drawing
        # selection_made = Did the user select an object?
        self.selection_rect = None
        self.start_point = None
        self.selection_made = False

        # QLabel to display object info
        # Displays camera coordinates in the application
        self.info_label = QLabel(self)
        self.info_label.setGeometry(10, 650, 780, 50)
        self.info_label.setStyleSheet("border: 1px solid black")
        self.info_label.setAlignment(Qt.AlignLeft)

        # Timer for video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Load YOLOv5 model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load YOLOv5 model: {e}")
            sys.exit(1)

        # IP Camera URL
        self.image_url = "http://10.20.1.46:8080/shot.jpg"  # Replace with your IP camera URL

        self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = self.calibrate_camera("calibration_image")

        # Timer for video feed
        self.timer.start(30)

        # Variables for detections and selection
        self.detections = []
        self.selected_object = None

    # def calibrate_camera(self, images_folder, pattern_size=(8, 6), square_size=0.0508):
    #     """
    #     Calibrate the camera using checkerboard images.
        
    #     :param images_folder: Path to folder containing checkerboard images.
    #     :param pattern_size: Number of internal corners in the checkerboard (cols, rows).
    #     :param square_size: Size of a square in your checkerboard (in meters).
    #     :return: Camera matrix, distortion coefficients, rotation and translation vectors.
    #     """
    #     # Prepare object points for the checkerboard
    #     objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    #     objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    #     objp *= square_size

    #     # Arrays to store object points and image points
    #     objpoints = []  # 3D points in real-world space
    #     imgpoints = []  # 2D points in image plane

    #     # Get all images in the folder
    #     images = glob.glob(f"{images_folder}/*.jpg")

    #     for fname in images:
    #         img = cv2.imread(fname)
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #         # Find the chessboard corners
    #         ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    #         if ret:
    #             objpoints.append(objp)
    #             imgpoints.append(corners)

    #             # Draw and display the corners
    #             cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    #             cv2.imshow('Checkerboard', img)
    #             cv2.waitKey(500)

    #     cv2.destroyAllWindows()

    #     # Calibrate the camera
    #     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    #     if ret:
    #         print("Camera Matrix:\n", mtx)
    #         print("\nDistortion Coefficients:\n", dist)
    #         return mtx, dist, rvecs, tvecs
    #     else:
    #         print("Camera calibration failed.")
    #         return None, None, None, None

    def calibrate_camera(self,images_folder, checkerboard_size=(8, 6)):
        # Define the dimensions of the checkerboard
        CHECKERBOARD = checkerboard_size

        # Stop the iteration when specified accuracy is reached
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Vector for 3D points
        threedpoints = []

        # Vector for 2D points
        twodpoints = []

        # 3D points in real-world coordinates
        objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        # Extract the path of individual images in the given directory
        images = glob.glob(f"{images_folder}/*.jpg")

        for filename in images:
            image = cv2.imread(filename)
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_FAST_CHECK +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If corners are detected, refine the pixel coordinates and add to points
            if ret == True:
                print(f"Corners detected in {filename}")
                threedpoints.append(objectp3d)

                # Refining pixel coordinates for given 2D points
                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)

                twodpoints.append(corners2)

                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
            else:
                print(f"No corners detected in {filename}")

            # Display the image with corners (optional)
            cv2.imshow('img', image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

        # Perform camera calibration
        h, w = image.shape[:2]
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

        # Display the calibration results
        print("Camera matrix:")
        print(matrix)

        print("\nDistortion coefficients:")
        print(distortion)

        print("\nRotation Vectors:")
        print(r_vecs)

        print("\nTranslation Vectors:")
        print(t_vecs)

        return matrix, distortion, r_vecs, t_vecs

    def fetch_frame_from_ip_camera(self):
        try:
            # Fetch the image from the URL
            response = requests.get(self.image_url, stream=True)
            if not response.ok:
                self.info_label.setText("Error: Unable to fetch image from IP camera.")
                return None

            # Convert the image to a NumPy array
            image_array = np.array(bytearray(response.content), dtype=np.uint8)

            # Decode the image to OpenCV format
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Resize the frame to 640x640
            frame_resized = cv2.resize(frame, (640, 640))

            # Undistort using original camera matrix (no scaling needed)
            undistorted_frame = cv2.undistort(frame_resized, self.camera_matrix, self.dist_coeffs)

            return undistorted_frame

        except Exception as e:
            self.info_label.setText(f"Error fetching IP camera frame: {e}")
            return None

    def update_frame(self):
        # Get the frame from the IP camera
        frame = self.fetch_frame_from_ip_camera()
        if frame is None:
            return

        # Draw selection rectangle if selected
        if self.selection_rect:
            x1, y1, x2, y2 = self.selection_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Only perform YOLOv5 inference if a selection has been made
        if self.selection_made:
            x1, y1, x2, y2 = self.selection_rect
            cropped_frame = frame[y1:y2, x1:x2]

            if cropped_frame.size == 0 or cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
                self.info_label.setText("Invalid selection area. Please select a valid region.")
                return

            try:
                results = self.model(cropped_frame)
                detections = results.pandas().xyxy[0]
                self.detections = []

                for _, row in detections.iterrows():
                    x1_crop, y1_crop, x2_crop, y2_crop = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    x1_orig, y1_orig, x2_orig, y2_orig = x1 + x1_crop, y1 + y1_crop, x1 + x2_crop, y1 + y2_crop
                    self.detections.append({'bbox': (x1_orig, y1_orig, x2_orig, y2_orig), 'label': row['name'], 'confidence': row['confidence']})

                    label = f"{row['name']} {row['confidence']:.2f}"
                    cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if self.selected_object:
                    x1_sel, y1_sel, x2_sel, y2_sel = self.selected_object['bbox']
                    center_x = (x1_sel + x2_sel) // 2
                    center_y = (y1_sel + y2_sel) // 2
                    pixel_coords = np.array([center_x,center_y,1.35])
                    camera_coords = np.linalg.inv(self.camera_matrix) @ pixel_coords
                    X = camera_coords[0]
                    Y = camera_coords[1]
                    Z = 1.35
                    print(f"Camera coordinates: ({X:.2f}, {Y:.2f}, {Z:.2f})")
                    label = self.selected_object['label']
                    cv2.rectangle(frame, (x1_sel, y1_sel), (x2_sel, y2_sel), (0, 255, 255), 2)
                    self.info_label.setText(f"Selected: {label} ({x1_sel}, {y1_sel}) to ({x2_sel}, {y2_sel})")

            except Exception as e:
                self.info_label.setText(f"Inference error: {e}")
                return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            label_pos = self.video_label.pos()
            frame_width = 640  # Width of the frame (resized to 640x640)
            frame_height = 640  # Height of the frame (resized to 640x640)
            label_width, label_height = self.video_label.width(), self.video_label.height()
            x, y = (event.x() - label_pos.x()) * frame_width // label_width, (event.y() - label_pos.y()) * frame_height // label_height
            if self.selection_made:
                for detection in self.detections:
                    x1, y1, x2, y2 = detection['bbox']
                    print(x,y,x1,y1,x2,y2)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        pixel_coords = np.array([center_x,center_y,1.35])
                        camera_coords = np.linalg.inv(self.camera_matrix) @ pixel_coords
                        X = camera_coords[0]
                        Y = camera_coords[1]
                        Z = 1.35
                        print(f"Camera coordinates: ({X:.2f}, {Y:.2f}, {Z:.2f})")
                        self.selected_object = detection
                        self.info_label.setText(f"Selected: {detection['label']} ({x1}, {y1}) to ({x2}, {y2})")
                        self.update_frame()  # Update the frame with yellow border on selected object
                        return
            self.start_point = QPoint(x, y)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point:
            x, y = self.start_point.x(), self.start_point.y()
            x2, y2 = event.x(), event.y()
            self.selection_rect = (min(x, x2), min(y, y2), max(x, x2), max(y, y2))
            self.selection_made = True
            self.start_point = None
            self.update_frame()

        elif event.button() == Qt.RightButton:
            self.selection_rect = None
            self.selection_made = False
            self.selected_object = None
            self.info_label.setText("")

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())

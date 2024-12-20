import sys
import cv2
import torch
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up main window
        self.setWindowTitle("YOLOv5 Object Detection")
        self.setGeometry(100, 100, 800, 600)

        # QLabel to display video feed
        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 780, 500)
        self.video_label.setStyleSheet("border: 1px solid black")
        self.selection_rect = None
        self.start_point = None
        self.selection_made = False  # Flag to check if a selection is made

        # QLabel to display object info
        self.info_label = QLabel(self)
        self.info_label.setGeometry(10, 520, 780, 50)
        self.info_label.setStyleSheet("border: 1px solid black")
        self.info_label.setAlignment(Qt.AlignLeft)

        # Timer for video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp3/weights/best.pt')
        # self.model.conf = 0.5  # Confidence threshold

        # Start capturing video
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Draw selection rectangle if selected
        if self.selection_rect:
            x1, y1, x2, y2 = self.selection_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Only perform YOLOv5 inference if a selection has been made
        if self.selection_made:
            # Crop the frame to the selected region
            x1, y1, x2, y2 = self.selection_rect
            cropped_frame = frame[y1:y2, x1:x2]

            # Perform YOLOv5 inference on the cropped section
            results = self.model(cropped_frame)
            detections = results.pandas().xyxy[0]  # Pandas DataFrame with detections

            # Draw detections on the cropped frame
            for _, row in detections.iterrows():
                # Get bounding box coordinates for the cropped image
                x1_crop, y1_crop, x2_crop, y2_crop = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                # Adjust the bounding box to be relative to the original frame
                x1_orig = x1 + x1_crop
                y1_orig = y1 + y1_crop
                x2_orig = x1 + x2_crop
                y2_orig = y1 + y2_crop

                # Draw bounding boxes on the original frame
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert frame to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = (event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point:
            x1, y1 = self.start_point
            x2, y2 = event.x(), event.y()
            self.selection_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.selection_made = True  # Enable object detection after selection
            self.start_point = None
            self.update_frame()

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())

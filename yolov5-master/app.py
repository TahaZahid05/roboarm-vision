import sys
import cv2
import torch
import numpy as np
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
        self.setGeometry(100, 100, 800, 600)

        # QLabel to display video feed
        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 780, 500)
        self.video_label.setStyleSheet("border: 1px solid black")
        self.selection_rect = None
        self.start_point = None
        self.selection_made = False

        # QLabel to display object info
        self.info_label = QLabel(self)
        self.info_label.setGeometry(10, 520, 780, 50)
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

        # Start capturing video
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to access the webcam.")
            sys.exit(1)

        # Set camera resolution to 640x640
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        self.timer.start(30)

        # Variables for detections and selection
        self.detections = []
        self.selected_object = None


    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            self.info_label.setText("Failed to capture video frame.")
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
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            label_width, label_height = self.video_label.width(), self.video_label.height()
            x, y = (event.x() - label_pos.x()) * frame_width // label_width, (event.y() - label_pos.y()) * frame_height // label_height

            if self.selection_made:
                for detection in self.detections:
                    x1, y1, x2, y2 = detection['bbox']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.selected_object = detection
                        self.info_label.setText(f"Selected: {detection['label']} ({x1}, {y1}) to ({x2}, {y2})")
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

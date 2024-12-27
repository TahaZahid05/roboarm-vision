import cv2

# Replace with your RTSP URL
rtsp_url = "rtsp://192.168.0.36:8080/h264_ulaw.sdp"

# Open the video stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open video stream.")
else:
    print("Video stream opened successfully.")

# Loop to display the frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Display the frame
    cv2.imshow("RTSP Stream", frame)
    
    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

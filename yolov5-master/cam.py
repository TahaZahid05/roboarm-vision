import cv2
import requests
import numpy as np

# Replace with your mobile's single frame image URL
image_url = "http://192.168.0.101:8080/shot.jpg"

while True:
    try:
        # Fetch the image from the URL
        response = requests.get(image_url, stream=True)
        if not response.ok:
            print("Error: Unable to fetch image")
            break

        # Convert the image to a NumPy array
        image_array = np.array(bytearray(response.content), dtype=np.uint8)

        # Decode the image to OpenCV format
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Resize the frame to 640x640
        frame_resized = cv2.resize(frame, (640, 640))

        # Display the resized frame
        cv2.imshow("Mobile Stream (Resized to 640x640)", frame_resized)

        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error: {e}")
        break

# Release resources
cv2.destroyAllWindows()

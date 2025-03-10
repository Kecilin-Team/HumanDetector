import cv2
import requests
import numpy as np
import base64
import io
from PIL import Image

SERVER_URL = "http://localhost:5000/detect"

def send_image(image_path):
    """ Sends an image to the server for detection and saves the result. """
    # with open(image_path, "rb") as img_file:
    frame = cv2.imread(image_path)

    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
    data = {"mode": "image"}  # Use image mode since we send frames one by one

    response = requests.post(SERVER_URL, files=files, data=data)
    if response.status_code == 200:
        result = response.json()['results']
        person_count = result['total_persons']

        detections = result['detections']

        # Draw bounding boxes and labels
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {detection['id']}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display total person count
        cv2.putText(frame, f"Total Persons: {person_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite('results.png', frame)
        cv2.imshow('results', frame)
        cv2.waitKey(0)
        print('results saved successfully')
    else:
        print("Error:", response.json())

def send_live_feed():
    """ Streams webcam frames to the server for real-time detection. """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {"mode": "image"}  # Use image mode since we send frames one by one

        response = requests.post(SERVER_URL, files=files, data=data)

        if response.status_code == 200:
            result = response.json()['results']
            person_count = result['total_persons']

            detections = result['detections']

            # Draw bounding boxes and labels
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {detection['id']}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display total person count
            cv2.putText(frame, f"Total Persons: {person_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display processed frame
            cv2.imshow("Processed Frame", frame)
        else:
            print("Error:", response.json())

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Enter 'image' for image detection or 'live' for webcam detection: ").strip().lower()
    
    if mode == "image":
        image_path = input("Enter the image path: ").strip()
        send_image(image_path)
    elif mode == "live":
        send_live_feed()
    else:
        print("Invalid option. Please choose 'image' or 'live'.")

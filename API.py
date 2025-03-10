from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import base64

app = Flask(__name__)

# Load YOLO model
model = YOLO("./weights/yolov8_best.pt")

def process_frame(img):
    results = model(img)
    person_count = 0
    detections = []
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            if cls == 0:  # Class ID 0 corresponds to "person"
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({'id': person_count, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})
    
    return  {'total_persons': person_count, 'detections': detections}

@app.route('/detect', methods=['POST'])
def detect_humans():
    try:
        mode = request.form.get('mode', 'image')

        if mode == 'image':
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            results = process_frame(img)
            
            response = {
                'results': results,
            }
            return jsonify(response)
        
        elif mode == 'live':
            return jsonify({'message': 'Client should start webcam and send frames for processing'})
        
        else:
            return jsonify({'error': 'Invalid mode. Use "image" or "live"'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

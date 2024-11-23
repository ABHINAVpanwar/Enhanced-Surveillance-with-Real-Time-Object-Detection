import cv2
import numpy as np
import os
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, message_queue=os.getenv('REDIS_URL'))  # Redis message queue for production

# YOLO setup
model_weights_path = '/opt/render/project/src/application/yolov3-tiny.weights'
model_cfg_path = '/opt/render/project/src/application/yolov3-tiny.cfg'
coco_names_path = '/opt/render/project/src/application/coco.names'

net = cv2.dnn.readNet(model_weights_path, model_cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

confidence_threshold = 0.5

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Set a higher resolution for better detection clarity
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

def generate_frames():
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with YOLO
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

        # Draw bounding boxes and labels on the frame with improved clarity
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)  # Green by default
                if classes[class_ids[i]] == "person":
                    color = (0, 0, 255)  # Red for person
                # Increase box thickness and font size for better clarity
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Encode the frame to JPEG and yield it to the client
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the app with Gunicorn in production, Flask will bind to the correct port dynamically
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

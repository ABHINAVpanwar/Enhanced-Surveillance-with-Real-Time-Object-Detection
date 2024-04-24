import os
import cv2
import numpy as np

# Load the pre-trained YOLO model
net = cv2.dnn.readNet("C:/Users/Abhinav/OneDrive/Documents/PYTHON/ML_mini_project/yolov3.weights", "C:/Users/Abhinav/OneDrive/Documents/PYTHON/ML_mini_project/yolov3.cfg")
classes = []
with open("C:/Users/Abhinav/OneDrive/Documents/PYTHON/ML_mini_project/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Open camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, 1 for the next connected camera, and so on.

# Set confidence threshold
confidence_threshold = 0.5

# Create directory to save images
output_folder = "detected_images"
os.makedirs(output_folder, exist_ok=True)

# Initialize image counter
image_counter = 0
max_images = 50  # Maximum number of images to save

while True:
    ret, frame = cap.read()  # Read frame from the camera
    if not ret:
        print("Failed to read frame from the camera.")
        break

    if frame is None:
        continue

    height, width, channels = frame.shape

    # Preprocess image
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw the detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # If person is detected and image counter is less than maximum
            if classes[class_ids[i]] == "person" and image_counter < max_images:
                image_counter += 1
                image_name = f"person_{image_counter}.jpg"
                image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(image_path, frame)
                print(f"Image saved as {image_path}")

                # Draw a warning message
                cv2.putText(frame, f"Warning: Person detected! Image saved ({image_counter}/{max_images})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show results
    cv2.imshow("Object Detection", frame)

    # Exit loop if 'q' is pressed or maximum images reached
    if cv2.waitKey(1) & 0xFF == ord('q') or image_counter >= max_images:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

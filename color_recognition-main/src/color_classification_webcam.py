#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import time # For FPS calculation

cap = cv2.VideoCapture(0) # Use 0 for default webcam, 1 for external if 0 doesn't work
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prediction = 'n.a.'
votes = 0
k_used = 0
frame_count = 0
start_time = time.time()

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK) and os.path.getsize(PATH) > 0:
    print ('Training data is ready, classifier is loading...')
else:
    print ('Training data is being created...')
    # Ensure the training.data file is created/cleared before training
    color_histogram_feature_extraction.training()
    print ('Training data is ready, classifier is loading...')

while True:
    # Capture frame-by-frame
    (ret, frame) = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # --- Visual Enhancements ---
    # Get frame dimensions
    h, w, _ = frame.shape

    # Define a central region of interest (ROI) for visual emphasis
    # This rectangle indicates the area from which the color is primarily analyzed.
    roi_x1, roi_y1 = int(w * 0.3), int(h * 0.3)
    roi_x2, roi_y2 = int(w * 0.7), int(h * 0.7)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2) # Green rectangle, 2px thickness

    # Add prediction text
    text_prediction = f'Color: {prediction}'
    text_confidence = f'Confidence: {votes}/{k_used}' # Display votes out of K neighbors
    text_fps = f'FPS: {fps:.2f}'

    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (0, 0, 255) # Red color for text

    # Position text (top-left corner, with padding)
    cv2.putText(frame, text_prediction, (20, 40), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame, text_confidence, (20, 80), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame, text_fps, (w - 150, 40), font, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA) # Yellow for FPS

    # Display the resulting frame
    cv2.imshow('Color Classifier - Webcam', frame)

    # Process color only every few frames to improve performance
    if frame_count % 5 == 0: # Process every 5th frame
        # Extract features from the current frame's ROI
        # You might want to extract features from the whole frame or a specific ROI
        # For simplicity, we'll use the whole frame as the original code does.
        color_histogram_feature_extraction.color_histogram_of_test_image(frame)
        prediction, votes, k_used = knn_classifier.main('training.data', 'test.data', k_value=5) # Using k=5
        # print(f'Detected color: {prediction} (Votes: {votes}/{k_used})') # For console output

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


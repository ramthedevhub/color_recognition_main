#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import sys

# read the test image
image_path = 'black_cat.jpg' # Default image
if len(sys.argv) > 1:
    image_path = sys.argv[1]

source_image = cv2.imread(image_path)

if source_image is None:
    print(f"Error: Could not load image from {image_path}. Please check the path.")
    sys.exit(1)

prediction = 'n.a.'
votes = 0
k_used = 0

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK) and os.path.getsize(PATH) > 0:
    print ('Training data is ready, classifier is loading...')
else:
    print ('Training data is being created...')
    # Ensure the training.data file is created/cleared before training
    color_histogram_feature_extraction.training()
    print ('Training data is ready, classifier is loading...')

# get the prediction
color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
prediction, votes, k_used = knn_classifier.main('training.data', 'test.data', k_value=5) # Using k=5 for potentially better stability
print(f'Detected color is: {prediction} (Votes: {votes}/{k_used})')

# --- Visual Enhancements ---
# Get image dimensions
h, w, _ = source_image.shape

# Define a central region of interest (ROI) for visual emphasis
# This rectangle indicates the area from which the color is primarily analyzed.
roi_x1, roi_y1 = int(w * 0.3), int(h * 0.3)
roi_x2, roi_y2 = int(w * 0.7), int(h * 0.7)
cv2.rectangle(source_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2) # Green rectangle, 2px thickness

# Add prediction text
text_prediction = f'Color: {prediction}'
text_confidence = f'Confidence: {votes}/{k_used}' # Display votes out of K neighbors

# Text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 3
text_color = (0, 0, 255) # Red color for text

# Calculate text size to position it better
(text_w_pred, text_h_pred), _ = cv2.getTextSize(text_prediction, font, font_scale, font_thickness)
(text_w_conf, text_h_conf), _ = cv2.getTextSize(text_confidence, font, font_scale, font_thickness)

# Position text (top-left corner, with padding)
cv2.putText(source_image, text_prediction, (20, text_h_pred + 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
cv2.putText(source_image, text_confidence, (20, text_h_pred + text_h_conf + 40), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Display the resulting frame
cv2.imshow('Color Classifier - Image', source_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Plot histogram for the test image (for debugging/analysis)
# test_image_for_hist = cv2.imread(image_path)
# if test_image_for_hist is not None:
#     color_histogram_feature_extraction.plot_color_histogram(test_image_for_hist, title=f"Histogram for {prediction} image")

Camera Calc

Camera Calc is a Python-based hand tracking calculator that uses OpenCV and MediaPipe for gesture recognition. This project enables users to interact with a webcam to perform calculations based on hand gestures, leveraging the K-Nearest-Neighbors (KNN) algorithm for classification.

Features:
  Real-time Hand Tracking: Utilizes MediaPipe to track the movements of the index fingertip.
  Path Normalization: Captures and normalizes fingertip movement paths to create consistent features.
  Gesture-based Input: Translates hand gestures into numeric or operational inputs.
  Simple Calculations: Supports basic arithmetic calculations using gestures.
  CSV Data Storage: Saves gesture path data in CSV files for training and debugging purposes.

Technologies Used:
  Python.
  OpenCV: For webcam integration and image processing.
  MediaPipe: For efficient hand landmark detection.
  K-Nearest-Neighbors (KNN): Algorithm for classifying gestures.

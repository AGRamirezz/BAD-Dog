# -*- coding: utf-8 -*-
"""
Face Detection and Analysis Module

This module provides functionality for real-time face detection and analysis
using a webcam stream and a pre-trained neural network model.
"""

import cv2
import numpy as np
import PIL
from PIL import Image
import io
from base64 import b64decode, b64encode
import tensorflow as tf
from tensorflow import keras
import time
import os

# Uncomment if you need these packages installed
# import subprocess
# subprocess.check_call(['pip', 'install', 'keras'])
# subprocess.check_call(['pip', 'install', 'opencv-python'])
# subprocess.check_call(['pip', 'install', 'tensorflow'])


class FaceAnalyzer:
    """
    A class for analyzing faces in real-time video streams.
    
    This class handles webcam capture, face detection using Haar Cascades,
    and classification using a pre-trained neural network.
    """
    
    def __init__(self, model_path='BADNet_Acc.h5'):
        """
        Initialize the FaceAnalyzer with detection models.
        
        Args:
            model_path (str): Path to the pre-trained Keras model
        """
        # Initialize the Haar Cascade face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        )
        
        # Load the pre-trained model if the file exists
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            self.model.summary()
        else:
            self.model = None
            print(f"Warning: Model file {model_path} not found")
    
    def js_to_image(self, js_reply):
        """
        Convert a JavaScript object containing image data to an OpenCV image.
        
        Args:
            js_reply: JavaScript object containing base64 encoded image from webcam
            
        Returns:
            img: OpenCV BGR image
        """
        # Decode base64 image
        image_bytes = b64decode(js_reply.split(',')[1])
        
        # Convert bytes to numpy array
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Decode numpy array into OpenCV BGR image
        img = cv2.imdecode(jpg_as_np, flags=1)
        
        return img
    
    def bbox_to_bytes(self, bbox_array):
        """
        Convert an OpenCV Rectangle bounding box image into base64 byte string.
        
        This allows the bounding box to be overlaid on a video stream.
        
        Args:
            bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream
            
        Returns:
            bytes: Base64 image byte string
        """
        # Convert array into PIL image
        bbox_PIL = Image.fromarray(bbox_array, 'RGBA')
        iobuf = io.BytesIO()
        
        # Format bbox into png for return
        bbox_PIL.save(iobuf, format='png')
        
        # Format return string
        bbox_bytes = 'data:image/png;base64,{}'.format(
            (str(b64encode(iobuf.getvalue()), 'utf-8'))
        )
        
        return bbox_bytes
    
    def process_frame(self, img):
        """
        Process a single frame to detect and analyze faces.
        
        Args:
            img: OpenCV image to process
            
        Returns:
            bbox_bytes: Base64 encoded image with bounding boxes
        """
        # Create transparent overlay for bounding box
        bbox_array = np.zeros([224, 224, 4], dtype=np.uint8)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Get face region coordinates
        faces = self.face_cascade.detectMultiScale(gray)
        
        if self.model is not None and len(faces) > 0:
            # Prepare image for model inference
            img2 = np.expand_dims(img, axis=0)
            
            # Get model predictions
            out = self.model.predict(img2, batch_size=1, verbose=0)
            yhat_probs = np.max(out, axis=-1)
            yhat_class = np.argmax(out, axis=-1).flatten()
            
            # Draw bounding boxes based on model predictions
            for (x, y, w, h) in faces:
                if yhat_class[0] == 0:
                    # Green box for class 0
                    bbox_array = cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                elif yhat_class[0] == 1:
                    if yhat_probs > 0.94:
                        # Blue box for high confidence class 1
                        bbox_array = cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    elif yhat_probs < 0.85:
                        # Green box for low confidence class 1
                        bbox_array = cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    else:
                        # Red box for medium confidence class 1
                        bbox_array = cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            # If no model or no faces, just draw boxes around detected faces
            for (x, y, w, h) in faces:
                bbox_array = cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Set alpha channel for transparency
        bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
        
        # Convert overlay to bytes
        bbox_bytes = self.bbox_to_bytes(bbox_array)
        
        return bbox_bytes
    
    def run_webcam(self, video_stream_func=None, video_frame_func=None):
        """
        Run the face analyzer on a webcam stream.
        
        Args:
            video_stream_func: Function to initialize video streaming
            video_frame_func: Function to get a frame from the video stream
        """
        if video_stream_func is None or video_frame_func is None:
            print("Error: Video stream functions not provided")
            print("This function requires external video stream handling functions")
            return
        
        # Start streaming video from webcam
        video_stream_func()
        
        # Label for video
        label_html = 'Capturing...'
        
        # Initialize bounding box to empty
        bbox = ''
        
        try:
            while True:
                # Get frame from video stream
                js_reply = video_frame_func(label_html, bbox)
                
                if not js_reply:
                    break
                
                # Convert JS response to OpenCV Image
                img = self.js_to_image(js_reply["img"])
                
                # Process the frame
                bbox = self.process_frame(img)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("Stopping video capture...")
        
        print("Video capture ended")


def run_opencv_camera():
    """
    Alternative implementation using OpenCV's built-in camera functions.
    This can be used instead of JavaScript-based webcam access.
    """
    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Create transparent overlay for bounding box
            bbox_array = np.zeros([frame.shape[0], frame.shape[1], 4], dtype=np.uint8)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get face region coordinates
            faces = analyzer.face_cascade.detectMultiScale(gray)
            
            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Face Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release the capture when done
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage with OpenCV camera
    print("Starting face detection with OpenCV camera...")
    run_opencv_camera()
    
    # Note: The original JavaScript-based implementation requires
    # the video_stream and video_frame functions which were not provided
    # in the original code. If you have these functions, you can use:
    #
    # analyzer = FaceAnalyzer()
    # analyzer.run_webcam(video_stream, video_frame)

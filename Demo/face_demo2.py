"""
Face Detection and Emotion Analysis Module

This module provides functionality for real-time face detection and emotion analysis
using a webcam stream and DeepFace for facial emotion recognition.
"""

import cv2
import numpy as np
import time
import os
from deepface import DeepFace

class FaceEmotionAnalyzer:
    """
    A class for analyzing facial emotions in real-time video streams.
    
    This class handles webcam capture, face detection using Haar Cascades,
    and emotion recognition using DeepFace.
    """
    
    def __init__(self):
        """
        Initialize the FaceEmotionAnalyzer with detection models.
        """
        # Initialize the Haar Cascade face detection model
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Error: Could not load face cascade classifier from {cascade_path}")
        
        print("Face detection model loaded successfully")
        
        # Define emotion colors (BGR format)
        self.emotion_colors = {
            'angry': (0, 0, 255),    # Red
            'disgust': (0, 140, 255), # Orange
            'fear': (0, 0, 128),     # Dark red
            'happy': (0, 255, 0),    # Green
            'sad': (255, 0, 0),      # Blue
            'surprise': (255, 255, 0), # Cyan
            'neutral': (255, 255, 255) # White
        }
        
    def analyze_emotion(self, face_img):
        """
        Analyze the emotion of a detected face using DeepFace.
        
        Args:
            face_img: Cropped image of a face
            
        Returns:
            dominant_emotion: String representing the detected emotion
            emotion_score: Confidence score for the detected emotion
        """
        try:
            # Use DeepFace to analyze emotions
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
            
            if isinstance(result, list):
                result = result[0]
                
            # Get the dominant emotion and its score
            dominant_emotion = result['dominant_emotion']
            emotion_score = result['emotion'][dominant_emotion]
            
            return dominant_emotion, emotion_score
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            return "unknown", 0
    
    def run_webcam(self):
        """
        Run face detection and emotion analysis on webcam feed.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting face detection and emotion analysis...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture image")
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Analyze emotion
                    emotion, score = self.analyze_emotion(face_roi)
                    
                    # Get color for the detected emotion
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Create a label with emotion and confidence
                    label = f"{emotion} ({score:.2f}%)"
                    
                    # Calculate position for text
                    label_y = y - 10 if y - 10 > 10 else y + h + 10
                    
                    # Add text with the emotion label
                    cv2.putText(
                        display_frame,
                        label,
                        (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                
                # Display the resulting frame
                cv2.imshow('Facial Emotion Analysis', display_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
        
        finally:
            # Release the capture when done
            cap.release()
            cv2.destroyAllWindows()
            print("Video capture ended")


if __name__ == "__main__":
    print("Starting facial emotion analysis using DeepFace...")
    analyzer = FaceEmotionAnalyzer()
    analyzer.run_webcam()

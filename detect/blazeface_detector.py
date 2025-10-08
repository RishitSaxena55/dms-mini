"""
BlazeFace face detector wrapper.
Uses MediaPipe or TFLite backend for fast CPU inference.
"""

import cv2
import numpy as np
from pathlib import Path


class BlazeFaceDetector:
    """Lightweight face detector using MediaPipe BlazeFace."""

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Args:
            min_detection_confidence: Minimum confidence threshold
            model_selection: 0 for short-range (<2m), 1 for full-range
        """
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence,
                model_selection=model_selection
            )
            self.backend = 'mediapipe'
        except ImportError:
            print("MediaPipe not available, falling back to OpenCV Haar Cascade")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.backend = 'opencv'

    def detect(self, image):
        """
        Detect faces in image.

        Args:
            image: BGR image (numpy array)

        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        if self.backend == 'mediapipe':
            return self._detect_mediapipe(image)
        else:
            return self._detect_opencv(image)

    def _detect_mediapipe(self, image):
        """Detect using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure valid bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                faces.append((x, y, width, height))

        return faces

    def _detect_opencv(self, image):
        """Detect using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [tuple(face) for face in faces]

    def crop_face(self, image, bbox, margin=0.2, target_size=(64, 64)):
        """
        Crop and resize face region with margin.

        Args:
            image: Source image
            bbox: Face bounding box (x, y, w, h)
            margin: Margin ratio to add around face
            target_size: Output size (width, height)

        Returns:
            Cropped and resized face image
        """
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]

        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        # Crop and resize
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, target_size)

        return face

    def __del__(self):
        if hasattr(self, 'detector') and self.backend == 'mediapipe':
            self.detector.close()

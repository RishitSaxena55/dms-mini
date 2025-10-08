"""
Real-time emotion detection from webcam with optional seat-belt detection.
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from detect.blazeface_detector import BlazeFaceDetector


class EmotionClassifier:
    """Wrapper for emotion classification inference."""

    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    COLORS = [
        (0, 0, 255),  # Angry - red
        (0, 128, 0),  # Disgust - green
        (128, 0, 128),  # Fear - purple
        (0, 255, 255),  # Happy - yellow
        (255, 0, 0),  # Sad - blue
        (255, 165, 0),  # Surprise - orange
        (128, 128, 128)  # Neutral - gray
    ]

    def __init__(self, model_path, img_size=64, backend='onnx'):
        self.img_size = img_size
        self.backend = backend

        if backend == 'onnx':
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
        elif backend == 'tflite':
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def preprocess(self, face):
        """Preprocess face for inference."""
        # Resize
        face = cv2.resize(face, (self.img_size, self.img_size))

        # Convert to grayscale
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Normalize
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5

        # Add batch and channel dims
        face = face[np.newaxis, np.newaxis, ...]

        return face

    def predict(self, face):
        """Predict emotion from face."""
        img = self.preprocess(face)

        if self.backend == 'onnx':
            output = self.session.run(None, {'input': img})[0]
        else:  # tflite
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Softmax
        output = np.exp(output - np.max(output))
        output = output / output.sum()

        pred_idx = np.argmax(output)
        confidence = output[0, pred_idx]

        return self.EMOTIONS[pred_idx], confidence, self.COLORS[pred_idx]


def draw_results(frame, bbox, emotion, confidence, color):
    """Draw bounding box and emotion label."""
    x, y, w, h = bbox

    # Draw box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Draw label
    label = f"{emotion}: {confidence:.2f}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(frame, (x, y - label_size[1] - 10),
                  (x + label_size[0], y), color, -1)
    cv2.putText(frame, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description='Real-time emotion detection')
    parser.add_argument('--detector', type=str, default='blazeface',
                        help='Face detector type')
    parser.add_argument('--classifier', type=str, required=True,
                        help='Emotion classifier model (.onnx or .tflite)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--img-size', type=int, default=64, help='Classifier input size')
    parser.add_argument('--seatbelt', action='store_true', help='Enable seat-belt detection')

    args = parser.parse_args()

    # Initialize detector and classifier
    print("Loading models...")
    detector = BlazeFaceDetector()

    backend = 'onnx' if args.classifier.endswith('.onnx') else 'tflite'
    classifier = EmotionClassifier(args.classifier, args.img_size, backend)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return

    print("Press 'q' to quit")

    fps_counter = []

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect(frame)

        # Process each face
        for bbox in faces:
            x, y, w, h = bbox

            # Crop face
            face_crop = frame[y:y + h, x:x + w]

            if face_crop.size == 0:
                continue

            # Predict emotion
            emotion, confidence, color = classifier.predict(face_crop)

            # Draw results
            draw_results(frame, bbox, emotion, confidence, color)

            # Seat-belt ROI (bonus)
            if args.seatbelt and len(faces) > 0:
                roi_y = min(y + int(h * 1.2), frame.shape[0])
                roi_h = min(int(h * 0.6), frame.shape[0] - roi_y)
                cv2.rectangle(frame, (0, roi_y), (frame.shape[1], roi_y + roi_h),
                              (255, 255, 0), 2)
                cv2.putText(frame, "Seat-belt ROI", (10, roi_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        fps_counter.append(fps)
        if len(fps_counter) > 30:
            fps_counter.pop(0)

        avg_fps = np.mean(fps_counter)

        # Draw FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('DMS Mini - Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {np.mean(fps_counter):.2f}")


if __name__ == '__main__':
    main()

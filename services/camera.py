"""Webcam and Haar face detection for registration UI."""
import sys
import cv2

_haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def open_camera():
    """Open webcam; use DirectShow on Windows to avoid MSMF warning/errors."""
    if sys.platform == 'win32':
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cv2.VideoCapture(0)


def face_boxes_haar(frame):
    """Detect faces with Haar; returns list of (x, y, w, h)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if _haar_face.empty():
        return []
    rects = _haar_face.detectMultiScale(gray, 1.3, 5)
    return list(rects)

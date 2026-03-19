"""Shared app-level objects (locks, etc.)."""
import threading

# Serialize camera + face_recognition work across Take Attendance / Add User / delete retrain
camera_lock = threading.Lock()

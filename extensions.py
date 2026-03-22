"""Shared app-level objects (locks, DB, etc.)."""
import threading

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Serialize camera + face_recognition work across Take Attendance / Add User / delete retrain
camera_lock = threading.Lock()

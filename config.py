"""Paths, dates, MySQL URI, and startup directory setup."""
import os
from datetime import date

# Dates (computed at import; restart app for a new day)
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Direct mode: static/faces + Attendance/ CSV
# KNN mode: static/faces_KNN + MySQL (users + attendance)
FACES_DIR = 'static/faces'
FACES_KNN_DIR = 'static/faces_KNN'
ATTENDANCE_DIR = 'Attendance'

KNOWN_FACES_PATH = 'static/known_faces.pkl'
KNN_MODEL_PATH = 'static/trained_knn_model.clf'

SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-production')

# MySQL for KNN attendance — create DB first, e.g. CREATE DATABASE face_attendance;
MYSQL_URI = os.environ.get(
    'MYSQL_URI',
    'mysql+pymysql://root:@127.0.0.1/face_attendance',
)


def faces_dir(use_knn):
    return FACES_KNN_DIR if use_knn else FACES_DIR


def attendance_dir(use_knn):
    """CSV folder only used for direct mode."""
    return ATTENDANCE_DIR


def attendance_csv_path(use_knn):
    return os.path.join(attendance_dir(use_knn), f'Attendance-{datetoday}.csv')


def ensure_data_dirs():
    """Create face folders and direct-mode attendance CSV only (no Attendance_KNN)."""
    for _dir in (ATTENDANCE_DIR, FACES_DIR, FACES_KNN_DIR):
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

    _csv = attendance_csv_path(False)
    _folder = attendance_dir(False)
    if f'Attendance-{datetoday}.csv' not in os.listdir(_folder):
        with open(_csv, 'w') as f:
            f.write('Name,Roll,Date,Check-In,Check-Out')

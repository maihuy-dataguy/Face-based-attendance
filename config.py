"""Paths, dates, and startup directory setup."""
import os
from datetime import date

# Dates (computed at import; restart app for a new day)
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Non-KNN (direct): static/faces + Attendance/
# KNN: static/faces_KNN + Attendance_KNN/
FACES_DIR = 'static/faces'
FACES_KNN_DIR = 'static/faces_KNN'
ATTENDANCE_DIR = 'Attendance'
ATTENDANCE_KNN_DIR = 'Attendance_KNN'

KNOWN_FACES_PATH = 'static/known_faces.pkl'
KNN_MODEL_PATH = 'static/trained_knn_model.clf'

SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-production')


def faces_dir(use_knn):
    return FACES_KNN_DIR if use_knn else FACES_DIR


def attendance_dir(use_knn):
    return ATTENDANCE_KNN_DIR if use_knn else ATTENDANCE_DIR


def attendance_csv_path(use_knn):
    return os.path.join(attendance_dir(use_knn), f'Attendance-{datetoday}.csv')


def ensure_data_dirs():
    """Create face + attendance folders and today's CSVs if missing."""
    for _dir in (ATTENDANCE_DIR, ATTENDANCE_KNN_DIR, FACES_DIR, FACES_KNN_DIR):
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

    for _use_knn in (False, True):
        _csv = attendance_csv_path(_use_knn)
        _folder = attendance_dir(_use_knn)
        if f'Attendance-{datetoday}.csv' not in os.listdir(_folder):
            with open(_csv, 'w') as f:
                f.write('Name,Roll,Date,Check-In,Check-Out')

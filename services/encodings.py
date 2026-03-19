"""Face encodings: known faces pickle, detection helpers, direct matching."""
import os
import pickle

import face_recognition
import numpy as np

from config import FACES_DIR, KNOWN_FACES_PATH


def _build_known_faces_from_folders():
    """Build known face encodings from static/faces/{name_id}/*."""
    encodings = []
    names = []
    if not os.path.isdir(FACES_DIR):
        return encodings, names
    for user in os.listdir(FACES_DIR):
        user_path = os.path.join(FACES_DIR, user)
        if not os.path.isdir(user_path):
            continue
        user_encodings = []
        for fname in os.listdir(user_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(user_path, fname)
            try:
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    user_encodings.append(encs[0])
            except Exception:
                continue
        if user_encodings:
            encodings.append(np.mean(user_encodings, axis=0))
            names.append(user)
    return encodings, names


def get_known_faces():
    """Load (encodings, names) from pickle or build from static/faces."""
    if os.path.isfile(KNOWN_FACES_PATH):
        try:
            with open(KNOWN_FACES_PATH, 'rb') as f:
                data = pickle.load(f)
            if data.get('encodings') and data.get('names'):
                return data['encodings'], data['names']
        except Exception:
            pass
    encodings, names = _build_known_faces_from_folders()
    if encodings:
        with open(KNOWN_FACES_PATH, 'wb') as f:
            pickle.dump({'encodings': encodings, 'names': names}, f)
    return encodings, names


def extract_faces_rgb(rgb_frame, small_frame=None):
    """
    Detect faces using face_recognition. Returns list of (x, y, w, h) in original frame coordinates.
    """
    scale = 4 if small_frame is not None else 1
    frame_to_use = small_frame if small_frame is not None else rgb_frame
    face_locations = face_recognition.face_locations(frame_to_use)
    out = []
    for (top, right, bottom, left) in face_locations:
        x = left * scale
        y = top * scale
        w = (right - left) * scale
        h = (bottom - top) * scale
        out.append((x, y, w, h))
    return out


def identify_face_from_encoding(face_encoding, known_encodings, known_names, tolerance=0.6):
    """Match one face encoding to known encodings. Returns name or None."""
    if not known_encodings or not known_names:
        return None
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = np.argmin(face_distances)
    if matches[best_idx]:
        return known_names[best_idx]
    return None


def rebuild_known_faces_pickle_full():
    """Rebuild known_faces.pkl from static/faces (direct mode)."""
    if os.path.isfile(KNOWN_FACES_PATH):
        try:
            os.remove(KNOWN_FACES_PATH)
        except OSError:
            pass
    encodings, names = _build_known_faces_from_folders()
    if encodings:
        with open(KNOWN_FACES_PATH, 'wb') as f:
            pickle.dump({'encodings': encodings, 'names': names}, f)

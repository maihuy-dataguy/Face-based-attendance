"""Sklearn KNN training and prediction."""
import math
import os
import pickle

import face_recognition
from sklearn import neighbors

from config import FACES_KNN_DIR, KNN_MODEL_PATH

from services import storage as storage_mod


def train_knn(train_dir, model_save_path, n_neighbors=None, knn_algo='ball_tree'):
    """
    Train a KNN classifier. Each subdir in train_dir is one person;
    images with exactly one face are used.
    """
    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_path, fname)
            try:
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)
                if len(face_bounding_boxes) != 1:
                    continue
                enc = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                X.append(enc)
                y.append(class_dir)
            except Exception:
                continue
    if len(X) == 0:
        return None
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)
    return knn_clf


def predict_with_knn(face_encoding, model_path=None, distance_threshold=0.6):
    """Predict name for one face encoding; returns name or None if unknown."""
    path = model_path or KNN_MODEL_PATH
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        knn_clf = pickle.load(f)
    pred = knn_clf.predict([face_encoding])[0]
    dists = knn_clf.kneighbors([face_encoding], n_neighbors=1)
    if dists[0][0][0] <= distance_threshold:
        return pred
    return None


def retrain_knn_model():
    """Retrain or remove KNN file after user folder changes."""
    if storage_mod.totalreg(True) == 0:
        if os.path.isfile(KNN_MODEL_PATH):
            try:
                os.remove(KNN_MODEL_PATH)
            except OSError:
                pass
        return
    train_knn(FACES_KNN_DIR, KNN_MODEL_PATH)

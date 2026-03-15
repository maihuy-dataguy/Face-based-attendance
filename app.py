import asyncio
import math
import threading
import cv2
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from datetime import date, datetime
from sklearn import neighbors

import face_recognition

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

KNOWN_FACES_PATH = 'static/known_faces.pkl'
KNN_MODEL_PATH = 'static/trained_knn_model.clf'

# Only one camera operation at a time (Take Attendance or Add User); prevents crash when both run together
_camera_lock = threading.Lock()

def open_camera():
    """Open webcam; use DirectShow on Windows to avoid MSMF warning/errors."""
    if sys.platform == 'win32':
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Date,Check-In,Check-Out')

# Haar cascade for Add User (capture 50 face crops without running dlib on webcam)
_haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def _face_boxes_haar(frame):
    """Detect faces with Haar; returns list of (x, y, w, h)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if _haar_face.empty():
        return []
    rects = _haar_face.detectMultiScale(gray, 1.3, 5)
    return list(rects)


def train_knn(train_dir, model_save_path, n_neighbors=None, knn_algo='ball_tree'):
    """
    Train a KNN classifier for face recognition (from face_recognition_knn.py).
    Each subdir in train_dir is one person; images with exactly one face are used.
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


def predict_with_knn(face_encoding, model_path, distance_threshold=0.6):
    """Predict name for one face encoding using trained KNN; returns name or None if unknown."""
    if not os.path.isfile(model_path):
        return None
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    pred = knn_clf.predict([face_encoding])[0]
    dists = knn_clf.kneighbors([face_encoding], n_neighbors=1)
    if dists[0][0][0] <= distance_threshold:
        return pred
    return None


def get_known_faces():
    """Load (encodings, names) from pickle or build from static/faces folders."""
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


def _build_known_faces_from_folders():
    """Build known face encodings from static/faces/{name_id}/*.jpg"""
    encodings = []
    names = []
    faces_dir = 'static/faces'
    if not os.path.isdir(faces_dir):
        return encodings, names
    for user in os.listdir(faces_dir):
        user_path = os.path.join(faces_dir, user)
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


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces_rgb(rgb_frame, small_frame=None):
    """
    Detect faces using face_recognition. Returns list of (x, y, w, h) in original frame coordinates.
    If small_frame is provided (e.g. 1/4 size), scale is 4; otherwise scale is 1.
    """
    scale = 4 if small_frame is not None else 1
    if small_frame is not None:
        frame_to_use = small_frame
    else:
        frame_to_use = rgb_frame
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


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    if 'Date' in df.columns:
        dates = df['Date'].fillna(datetoday2).astype(str)
    else:
        dates = pd.Series([datetoday2] * len(df))
    if 'Check-In' in df.columns:
        check_ins = df['Check-In'].astype(str)
        check_outs = df['Check-Out'].fillna('').astype(str)
    else:
        check_ins = df['Time'].astype(str) if 'Time' in df.columns else pd.Series([''] * len(df))
        check_outs = pd.Series([''] * len(df))
    l = len(df)
    return names, rolls, dates, check_ins, check_outs, l


def _ensure_attendance_format(df):
    if 'Date' not in df.columns:
        df.insert(2, 'Date', datetoday2)
    if 'Check-In' not in df.columns:
        if 'Time' in df.columns:
            df = df.rename(columns={'Time': 'Check-In'})
            df['Check-Out'] = ''
        else:
            df['Check-In'] = ''
            df['Check-Out'] = ''
    for col in ['Check-In', 'Check-Out']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    if 'Date' in df.columns:
        df['Date'] = df['Date'].fillna(datetoday2).astype(str)
    return df


def _is_checked_in_today(name):
    userid = name.split('_')[1]
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if 'Roll' not in df.columns:
        return False
    return int(userid) in list(df['Roll'])


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    df = pd.read_csv(csv_path)
    df = _ensure_attendance_format(df)
    roll_int = int(userid)
    if roll_int not in list(df['Roll']):
        new_row = pd.DataFrame([{'Name': username, 'Roll': roll_int, 'Date': datetoday2, 'Check-In': current_time, 'Check-Out': ''}])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df.loc[df['Roll'] == roll_int, 'Check-Out'] = current_time
    df.to_csv(csv_path, index=False)


################## ROUTING #########################

@app.route('/')
async def home():
    names, rolls, dates, check_ins, check_outs, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l, totalreg=totalreg(), datetoday2=datetoday2)


def _attendance_camera():
    """Open camera, wait for same face 5 sec, then record check-in or check-out once."""
    cap = open_camera()
    face_detected_time = None
    detected_person = None
    process_this_frame = True
    last_box = None
    last_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = small_frame[:, :, ::-1]
            face_boxes = extract_faces_rgb(rgb_small, small_frame=rgb_small)
            # Only update last_box/last_label when a face is detected; do not clear on missed frames to avoid flicker
            if face_boxes:
                (x, y, w, h) = face_boxes[0]
                rgb_frame = frame[:, :, ::-1]
                top, right, bottom, left = y, x + w, y + h, x
                face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
                if face_encodings:
                    identified_person = predict_with_knn(face_encodings[0], KNN_MODEL_PATH)
                    if identified_person:
                        display_name = identified_person.split('_')[0]
                        action = 'Check-out' if _is_checked_in_today(identified_person) else 'Check-in'
                        last_label = f'{display_name}  |  {action}'
                        last_box = (x, y, w, h)
                        text_y = max(y - 8, 24)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                        cv2.putText(frame, last_label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 20), 2, cv2.LINE_AA)

                        if face_detected_time is None:
                            face_detected_time = time.time()
                            detected_person = identified_person
                        elif detected_person != identified_person:
                            face_detected_time = time.time()
                            detected_person = identified_person

                        if time.time() - face_detected_time >= 2:
                            add_attendance(detected_person)
                            break
                    else:
                        face_detected_time = None
                        detected_person = None
                        last_box = (x, y, w, h)
                        last_label = 'Unknown'
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                        cv2.putText(frame, 'Unknown', (x, max(y - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    face_detected_time = None
                    detected_person = None
                    last_box = (x, y, w, h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            else:
                # No face this frame: keep last_box and timer so display and hold state don't reset on brief misses
                pass
        elif last_box is not None:
            x, y, w, h = last_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            if last_label:
                cv2.putText(frame, last_label, (x, max(y - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 20) if last_label != 'Unknown' else (0, 0, 255), 2, cv2.LINE_AA)

        process_this_frame = not process_this_frame
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return extract_attendance()


def _check_model_and_reg():
    if totalreg() == 0:
        names, rolls, dates, check_ins, check_outs, l = extract_attendance()
        return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                    totalreg=totalreg(), datetoday2=datetoday2, mess='No users in database. Please add a new user first.'), None
    if not os.path.isfile(KNN_MODEL_PATH):
        if totalreg() > 0:
            train_knn('static/faces', KNN_MODEL_PATH)
        if not os.path.isfile(KNN_MODEL_PATH):
            names, rolls, dates, check_ins, check_outs, l = extract_attendance()
            return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                        totalreg=totalreg(), datetoday2=datetoday2,
                        mess='No trained model. Please add a new user first.'), None
    return None, None


@app.route('/start', methods=['GET'])
async def start():
    if not _camera_lock.acquire(blocking=False):
        names, rolls, dates, check_ins, check_outs, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2,
                               mess='Camera is in use (e.g. Add User or another Take Attendance). Please wait and try again.')
    try:
        err, _ = _check_model_and_reg()
        if err is not None:
            return render_template('home.html', **err)
        names, rolls, dates, check_ins, check_outs, l = await asyncio.to_thread(_attendance_camera)
        return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2)
    finally:
        _camera_lock.release()


def _add_user(newusername, newuserid):
    """Blocking: capture 50 face images and train KNN. Returns template data dict."""
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = open_camera()
    i, j = 0, 0
    target_count = 50

    while i < target_count:
        ret, frame = cap.read()
        if not ret:
            break
        face_boxes = _face_boxes_haar(frame)
        for (x, y, w, h) in face_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            if j % 5 == 0:
                img_path = os.path.join(userimagefolder, f'{newusername}_{i}.jpg')
                face_crop = frame[y:y + h, x:x + w]
                cv2.imwrite(img_path, face_crop)
                i += 1
                if i >= target_count:
                    break
            j += 1
        cv2.putText(frame, f'Images captured: {i}/{target_count}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Adding New User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.isdir(userimagefolder) and len(os.listdir(userimagefolder)) > 0:
        train_knn('static/faces', KNN_MODEL_PATH)
        print(f"Saved 50 images and retrained KNN for {newusername}_{newuserid}")

    names, rolls, dates, check_ins, check_outs, l = extract_attendance()
    return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l)


@app.route('/add', methods=['POST'])
async def add():
    if not _camera_lock.acquire(blocking=False):
        names, rolls, dates, check_ins, check_outs, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2,
                               mess='Camera is in use. Please wait for Take Attendance or Add User to finish, then try again.')
    try:
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        data = await asyncio.to_thread(_add_user, newusername, newuserid)
        return render_template('home.html', **data, totalreg=totalreg(), datetoday2=datetoday2)
    finally:
        _camera_lock.release()


if __name__ == '__main__':
    app.run(debug=True)

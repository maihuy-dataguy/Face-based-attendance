import asyncio
import math
import shutil
import threading
import cv2
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, session, redirect, url_for
from datetime import date, datetime
from sklearn import neighbors

import face_recognition

#### Defining Flask App
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-production')

#### Saving Date today in 2 different formats
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

# Only one camera operation at a time (Take Attendance or Add User); prevents crash when both run together
_camera_lock = threading.Lock()


def faces_dir(use_knn):
    return FACES_KNN_DIR if use_knn else FACES_DIR


def attendance_dir(use_knn):
    return ATTENDANCE_KNN_DIR if use_knn else ATTENDANCE_DIR


def attendance_csv_path(use_knn):
    return os.path.join(attendance_dir(use_knn), f'Attendance-{datetoday}.csv')


def open_camera():
    """Open webcam; use DirectShow on Windows to avoid MSMF warning/errors."""
    if sys.platform == 'win32':
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cv2.VideoCapture(0)


#### If these directories / CSVs don't exist, create them
for _dir in (ATTENDANCE_DIR, ATTENDANCE_KNN_DIR, FACES_DIR, FACES_KNN_DIR):
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

for _use_knn in (False, True):
    _csv = attendance_csv_path(_use_knn)
    _folder = attendance_dir(_use_knn)
    if f'Attendance-{datetoday}.csv' not in os.listdir(_folder):
        with open(_csv, 'w') as f:
            f.write('Name,Roll,Date,Check-In,Check-Out')

# Haar cascade for Add User (KNN: capture face crops; direct: optional hint on frame)
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
    """Load (encodings, names) from pickle or build from static/faces (non-KNN only)."""
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
    """Build known face encodings from static/faces/{name_id}/* (one full photo per user is enough)."""
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


def totalreg(use_knn=False):
    d = faces_dir(use_knn)
    if not os.path.isdir(d):
        return 0
    return len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])


def _parse_folder_display(folder):
    """Folder name is {name}_{id}; split on last underscore."""
    parts = folder.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return folder, '—'


def _list_user_rows(base_dir):
    """List of dicts: folder, display_name, user_id for template tables."""
    rows = []
    if not os.path.isdir(base_dir):
        return rows
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        display_name, user_id = _parse_folder_display(name)
        rows.append({'folder': name, 'display_name': display_name, 'user_id': user_id})
    return rows


def _safe_user_folder(base, folder):
    """Resolve user folder path or None if invalid / path traversal."""
    if not folder or '..' in folder or '/' in folder or '\\' in folder:
        return None
    base = os.path.normpath(os.path.abspath(base))
    path = os.path.normpath(os.path.abspath(os.path.join(base, folder)))
    if not path.startswith(base + os.sep):
        return None
    if not os.path.isdir(path):
        return None
    return path


def _retrain_knn_model():
    """Retrain or remove KNN file after user folder changes."""
    if totalreg(True) == 0:
        if os.path.isfile(KNN_MODEL_PATH):
            try:
                os.remove(KNN_MODEL_PATH)
            except OSError:
                pass
        return
    train_knn(FACES_KNN_DIR, KNN_MODEL_PATH)


def _rebuild_known_faces_pickle_full():
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


def _delete_user_folder_and_retrain(mode, folder):
    """Remove user folder; retrain KNN or rebuild known_faces.pkl. Returns True if deleted."""
    base = FACES_KNN_DIR if mode == 'knn' else FACES_DIR
    path = _safe_user_folder(base, folder)
    if not path:
        return False
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    if mode == 'knn':
        _retrain_knn_model()
    else:
        _rebuild_known_faces_pickle_full()
    return True


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


def extract_attendance(use_knn=False):
    csv_path = attendance_csv_path(use_knn)
    df = pd.read_csv(csv_path)
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


def _is_checked_in_today(name, use_knn=False):
    userid = name.split('_')[1]
    csv_path = attendance_csv_path(use_knn)
    if not os.path.isfile(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if 'Roll' not in df.columns:
        return False
    return int(userid) in list(df['Roll'])


def add_attendance(name, use_knn=False):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    csv_path = attendance_csv_path(use_knn)
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

def _session_use_knn():
    return session.get('use_knn', True)


@app.route('/')
async def home():
    use_knn = _session_use_knn()
    names, rolls, dates, check_ins, check_outs, l = extract_attendance(use_knn)
    return render_template(
        'home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
        totalreg=totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
    )


@app.route('/users')
async def users_list():
    mess = session.pop('flash_mess', None)
    use_knn = _session_use_knn()
    rows = _list_user_rows(FACES_KNN_DIR if use_knn else FACES_DIR)
    return render_template(
        'users.html', rows=rows, use_knn=use_knn,
        active_page='users', mess=mess,
    )


@app.route('/users/delete', methods=['POST'])
async def delete_user():
    mode = request.form.get('mode')
    folder = (request.form.get('folder') or '').strip()
    if mode not in ('knn', 'direct') or not folder:
        session['flash_mess'] = 'Invalid request.'
        return redirect(url_for('users_list'))
    if not _camera_lock.acquire(blocking=False):
        session['flash_mess'] = 'Another operation is in progress. Please try again.'
        return redirect(url_for('users_list'))
    try:
        ok = await asyncio.to_thread(_delete_user_folder_and_retrain, mode, folder)
        if ok:
            session['flash_mess'] = (
                'User deleted; KNN model retrained.' if mode == 'knn' else 'User deleted; known_faces.pkl rebuilt.'
            )
        else:
            session['flash_mess'] = 'Could not delete user (invalid folder or error).'
    finally:
        _camera_lock.release()
    return redirect(url_for('users_list'))


@app.route('/toggle_recognition_mode', methods=['POST'])
async def toggle_recognition_mode():
    session['use_knn'] = not session.get('use_knn', True)
    return redirect(request.referrer or url_for('home'))


def _attendance_camera(use_knn=True):
    """Open camera, wait for same face 5 sec, then record check-in or check-out once.
    use_knn: True = sklearn KNN model; False = direct compare vs mean encodings per user."""
    known_encodings, known_names = (None, None)
    if not use_knn:
        known_encodings, known_names = get_known_faces()
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
                    if use_knn:
                        identified_person = predict_with_knn(face_encodings[0], KNN_MODEL_PATH)
                    else:
                        identified_person = identify_face_from_encoding(
                            face_encodings[0], known_encodings, known_names, tolerance=0.6
                        )
                    if identified_person:
                        display_name = identified_person.split('_')[0]
                        action = 'Check-out' if _is_checked_in_today(identified_person, use_knn) else 'Check-in'
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
                            add_attendance(detected_person, use_knn)
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
    return extract_attendance(use_knn)


def _check_model_and_reg(use_knn=True):
    if totalreg(use_knn) == 0:
        names, rolls, dates, check_ins, check_outs, l = extract_attendance(use_knn)
        return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                    totalreg=totalreg(use_knn), datetoday2=datetoday2, mess='No users in database. Please add a new user first.'), None
    if use_knn:
        if not os.path.isfile(KNN_MODEL_PATH):
            if totalreg(use_knn) > 0:
                train_knn(FACES_KNN_DIR, KNN_MODEL_PATH)
            if not os.path.isfile(KNN_MODEL_PATH):
                names, rolls, dates, check_ins, check_outs, l = extract_attendance(use_knn)
                return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                            totalreg=totalreg(use_knn), datetoday2=datetoday2,
                            mess='No trained KNN model. Please add a new user first.'), None
    else:
        encs, names = get_known_faces()
        if not encs or not names:
            names, rolls, dates, check_ins, check_outs, l = extract_attendance(use_knn)
            return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                        totalreg=totalreg(use_knn), datetoday2=datetoday2,
                        mess='No face encodings for direct mode. Add a user or switch to KNN.'), None
    return None, None


@app.route('/start', methods=['GET'])
async def start():
    use_knn = _session_use_knn()
    if not _camera_lock.acquire(blocking=False):
        names, rolls, dates, check_ins, check_outs, l = extract_attendance(use_knn)
        return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                               totalreg=totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
                               mess='Camera is in use (e.g. Add User or another Take Attendance). Please wait and try again.')
    try:
        err, _ = _check_model_and_reg(use_knn=use_knn)
        if err is not None:
            return render_template('home.html', **err, use_knn=use_knn, active_page='home')
        names, rolls, dates, check_ins, check_outs, l = await asyncio.to_thread(_attendance_camera, use_knn)
        return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                               totalreg=totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home')
    finally:
        _camera_lock.release()


def _add_user_knn(newusername, newuserid):
    """Capture 50 face crops into faces_KNN, then train KNN."""
    userimagefolder = os.path.join(FACES_KNN_DIR, f'{newusername}_{newuserid}')
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
        cv2.imshow('Adding New User (KNN)', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.isdir(userimagefolder) and len(os.listdir(userimagefolder)) > 0:
        train_knn(FACES_KNN_DIR, KNN_MODEL_PATH)
        print(f"Saved KNN training images and retrained KNN for {newusername}_{newuserid}")


def _add_user_direct(newusername, newuserid):
    """Capture one full-frame photo into static/faces (non-KNN)."""
    userimagefolder = os.path.join(FACES_DIR, f'{newusername}_{newuserid}')
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = open_camera()
    saved = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for (x, y, w, h) in _face_boxes_haar(frame):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
        cv2.putText(frame, 'Press SPACE: save 1 full photo | ESC: cancel', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Adding New User (direct)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:  # SPACE
            img_path = os.path.join(userimagefolder, 'photo.jpg')
            cv2.imwrite(img_path, frame)
            saved = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved:
        if os.path.isfile(KNOWN_FACES_PATH):
            try:
                os.remove(KNOWN_FACES_PATH)
            except OSError:
                pass
        print(f"Saved full photo for direct mode: {newusername}_{newuserid}")


def _add_user(newusername, newuserid, use_knn=True):
    """Add user: KNN = 50 crops + train; direct = one full frame."""
    if use_knn:
        _add_user_knn(newusername, newuserid)
    else:
        _add_user_direct(newusername, newuserid)
    return extract_attendance(use_knn)


@app.route('/add', methods=['POST'])
async def add():
    use_knn = _session_use_knn()
    if not _camera_lock.acquire(blocking=False):
        names, rolls, dates, check_ins, check_outs, l = extract_attendance(use_knn)
        return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                               totalreg=totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
                               mess='Another operation is in progress. Please wait for Take Attendance or Add User to finish, then try again.')
    try:
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        names, rolls, dates, check_ins, check_outs, l = await asyncio.to_thread(_add_user, newusername, newuserid, use_knn)
        return render_template(
            'home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
            totalreg=totalreg(use_knn), datetoday2=datetoday2, use_knn=use_knn, active_page='home',
        )
    finally:
        _camera_lock.release()


if __name__ == '__main__':
    app.run(debug=True)

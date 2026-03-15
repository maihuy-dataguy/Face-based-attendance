import cv2
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from datetime import date, datetime

import face_recognition

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

KNOWN_FACES_PATH = 'static/known_faces.pkl'

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
def home():
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
    known_encodings, known_names = get_known_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = small_frame[:, :, ::-1]
            face_boxes = extract_faces_rgb(rgb_small, small_frame=rgb_small)
            last_box = None
            last_label = None

            if face_boxes:
                (x, y, w, h) = face_boxes[0]
                rgb_frame = frame[:, :, ::-1]
                top, right, bottom, left = y, x + w, y + h, x
                face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
                if face_encodings:
                    identified_person = identify_face_from_encoding(face_encodings[0], known_encodings, known_names)
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

                        if time.time() - face_detected_time >= 5:
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
                face_detected_time = None
                detected_person = None
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
    known_encodings, known_names = get_known_faces()
    if not known_encodings or not known_names:
        names, rolls, dates, check_ins, check_outs, l = extract_attendance()
        return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                    totalreg=totalreg(), datetoday2=datetoday2,
                    mess='No known faces. Please add a new user (with clear face photos).'), None
    return None, None


@app.route('/start', methods=['GET'])
def start():
    err, _ = _check_model_and_reg()
    if err is not None:
        return render_template('home.html', **err)
    names, rolls, dates, check_ins, check_outs, l = _attendance_camera()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    known_encodings, known_names = get_known_faces()
    cap = open_camera()
    photo_path = os.path.join(userimagefolder, f'{newusername}_{newuserid}.jpg')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, 'Press SPACE to take photo (ESC to cancel)', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Adding New User', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE - save full frame (one picture, no crop)
            cv2.imwrite(photo_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    user_encodings = []
    if os.path.isfile(photo_path):
        try:
            img = face_recognition.load_image_file(photo_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                user_encodings.append(encs[0])
        except Exception:
            pass

    if user_encodings:
        new_encoding = user_encodings[0]
        user_label = f'{newusername}_{newuserid}'
        known_encodings.append(new_encoding)
        known_names.append(user_label)
        with open(KNOWN_FACES_PATH, 'wb') as f:
            pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
        print(f"Saved encoding for {user_label}")
    else:
        print("Warning: No face encodings extracted from captured images.")

    names, rolls, dates, check_ins, check_outs, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)

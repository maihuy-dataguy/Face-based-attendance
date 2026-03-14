import cv2
import os
import sys
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


def open_camera():
    """Open webcam; use DirectShow on Windows to avoid MSMF warning/errors."""
    if sys.platform == 'win32':
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cv2.VideoCapture(0)


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = open_camera()


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Date,Check-In,Check-Out')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if face_detector.empty():
        print("Error: Haarcascade file not loaded properly.")
        return []
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points if len(face_points) > 0 else []



#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    # Date column (backfill from today if missing)
    if 'Date' in df.columns:
        dates = df['Date'].fillna(datetoday2).astype(str)
    else:
        dates = pd.Series([datetoday2] * len(df))
    # Support old format (Time) and new format (Check-In, Check-Out)
    if 'Check-In' in df.columns:
        check_ins = df['Check-In'].astype(str)
        check_outs = df['Check-Out'].fillna('').astype(str)
    else:
        check_ins = df['Time'].astype(str) if 'Time' in df.columns else pd.Series([''] * len(df))
        check_outs = pd.Series([''] * len(df))
    l = len(df)
    return names, rolls, dates, check_ins, check_outs, l


def _ensure_attendance_format(df):
    """Ensure CSV has Date, Check-In, Check-Out as strings."""
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
    """True if this person is already in today's attendance (so next scan = check-out)."""
    userid = name.split('_')[1]
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if 'Roll' not in df.columns:
        return False
    return int(userid) in list(df['Roll'])


#### Record attendance: first time today = check-in, next time = check-out (one button for both)
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


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names, rolls, dates, check_ins, check_outs, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l, totalreg=totalreg(), datetoday2=datetoday2) 


import time

def _attendance_camera():
    """Open camera, wait for same face 5 sec, then record check-in or check-out once. Returns attendance data."""
    cap = open_camera()
    face_detected_time = None
    detected_person = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) != 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            display_name = identified_person.split('_')[0]
            action = 'Check-out' if _is_checked_in_today(identified_person) else 'Check-in'
            text_y = max(y - 8, 24)
            cv2.putText(frame, f'{display_name}  |  {action}', (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 20), 2, cv2.LINE_AA)

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

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return extract_attendance()


def _check_model_and_reg():
    """Return (err_kwargs, None) if error else (None, None). err_kwargs are kwargs for render_template('home.html', **err)."""
    if totalreg() == 0:
        names, rolls, dates, check_ins, check_outs, l = extract_attendance()
        return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                    totalreg=totalreg(), datetoday2=datetoday2, mess='No users in database. Please add a new user first.'), None
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, dates, check_ins, check_outs, l = extract_attendance()
        return dict(names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                    totalreg=totalreg(), datetoday2=datetoday2,
                    mess='There is no trained model in the static folder. Please add a new face to continue.'), None
    return None, None


@app.route('/start', methods=['GET'])
def start():
    """One button: first scan today = check-in, later scan = check-out."""
    err, _ = _check_model_and_reg()
    if err is not None:
        return render_template('home.html', **err)
    names, rolls, dates, check_ins, check_outs, l = _attendance_camera()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)



#### This function will run when we add a new user
@app.route('/add', methods=['POST'])
def add():
    # Get user input from form
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    
    # Define folder to store images in "static/faces"
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)  # Create folder if it doesn't exist

    cap = open_camera()
    i, j = 0, 0

    while i < 50:  # Capture 50 images per user
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            if j % 5 == 0:  # Capture every 5th frame for variation
                img_name = f"{newusername}_{i}.jpg"
                img_path = os.path.join(userimagefolder, img_name)
                face_crop = frame[y:y + h, x:x + w]  # Crop only the face
                cv2.imwrite(img_path, face_crop)  # Save cropped face image
                i += 1

            j += 1

        cv2.imshow('Adding New User', frame)
        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ Images stored in {userimagefolder}")
    print("⚡ Training Model with New Data...")
    train_model()  # Train the model after capturing images

    # Fetch updated attendance list
    names, rolls, dates, check_ins, check_outs, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)
 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
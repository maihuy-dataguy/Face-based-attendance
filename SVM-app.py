import cv2
import os
import sys
import time
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.svm import SVC 
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
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
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
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


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
    svc = SVC(kernel='linear', probability=True)
    svc.fit(faces,labels)
    joblib.dump(svc,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
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
    """True if this person is already in today's attendance (so next scan = check-out)."""
    userid = name.split('_')[1]
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if 'Roll' not in df.columns:
        return False
    return int(userid) in list(df['Roll'])

def add_attendance(name):
    """First time today = check-in, next time = check-out."""
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


def _attendance_camera():
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
    err, _ = _check_model_and_reg()
    if err is not None:
        return render_template('home.html', **err)
    names, rolls, dates, check_ins, check_outs, l = _attendance_camera()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2) 


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = open_camera()
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, dates, check_ins, check_outs, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2) 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
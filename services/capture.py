"""OpenCV camera loops: take attendance, add user, model checks."""
import os
import time

import cv2
import face_recognition

from config import FACES_DIR, FACES_KNN_DIR, KNN_MODEL_PATH, KNOWN_FACES_PATH, datetoday2

from services import attendance as attendance_svc
from services import encodings
from services import knn
from services.camera import face_boxes_haar, open_camera


def attendance_camera(use_knn=True):
    """Wait for recognized face ~2s, then record check-in or check-out once."""
    known_encodings, known_names = (None, None)
    if not use_knn:
        known_encodings, known_names = encodings.get_known_faces()
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
            face_boxes = encodings.extract_faces_rgb(rgb_small, small_frame=rgb_small)
            if face_boxes:
                (x, y, w, h) = face_boxes[0]
                rgb_frame = frame[:, :, ::-1]
                top, right, bottom, left = y, x + w, y + h, x
                face_encs = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
                if face_encs:
                    if use_knn:
                        identified_person = knn.predict_with_knn(face_encs[0])
                    else:
                        identified_person = encodings.identify_face_from_encoding(
                            face_encs[0], known_encodings, known_names, tolerance=0.6
                        )
                    if identified_person:
                        display_name = identified_person.split('_')[0]
                        action = 'Check-out' if attendance_svc.is_checked_in_today(identified_person, use_knn) else 'Check-in'
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
                            attendance_svc.add_attendance(detected_person, use_knn)
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
    return attendance_svc.extract_attendance(use_knn)


def check_model_and_reg(use_knn=True):
    """Returns (error_dict, None) if cannot start attendance, else (None, None)."""
    from services import storage

    if storage.totalreg(use_knn) == 0:
        names, rolls, dates, check_ins, check_outs, l = attendance_svc.extract_attendance(use_knn)
        return dict(
            names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
            totalreg=storage.totalreg(use_knn), datetoday2=datetoday2,
            mess='No users in database. Please add a new user first.',
        ), None
    if use_knn:
        if not os.path.isfile(KNN_MODEL_PATH):
            if storage.totalreg(use_knn) > 0:
                knn.train_knn(FACES_KNN_DIR, KNN_MODEL_PATH)
            if not os.path.isfile(KNN_MODEL_PATH):
                names, rolls, dates, check_ins, check_outs, l = attendance_svc.extract_attendance(use_knn)
                return dict(
                    names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                    totalreg=storage.totalreg(use_knn), datetoday2=datetoday2,
                    mess='No trained KNN model. Please add a new user first.',
                ), None
    else:
        encs, names = encodings.get_known_faces()
        if not encs or not names:
            names, rolls, dates, check_ins, check_outs, l = attendance_svc.extract_attendance(use_knn)
            return dict(
                names=names, rolls=rolls, dates=dates, check_ins=check_ins, check_outs=check_outs, l=l,
                totalreg=storage.totalreg(use_knn), datetoday2=datetoday2,
                mess='No face encodings for direct mode. Add a user or switch to KNN.',
            ), None
    return None, None


def add_user_knn(newusername, newuserid):
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
        face_boxes = face_boxes_haar(frame)
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
        knn.train_knn(FACES_KNN_DIR, KNN_MODEL_PATH)
        print(f"Saved KNN training images and retrained KNN for {newusername}_{newuserid}")


def add_user_direct(newusername, newuserid):
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
        for (x, y, w, h) in face_boxes_haar(frame):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
        cv2.putText(frame, 'Press SPACE: save 1 full photo | ESC: cancel', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Adding New User (direct)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
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


def add_user(newusername, newuserid, use_knn=True):
    if use_knn:
        add_user_knn(newusername, newuserid)
    else:
        add_user_direct(newusername, newuserid)
    return attendance_svc.extract_attendance(use_knn)

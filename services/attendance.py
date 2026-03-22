"""Attendance: KNN → MySQL; direct mode → CSV."""
import os
from datetime import datetime

import pandas as pd

from config import attendance_csv_path, datetoday2

from services import attendance_mysql


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


def extract_attendance(use_knn=False):
    if use_knn:
        try:
            return attendance_mysql.extract_attendance_knn()
        except Exception:
            empty = pd.Series([], dtype=object)
            return empty, empty, empty, empty, empty, 0
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


def is_checked_in_today(name, use_knn=False):
    if use_knn:
        try:
            return attendance_mysql.is_checked_in_today_knn(name)
        except Exception:
            return False
    userid = name.rsplit('_', 1)[1]
    csv_path = attendance_csv_path(use_knn)
    if not os.path.isfile(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if 'Roll' not in df.columns:
        return False
    return int(userid) in list(df['Roll'])


def add_attendance(name, use_knn=False):
    if use_knn:
        attendance_mysql.add_attendance_knn(name)
        return
    username, userid = name.rsplit('_', 1)
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

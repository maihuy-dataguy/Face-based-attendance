"""KNN attendance stored in MySQL (users + attendance tables)."""
from datetime import date, datetime

import pandas as pd
from sqlalchemy import text

from config import datetoday2
from extensions import db
from models import Attendance, User


def _ping_db():
    """Return True if DB is reachable."""
    try:
        with db.engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        return True
    except Exception:
        return False


def register_user_if_needed(name, user_id_str):
    """Ensure a User row exists (e.g. after Add User KNN). Reactivates if previously soft-deleted."""
    u = User.query.filter_by(user_id=user_id_str).first()
    if not u:
        u = User(user_id=user_id_str, name=name)
        db.session.add(u)
    else:
        u.name = name
        u.deleted_at = None
    db.session.commit()
    return u


def extract_attendance_knn():
    """Today's attendance rows for home table (same shape as CSV path)."""
    today = date.today()
    rows = (
        db.session.query(Attendance, User)
        .join(User, Attendance.user_id == User.id)
        .filter(Attendance.date == today, User.deleted_at.is_(None))
        .order_by(Attendance.id)
        .all()
    )
    names, rolls, dates, check_ins, check_outs = [], [], [], [], []
    for att, user in rows:
        names.append(user.name)
        rolls.append(user.user_id)
        dates.append(datetoday2)
        check_ins.append(att.check_in_time.strftime('%H:%M:%S') if att.check_in_time else '')
        check_outs.append(att.check_out_time.strftime('%H:%M:%S') if att.check_out_time else '')
    l = len(names)
    return (
        pd.Series(names),
        pd.Series(rolls),
        pd.Series(dates),
        pd.Series(check_ins),
        pd.Series(check_outs),
        l,
    )


def is_checked_in_today_knn(folder_key):
    """folder_key like name_123 — user already has check-in today?"""
    try:
        userid_str = folder_key.rsplit('_', 1)[1]
    except (ValueError, IndexError):
        return False
    user = User.query.filter_by(user_id=userid_str).filter(User.deleted_at.is_(None)).first()
    if not user:
        return False
    att = Attendance.query.filter_by(user_id=user.id, date=date.today()).first()
    return att is not None and att.check_in_time is not None


def add_attendance_knn(folder_key):
    """Check-in or check-out from folder name name_userid."""
    username, userid_str = folder_key.rsplit('_', 1)
    user = User.query.filter_by(user_id=userid_str).filter(User.deleted_at.is_(None)).first()
    if not user:
        user = User(user_id=userid_str, name=username)
        db.session.add(user)
        db.session.flush()
    today = date.today()
    att = Attendance.query.filter_by(user_id=user.id, date=today).first()
    now = datetime.now().time()
    if not att:
        att = Attendance(user_id=user.id, date=today, check_in_time=now, check_out_time=None)
        db.session.add(att)
    else:
        att.check_out_time = now
    db.session.commit()

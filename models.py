"""SQLAlchemy models for KNN attendance (MySQL)."""
from extensions import db


class User(db.Model):
    """Registered person: natural key user_id (e.g. student id string), display name."""

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    deleted_at = db.Column(db.DateTime(timezone=True), nullable=True)

    attendances = db.relationship('Attendance', back_populates='user', lazy='dynamic')


class Attendance(db.Model):
    """One row per user per day (check-in / check-out). user_id column = FK to users.id."""

    __tablename__ = 'attendance'
    __table_args__ = (db.UniqueConstraint('user_id', 'date', name='uq_attendance_user_date'),)

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    check_in_time = db.Column(db.Time, nullable=True)
    check_out_time = db.Column(db.Time, nullable=True)

    user = db.relationship('User', back_populates='attendances')

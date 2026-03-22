"""
Face recognition attendance — Flask entry point.

Project layout:
  config.py          Paths, dates, MYSQL_URI, ensure_data_dirs()
  models.py          SQLAlchemy User + Attendance (KNN / MySQL)
  extensions.py      db, camera_lock
  services/          Business logic
  routes/views.py    Blueprint
"""
import os

from flask import Flask

from config import MYSQL_URI, SECRET_KEY, ensure_data_dirs
from extensions import db
from routes.views import register_routes

ensure_data_dirs()

app = Flask(__name__, template_folder='templates')
app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = MYSQL_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}

db.init_app(app)

# Import models so metadata is registered before create_all
import models  # noqa: F401, E402

with app.app_context():
    db.create_all()

register_routes(app)


if __name__ == '__main__':
    app.run(debug=True)

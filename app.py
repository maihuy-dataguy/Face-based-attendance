"""
Face recognition attendance — Flask entry point.

Project layout:
  config.py          Paths, dates, ensure_data_dirs()
  extensions.py      Shared locks
  services/          Business logic (camera, KNN, encodings, attendance CSV, users, capture)
  routes/views.py    Blueprint with HTTP handlers
"""
from flask import Flask

from config import SECRET_KEY, ensure_data_dirs
from routes.views import register_routes

ensure_data_dirs()

app = Flask(__name__, template_folder='templates')
app.secret_key = SECRET_KEY
register_routes(app)


if __name__ == '__main__':
    app.run(debug=True)

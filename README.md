# Facial-Recognition-Based-Attendance-System

Welcome to the GitHub repository for the project: a Facial Recognition-Based Attendance System. This project is built for presentation for Opentalk in NCC Plus.

## Project Overview

Our system leverages facial recognition technology to streamline the attendance process, making it more efficient and less time-consuming.
Implementing 2 aprroaches:

- Face_recognition library (just checking if two faces match or not (True or False), it's helpful to see how similar they are)
- Face_recognition library + KNN model (This example is useful when you wish to recognize a large set of known people,
  and make a prediction for an unknown person in a feasible computation time. Instead of checking similarity accross all faces in DB)

This ensures optimal performance across various devices, including both phones and PCs, making it accessible to schools with limited resources.

### Key Features

- **Low Computational Cost:** Designed with resource constraints in mind, ensuring it runs smoothly on minimal hardware.
- **Cross-Platform Compatibility:** Works seamlessly on both phones and PCs, providing flexibility in usage.
- **User-Friendly Interface:** Implemented using HTML, Bootstrap, and JavaScript for a smooth user experience.
- **Efficient Backend:** Powered by Flask, ensuring a robust and scalable application.

## Demo

![face-attendance-system](https://github.com/maihuy-dataguy/Face-based-attendance/blob/main/flask.jpg)

## Project layout

| Path                           | Role                                                        |
| ------------------------------ | ----------------------------------------------------------- |
| `app.py`                       | Flask app entry: `SECRET_KEY`, `register_routes()`          |
| `config.py`                    | Paths, `MYSQL_URI`, dates, `ensure_data_dirs()`             |
| `models.py`                    | SQLAlchemy `User` + `Attendance` (KNN attendance in MySQL)  |
| `extensions.py`                | `db` (SQLAlchemy), `camera_lock`                            |
| `routes/views.py`              | Blueprint `main`: HTTP routes only                          |
| `services/camera.py`           | Webcam + Haar boxes                                         |
| `services/knn.py`              | Train / predict KNN, `retrain_knn_model()`                  |
| `services/encodings.py`        | Known faces pickle, face locations, direct matching         |
| `services/attendance.py`       | Direct mode: CSV; KNN mode: delegates to `attendance_mysql` |
| `services/attendance_mysql.py` | MySQL attendance for KNN                                    |
| `sql/schema.sql`               | Optional manual MySQL DDL                                   |
| `services/storage.py`          | User folder listing, `totalreg()`, safe paths               |
| `services/users.py`            | Delete user folder + retrain / rebuild pickle               |
| `services/capture.py`          | Take attendance loop, add user, model checks                |
| `templates/`                   | Jinja HTML (`base.html`, `home.html`, `users.html`)         |

**KNN mode** stores attendance in **MySQL** (`users`, `attendance`). Create a database and set `MYSQL_URI` (see `.env.example`). Tables are also created automatically on startup via `db.create_all()`.

**Direct mode** still uses the `Attendance/` CSV files.

Run the app from the project root: `python app.py`.

## Getting Started

To get started, please follow the instructions below:

### Prerequisites

Ensure you have the following installed:

- Python 3.10
- Flask
- OpenCV
- A suitable web browser (Chrome/Firefox)

### Installation

1. Clone the repository to your local machine:

```bash
 git clone git@github.com:maihuy-dataguy/Face-based-attendance.git
```

2. Navigate to the cloned repository:

```bash
 cd Face-based-attendance
```

3. Install the required dependencies:

```
 pip install -r requirements.txt
```

5. Run the Flask application:

```
 flask run
```

## Future Work

We can integrate Face Anti-Spoofing (liveness detection) into our system to differentiate between fake and real people, preventing false facial verification by using a photo, video, mask or a different substitute for an authorized person’s face.

- Print attack: The attacker shows the picture of other person printed on a sheet of paper
- Replay attack: The attacker shows the screen of another device that plays a pre-recorded photo/video of the other person.

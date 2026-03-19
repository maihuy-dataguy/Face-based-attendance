# Facial-Recognition-Based-Attendance-System

Welcome to the GitHub repository for the project: a Facial Recognition-Based Attendance System designed for teachers and students. This project was developed in collaboration with a local non-profit organization aiming to bring sustainable, low computational cost solutions for automatic attendance systems to resource-limited schools in Pakistan.

## Project Overview

Our system leverages facial recognition technology to streamline the attendance process, making it more efficient and less time-consuming. By employing machine learning algorithms Haar cascade classifiers for face detection and k-Nearest Neighbors (k-NN) for face recognition, we've created a cost-effective and lightweight architecture. This ensures optimal performance across various devices, including both phones and PCs, making it accessible to schools with limited resources.

### Key Features

- **Low Computational Cost:** Designed with resource constraints in mind, ensuring it runs smoothly on minimal hardware.
- **Cross-Platform Compatibility:** Works seamlessly on both phones and PCs, providing flexibility in usage.
- **Sustainable Solution:** A collaboration with local non-profits to support educational institutions in Pakistan.
- **User-Friendly Interface:** Implemented using HTML, Bootstrap, and JavaScript for a smooth user experience.
- **Efficient Backend:** Powered by Flask, ensuring a robust and scalable application.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Flask app entry: `SECRET_KEY`, `register_routes()` |
| `config.py` | Paths (`static/faces`, `faces_KNN`, attendance dirs), dates, `ensure_data_dirs()` |
| `extensions.py` | Shared `camera_lock` (serialize camera + face_recognition work) |
| `routes/views.py` | Blueprint `main`: HTTP routes only |
| `services/camera.py` | Webcam + Haar boxes |
| `services/knn.py` | Train / predict KNN, `retrain_knn_model()` |
| `services/encodings.py` | Known faces pickle, face locations, direct matching |
| `services/attendance.py` | Today’s CSV read/write |
| `services/storage.py` | User folder listing, `totalreg()`, safe paths |
| `services/users.py` | Delete user folder + retrain / rebuild pickle |
| `services/capture.py` | Take attendance loop, add user, model checks |
| `templates/` | Jinja HTML (`base.html`, `home.html`, `users.html`) |

Run the app from the project root: `python app.py`.

## Getting Started

To get started, please follow the instructions below:

### Prerequisites

Ensure you have the following installed:

- Python 3.12
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

The current iteration of our system leverages traditional Machine Learning algorithms for facial detection and identification tasks. Moving forward, we aim to explore and integrate Deep Learning techniques to enhance the system's accuracy and efficiency.

-- MySQL schema for KNN attendance (Flask-SQLAlchemy create_all() also creates these.)
-- Usage: CREATE DATABASE face_attendance CHARACTER SET utf8mb4;
--        mysql -u root -p face_attendance < sql/schema.sql

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    deleted_at DATETIME(6) NULL,
    UNIQUE KEY uq_users_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    date DATE NOT NULL,
    check_in_time TIME NULL,
    check_out_time TIME NULL,
    CONSTRAINT fk_attendance_user FOREIGN KEY (user_id) REFERENCES users (id),
    UNIQUE KEY uq_attendance_user_date (user_id, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

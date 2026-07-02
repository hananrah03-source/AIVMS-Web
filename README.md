# AIVMS-Website
AI-Based Patient Vital Monitoring System
# 🩺 AI-Based Patient Vital Monitoring System (AIVMS)

An intelligent healthcare monitoring platform that combines **Internet of Things (IoT)**, **Artificial Intelligence (AI)**, and **real-time data visualization** to remotely monitor patients' vital signs and assist healthcare professionals in early decision-making.

---

## 📖 Overview

The AI-Based Patient Vital Monitoring System (AIVMS) is an innovative healthcare solution designed to continuously monitor patients' vital signs in real time. The system collects physiological data using IoT sensors connected to an ESP32 microcontroller, transmits the data to a FastAPI backend, analyzes the patient's condition using an XGBoost machine learning model, and displays the results through an interactive web dashboard and a Flutter mobile application.

The objective of this project is to improve patient monitoring by enabling early detection of abnormal health conditions, reducing response time, and supporting healthcare professionals with AI-assisted decision-making.

---

## ✨ Features

- 📡 Real-time monitoring of patient vital signs
- ❤️ Heart Rate (BPM) monitoring
- 🩸 Blood Oxygen Saturation (SpO₂) monitoring
- 🌡️ Body Temperature monitoring
- 🤖 AI-based health risk prediction using XGBoost
- 📊 Interactive web dashboard
- 📱 Flutter mobile application
- 🔔 Real-time alerts for abnormal readings
- 📈 Historical data visualization
- ☁️ REST API communication using FastAPI
- 🔐 Secure user authentication

---

## 🏗️ System Architecture

ESP32 + Sensors

⬇

FastAPI Backend

⬇

SQLite Database

⬇

XGBoost Prediction Model

⬇

Web Dashboard & Flutter Mobile App

---

## 🛠️ Technologies Used

### Programming Languages
- Python
- Dart
- HTML
- CSS
- JavaScript
- C++ (ESP32)

### Frameworks & Libraries
- FastAPI
- Flutter
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- Uvicorn

### Hardware
- ESP32-S3
- MAX30102 Pulse Oximeter Sensor
- MLX90614 Infrared Temperature Sensor
- OLED Display
- Push Buttons

### Development Tools
- VS Code
- PlatformIO
- Git
- GitHub

---

## 📂 Project Structure

```
AIVMS-Web/
│
├── backend/
├── templates/
├── static/
├── models/
├── database/
├── images/
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/hananrah03-source/AIVMS-Web.git
```

### Navigate to the project

```bash
cd AIVMS-Web
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
uvicorn main:app --reload
```

The application will be available at:

```
http://127.0.0.1:8000
```

---

## 📱 Mobile Application

A dedicated Flutter mobile application allows users to:

- View patient vital signs
- Receive notifications
- Monitor health status remotely
- Access historical records

---

## 🤖 Artificial Intelligence

The system integrates an **Extreme Gradient Boosting (XGBoost)** model trained on physiological data to classify patient health conditions.

The AI model analyzes:

- Heart Rate
- SpO₂
- Body Temperature

to provide intelligent predictions that support early intervention and improve monitoring efficiency.

---

## 📸 Screenshots

> *(Add screenshots here)*

- Login Page
- Dashboard
- Real-Time Monitoring
- AI Prediction
- Mobile Application
- ESP32 Prototype

---

## 🎯 Future Improvements

- Cloud deployment
- Multi-patient monitoring
- Doctor dashboard
- SMS and Email alerts
- ECG integration
- Wearable device compatibility
- Electronic Health Record (EHR) integration

---

## 👨‍💻 Author

**Hanan Rahah**

Master's Degree in Automation and Industrial Computer Science

Amar Telidji University – Laghouat, Algeria

📧 Email: hananrah03@gmail.com

🌐 GitHub: https://github.com/hananrah03-source

---

## 📜 License

This project is intended for academic research and innovation purposes.

© 2026 Hanan Rahmoun. All rights reserved.
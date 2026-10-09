# ⚡ Smart Energy Demand Forecasting & Anomaly Detection Platform

An end-to-end Data Science and Machine Learning project for electricity demand analysis, forecasting, and anomaly detection using Python, PostgreSQL, SQL, Scikit-learn, and Streamlit.

## 📌 Project Overview

This project analyzes electricity consumption data from multiple clients and provides:

- Electricity demand analysis
- Time-based demand patterns
- Peak demand identification
- Machine Learning based demand forecasting
- Anomaly detection
- PostgreSQL-based analytics
- Interactive Streamlit dashboard

The project follows a complete Data Science workflow:

Raw Data → Data Processing → EDA → Feature Engineering → ML Model → Anomaly Detection → SQL Analytics → Dashboard

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze electricity consumption patterns.
2. Identify peak and low-demand periods.
3. Build a Machine Learning model for demand forecasting.
4. Detect unusual electricity demand behavior.
5. Store and analyze energy data using PostgreSQL.
6. Build an interactive dashboard for users.

---

## 📊 Dataset

Dataset used:

**UCI ElectricityLoadDiagrams20112014**

The dataset contains electricity consumption measurements from multiple clients at 15-minute intervals.

### Dataset characteristics

- 370 electricity clients
- 15-minute intervals
- Electricity consumption data
- Time period: 2011–2014
- Raw dataset size: approximately 678 MB

The raw dataset is not included in this GitHub repository because of its large size.

---

## 🛠️ Technologies Used

### Programming
- Python

### Data Science
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Random Forest Regressor
- Joblib

### Database
- PostgreSQL
- SQL
- SQLAlchemy
- Psycopg2

### Dashboard
- Streamlit

### Environment Management
- Python-dotenv

---

## 🏗️ Project Structure

```text
Smart-Energy-Analytics/
│
├── data/
│   ├── raw/
│   │   └── LD2011_2014.txt
│   │
│   └── processed/
│       ├── energy_demand.csv
│       ├── ml_ready_energy.csv
│       └── anomaly_results.csv
│
├── notebooks/
│   └── 01_data_understanding.py
│
├── src/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── feature_importance.py
│   ├── baseline_comparison.py
│   ├── anomaly_detection.py
│   └── database.py
│
├── models/
│   └── demand_model.pkl
│
├── dashboard/
│   └── app.py
│
├── process_data.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
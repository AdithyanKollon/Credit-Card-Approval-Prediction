# 💳 Credit Card Approval Prediction App

An end-to-end machine learning project to predict credit card approval using **LightGBM** and **Logistic GAM** models, with an interactive **Streamlit** frontend.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Web App](#web-app)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots & Results](#screenshots--results)
- [Technologies Used](#technologies-used)

---

## 📝 Overview
This project predicts credit card approval for applicants based on demographic and financial details.  
It consists of:
- A Jupyter Notebook for data preprocessing, feature engineering, and model training.
- Saved models (`.pkl`) for deployment.
- A Streamlit web application for real-time prediction.

---

## ✨ Features
- Two-model pipeline: **LightGBM** (gradient boosting) + **Logistic GAM** (Generalized Additive Model).
- Preprocessing pipeline to handle categorical and numeric inputs.
- Probability-based approval prediction with validation logic (age limits, etc.).
- Interactive UI built in Streamlit for user-friendly input and instant results.

---

## 📂 Project Structure

credit-card-approval/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│ └── model_training.ipynb # Training and evaluation
│
├── models/
│ ├── lgb_model.pkl
│ ├── gam_model.pkl
│ └── X_columns.pkl
│
└── app/
└── app.py # Streamlit frontend app


---

## 🧠 Model Training
The model training notebook (`notebooks/model_training.ipynb`) covers:
- Data cleaning and feature engineering.
- Model building using LightGBM and Logistic GAM.
- Saving trained models with `joblib` for deployment.

<!-- Add screenshots of training graphs, metrics, and confusion matrices here -->
![Model Training Placeholder](path/to/your_screenshot.png)

---

## 🌐 Web App
The Streamlit app (`app/app.py`) provides:
- Interactive form for applicant data entry.
- Real-time prediction of approval/rejection with probability.
- Applicant detail summary table.

<!-- Add screenshots or gifs of your frontend here -->
![Frontend Placeholder](path/to/your_frontend_screenshot.png)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-approval.git
cd credit-card-approval

Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app/app.py

🖥️ Usage

Enter applicant information (age, income, credit score, etc.) in the Streamlit form.

Click Predict to see the approval result and probability.

View applicant summary below the prediction.

📊 Screenshots & Results
<!-- Insert your evaluation metrics here (accuracy, precision, recall, ROC curve, etc.) -->

<!-- Insert more screenshots if needed -->

🛠️ Technologies Used

Python (Pandas, NumPy, scikit-learn)

LightGBM

Logistic GAM

Joblib

Streamlit

Jupyter Notebook
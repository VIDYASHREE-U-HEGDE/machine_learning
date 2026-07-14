# 🚀 Customer Churn Prediction API

A Machine Learning-based REST API that predicts whether a customer is likely to churn. The application is built using **FastAPI**, containerized using **Docker**, and deployed on **Render**.

---

# 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their services. This project uses a **Random Forest Classifier** to predict customer churn based on customer information.

---

# ✨ Features

- Customer Churn Prediction using Machine Learning
- REST API developed with FastAPI
- Interactive Swagger API Documentation
- Dockerized Application
- Cloud Deployment using Render

---

# 🛠️ Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Docker
- Git
- GitHub
- Render

---

# 📁 Project Structure

```
Customer_churn_prediction/
│
├── app.py
├── model.py
├── churn_model.joblib
├── dataset.csv
├── Dockerfile
├── requirements.txt
├── README.md
```

---

# 🌐 Live Demo

Swagger Documentation

https://your-render-url.onrender.com/docs

---

# 📡 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Check whether the API is running |
| POST | `/predict` | Predict customer churn |

---

# 📥 Sample Request

```json
{
  "Age": 35,
  "Gender": "Male",
  "Tenure": 12,
  "Usage_Frequency": 15,
  "Support_Calls": 2,
  "Payment_Delay": 1,
  "Subscription_Type": "Premium",
  "Contract_Length": "Annual",
  "Total_Spend": 500.5,
  "Last_Interaction": 7
}
```

---

# 📤 Sample Response

```json
{
  "prediction": 1
}
```

---

# 🚀 How to Run Locally

### Clone the Repository

```bash
git clone https://github.com/username/Customer_churn_prediction.git
```

### Navigate to the Project

```bash
cd Customer_churn_prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# 📊 Machine Learning Model

- Algorithm: Random Forest Classifier
- Target: Customer Churn Prediction

---


GitHub:
https://github.com/VIDYASHREE-U-HEGDE/machine_learning.git

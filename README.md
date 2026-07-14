# 🚀 Customer Churn Prediction API

A Machine Learning-based REST API that predicts whether a customer is likely to churn. The application is built using **FastAPI**, containerized using **Docker**.

---

# 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their services. This project uses a **Random Forest Classifier** to predict customer churn based on customer information.

---

# ✨ Features

- Customer Churn Prediction using Machine Learning
- REST API developed with FastAPI
- Interactive Swagger API Documentation
- Dockerized Application

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

---

# 📁 Project Structure

```text
Customer_churn_prediction/
│
├── api/
│   └── app.py
├── src/
│   └── train.py
├── models/
│   └── churn_model.pkl
├── data/
│   └── customer_churn.csv
├── Dockerfile
├── requirements.txt
├── README.md
```

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
  "Gender": "Female",
  "Age": 35,
  "Tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 850.0,
  "ContractType": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check"
}
```

---

# 📤 Sample Response

```json
{
  "prediction": "Customer will stay"
}
```

or

```json
{
  "prediction": "Customer will churn"
}
```

---

# 🚀 How to Run Locally

### Clone the Repository

```bash
git clone https://github.com/VIDYASHREE-U-HEGDE/machine_learning.git
```

### Navigate to the Project

```bash
cd machine_learning
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

If `app.py` is inside the **api** folder:

```bash
uvicorn api.app:app --reload
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8000/docs
```

---

# 📊 Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Target:** Customer Churn Prediction

---

# 🔗 GitHub Repository

https://github.com/VIDYASHREE-U-HEGDE/machine_learning

---

# 👩‍💻 Author

**Vidyashree U Hegde**

GitHub: https://github.com/VIDYASHREE-U-HEGDE

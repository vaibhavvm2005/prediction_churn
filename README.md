# customer-churn-prediction-with-mlflow
# customer-churn-prediction-api
# 🚀 Customer Churn Prediction with FastAPI, MLflow, AWS S3 & Docker

A Machine Learning project that predicts whether a customer is likely to churn using an MLflow model deployed through FastAPI. The trained model is stored in AWS S3 and automatically downloaded when the application starts. The project is containerized using Docker and can be deployed on cloud platforms such as Render.

---

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. This project uses a trained Machine Learning model to predict customer churn based on customer information.

The API is built using **FastAPI**, the model is managed using **MLflow**, stored in **AWS S3**, and deployed using **Docker**.

---

## ✨ Features

- Customer Churn Prediction API
- FastAPI REST API
- Automatic MLflow Model Download from AWS S3
- Docker Support
- Swagger API Documentation
- Cloud Deployment Ready
- Input Validation using Pydantic
- JSON Prediction Response

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| FastAPI | REST API |
| MLflow | Model Management |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| AWS S3 | Model Storage |
| Boto3 | AWS SDK |
| Docker | Containerization |
| Render | Cloud Deployment |

---

# 📂 Project Structure

```
prediction_churn/
│
├── app.py
├── config.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── registermodel/
│   └── XGBoost/
│       ├── MLmodel
│       ├── python_model.pkl
│       ├── conda.yaml
│       └── ...
│
└── .gitignore
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/vaibhavvm2005/prediction_churn.git

cd prediction_churn
```

---

## Create Virtual Environment

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a **config.py**

```python
AWS_ACCESS_KEY="YOUR_ACCESS_KEY"

AWS_SECRET_KEY="YOUR_SECRET_KEY"

AWS_REGION="YOUR_REGION"

BUCKET_NAME="YOUR_BUCKET_NAME"
```

---

# ▶️ Run Application

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

# 📡 API Endpoints

## Home

### GET /

Response

```json
{
  "message":"Customer Churn Prediction API Running Successfully"
}
```

---

## Predict Customer Churn

### POST /predict

Example Request

```json
{
  "Age":35,
  "Gender":1,
  "Tenure":24,
  "Usage_Frequency":18,
  "Support_Calls":2,
  "Payment_Delay":5,
  "Subscription_Type":1,
  "Contract_Length":2,
  "Total_Spend":1200.50,
  "Last_Interaction":8
}
```

Example Response

```json
{
  "prediction":1,
  "churn":"Yes"
}
```

or

```json
{
  "prediction":0,
  "churn":"No"
}
```

---

# ☁️ AWS S3 Integration

The application automatically downloads the MLflow model from AWS S3 during startup.

Example S3 Structure

```
customer_churn/

└── registermodel/

      └── XGBoost/

            ├── MLmodel

            ├── python_model.pkl

            ├── conda.yaml

            └── ...
```

---

# 🐳 Docker

Build Docker Image

```bash
docker build -t customer-churn .
```

Run Docker Container

```bash
docker run -p 8000:8000 customer-churn
```

---

# 🚀 Deployment

This project can be deployed on:

- Render
- Railway
- AWS EC2
- Azure App Service
- Google Cloud Run

---

# 📊 Machine Learning Model

Algorithm Used

- XGBoost Classifier

Model Management

- MLflow

Storage

- AWS S3

---

# 📸 API Documentation

FastAPI automatically generates interactive API documentation.

Swagger UI

```
/docs
```

ReDoc

```
/redoc
```

---

# 📦 Requirements

- Python 3.9+
- FastAPI
- MLflow
- Pandas
- Scikit-learn
- Boto3
- Docker

---

# 👨‍💻 Author

**Vaibhav Madival**

🎓 CSE (Data Science) Student

GitHub

https://github.com/vaibhavvm2005

LinkedIn

(Add your LinkedIn profile here)

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is licensed under the MIT License.

import os
import boto3
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    BUCKET_NAME,
)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# =====================================================
# AWS S3 Client
# =====================================================

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# =====================================================
# Local Model Directory
# =====================================================

LOCAL_MODEL_DIR = "registermodel/XGBoost"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# =====================================================
# Download MLflow Model Folder
# =====================================================

def download_folder(bucket, prefix, local_dir):
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):

        if "Contents" not in page:
            continue

        for obj in page["Contents"]:

            key = obj["Key"]

            if key.endswith("/"):
                continue

            relative_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, relative_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            print(f"Downloading {key}")

            s3.download_file(
                bucket,
                key,
                local_path
            )

# =====================================================
# Download model if not exists
# =====================================================

mlmodel_file = os.path.join(LOCAL_MODEL_DIR, "MLmodel")

if not os.path.exists(mlmodel_file):

    print("Downloading MLflow Model From S3...")

    download_folder(
        BUCKET_NAME,
        "customer_churn/registermodel/XGBoost",
        LOCAL_MODEL_DIR
    )

    print("Download Completed.")

# =====================================================
# Show downloaded files
# =====================================================

print("\nDownloaded Files:\n")

for root, dirs, files in os.walk(LOCAL_MODEL_DIR):
    for file in files:
        print(os.path.join(root, file))

# =====================================================
# Load MLflow Model
# =====================================================

MODEL_PATH = LOCAL_MODEL_DIR

print(f"\nLoading model from: {MODEL_PATH}\n")

model = mlflow.pyfunc.load_model(MODEL_PATH)

print("Model Loaded Successfully.")

# =====================================================
# Request Schema
# =====================================================

class CustomerData(BaseModel):
    Age: int
    Gender: int
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: int
    Contract_Length: int
    Total_Spend: float
    Last_Interaction: int

# =====================================================
# Home API
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API Running Successfully"
    }

# =====================================================
# Prediction API
# =====================================================

@app.post("/predict")
def predict(data: CustomerData):

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0]),
        "churn": "Yes" if int(prediction[0]) == 1 else "No"
    }
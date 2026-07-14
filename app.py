import os
import traceback
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
    region_name=AWS_REGION,
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

            s3.download_file(bucket, key, local_path)

# =====================================================
# Download model if not exists
# =====================================================

mlmodel_file = os.path.join(LOCAL_MODEL_DIR, "MLmodel")

if not os.path.exists(mlmodel_file):

    print("Downloading MLflow Model From S3...")

    download_folder(
        BUCKET_NAME,
        "customer_churn/registermodel/XGBoost",
        LOCAL_MODEL_DIR,
    )

    print("Download Completed.")

# =====================================================
# Load Model
# =====================================================

print("\nLoading MLflow Model...")

model = mlflow.pyfunc.load_model(LOCAL_MODEL_DIR)

print("Model Loaded Successfully.")

try:
    print("Model Signature:")
    print(model.metadata.signature)
except Exception:
    pass

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
# Home
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API Running Successfully"
    }

# =====================================================
# Predict
# =====================================================

@app.post("/predict")
def predict(data: CustomerData):

    try:

        df = pd.DataFrame([{
            "Age": float(data.Age),
            "Gender": int(data.Gender),
            "Tenure": float(data.Tenure),
            "Usage Frequency": float(data.Usage_Frequency),
            "Support Calls": float(data.Support_Calls),
            "Payment Delay": float(data.Payment_Delay),
            "Subscription Type": int(data.Subscription_Type),
            "Contract Length": int(data.Contract_Length),
            "Total Spend": float(data.Total_Spend),
            "Last Interaction": float(data.Last_Interaction)
        }])

        print("\n========== INPUT ==========")
        print(df)
        print(df.dtypes)

        prediction = model.predict(df)

        pred = int(prediction[0])

        return {
            "success": True,
            "prediction": pred,
            "churn": "Yes" if pred == 1 else "No"
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "success": False,
            "error": str(e)
        }
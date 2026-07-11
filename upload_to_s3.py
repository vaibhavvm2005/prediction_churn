import os
import boto3
from config import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    BUCKET_NAME
)

# ==========================================
# Create S3 Client
# ==========================================

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# ==========================================
# Function to Upload Folder
# ==========================================

def upload_folder(local_folder, s3_folder):

    if not os.path.exists(local_folder):
        print(f"\n Folder '{local_folder}' not found.")
        return

    print(f"\nUploading folder: {local_folder}")

    for root, dirs, files in os.walk(local_folder):

        for file in files:

            local_path = os.path.join(root, file)

            relative_path = os.path.relpath(
                local_path,
                local_folder
            )

            s3_path = os.path.join(
                s3_folder,
                relative_path
            ).replace("\\", "/")

            print(f"Uploading {local_path}")

            s3.upload_file(
                local_path,
                BUCKET_NAME,
                s3_path
            )

            print(f"Uploaded → s3://{BUCKET_NAME}/{s3_path}")

# ==========================================
# Main
# ==========================================

print("=" * 60)
print("CUSTOMER CHURN MODEL UPLOAD TO AWS S3")
print("=" * 60)

print("Bucket Name :", BUCKET_NAME)
print("AWS Region  :", AWS_REGION)

# Upload models folder
upload_folder(
    local_folder="models",
    s3_folder="customer_churn/models"
)

# Upload registermodel folder
upload_folder(
    local_folder="registermodel",
    s3_folder="customer_churn/registermodel"
)

print("\n" + "=" * 60)
print("UPLOAD COMPLETED SUCCESSFULLY")
print("=" * 60)
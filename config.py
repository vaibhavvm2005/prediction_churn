# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

print(AWS_ACCESS_KEY, end="\n")
print(AWS_SECRET_KEY, end="\n")
print(AWS_REGION, end="\n")
print(BUCKET_NAME, end="\n")
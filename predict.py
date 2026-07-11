import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/model.pkl")

# Example customer data
sample = {
    "Age": 30,
    "Gender": 0,            # 0: Female, 1: Male
    "Tenure": 39,
    "Usage Frequency": 14,
    "Support Calls": 5,
    "Payment Delay": 18,
    "Subscription Type": 2, # 0: Basic, 1: Premium, 2: Standard
    "Contract Length": 0,    # 0: Annual, 1: Monthly, 2: Quarterly
    "Total Spend": 932,
    "Last Interaction": 17
}

df = pd.DataFrame([sample])

prediction = model.predict(df)

if prediction[0] == 1:
    print("Customer will Churn")
else:
    print("Customer will NOT Churn")
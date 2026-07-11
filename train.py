import os
import joblib
import pandas as pd
import kagglehub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================
# MLflow
# ==========================================

mlflow.set_experiment("Customer Churn Prediction")

# ==========================================
# Download Dataset
# ==========================================

path = kagglehub.dataset_download(
    "muhammadshahidazeem/customer-churn-dataset"
)

filename = os.path.join(
    path,
    "customer_churn_dataset-training-master.csv"
)

df = pd.read_csv(filename)

print("Dataset Shape:", df.shape)

# ==========================================
# Preprocessing
# ==========================================

df.dropna(inplace=True)

for col in ["customerID", "CustomerID"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {

    "Logistic Regression": LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

print("="*60)
print("TRAINING MODELS")
print("="*60)


# ==========================================
# Train Models
# ==========================================

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train Model
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Print Results
        print("\n" + "=" * 50)
        print(f"Model : {name}")
        print("=" * 50)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")

        # Store Results
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

        # Log Parameters
        mlflow.log_params(model.get_params())

        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.iloc[:5]
        )

        # Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name


# ==========================================
# Save Files
# ==========================================

os.makedirs("models", exist_ok=True)

# Save Best Model
joblib.dump(best_model, "models/model.pkl")

# Save Label Encoders
joblib.dump(encoders, "models/encoders.pkl")

# Save Model Comparison
comparison_df = pd.DataFrame(results)
comparison_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

# Log Best Model Separately
with mlflow.start_run(run_name="Best_Model"):

    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_accuracy", best_accuracy)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        input_example=X_test.iloc[:5]
    )

# ==========================================
# Final Output
# ==========================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Best Model      : {best_model_name}")
print(f"Best Accuracy   : {best_accuracy:.4f}")
print("Saved Model     : models/model.pkl")
print("Saved Encoders  : models/encoders.pkl")
print("Comparison File : models/model_comparison.csv")
print("=" * 60)
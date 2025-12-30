import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import joblib

DATA_PATH = "data/processed/insurance_processed.csv"
MODEL_DIR = "models"

def train_model():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop("claim_approved", axis=1)
    y = df["claim_approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # MLflow experiment
    mlflow.set_experiment("AllianzGI_Claim_Approval")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # Log params
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Save model locally
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "claim_approval_model.pkl")
        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        print("Training completed")
        print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()

import pandas as pd
import mlflow
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

DATA_PATH = "data/processed/insurance_processed.csv"
MODEL_PATH = "models/claim_approval_model.pkl"
DECISION_PATH = "models/model_decision.txt"

ACC_THRESHOLD = 0.85
RECALL_THRESHOLD = 0.80

def evaluate_model():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop("claim_approved", axis=1)
    y = df["claim_approved"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load model
    model = joblib.load(MODEL_PATH)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    # Promotion logic
    decision = "PROMOTED" if acc >= ACC_THRESHOLD and recall >= RECALL_THRESHOLD else "REJECTED"

    os.makedirs("models", exist_ok=True)
    with open(DECISION_PATH, "w") as f:
        f.write(decision)

    print("Model decision:", decision)

    # Log to MLflow
    with mlflow.start_run(run_name="Model_Evaluation"):
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_recall", recall)
        mlflow.log_metric("eval_precision", precision)
        mlflow.log_param("decision", decision)

if __name__ == "__main__":
    evaluate_model()

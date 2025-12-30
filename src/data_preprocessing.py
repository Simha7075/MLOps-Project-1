import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = "data/raw/insurance_claims.csv"
PROCESSED_DATA_PATH = "data/processed/insurance_processed.csv"

def preprocess_data():
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Basic validation
    if df.empty:
        raise ValueError("Raw dataset is empty")

    # Encode categorical columns
    categorical_cols = ["policy_type", "claim_type"]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Separate target
    if "claim_approved" not in df.columns:
        raise ValueError("Target column not found")

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Data preprocessing completed")
    print("Processed data saved at:", PROCESSED_DATA_PATH)
    print("Shape:", df.shape)

if __name__ == "__main__":
    preprocess_data()

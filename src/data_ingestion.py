import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = "data/raw/insurance_claims.csv"

def generate_insurance_data(rows=2000):
    np.random.seed(42)

    data = {
        "age": np.random.randint(18, 75, rows),
        "policy_tenure": np.random.randint(1, 120, rows),  # months
        "premium_amount": np.random.randint(5000, 50000, rows),
        "claim_amount": np.random.randint(1000, 300000, rows),
        "previous_claims": np.random.randint(0, 6, rows),
        "policy_type": np.random.choice(["Basic", "Silver", "Gold"], rows),
        "claim_type": np.random.choice(["Accident", "Theft", "Health"], rows),
        "fraud_score": np.round(np.random.uniform(0, 1, rows), 2)
    }

    df = pd.DataFrame(data)

    # Business rule to generate target
    df["claim_approved"] = np.where(
        (df["fraud_score"] < 0.6) & (df["claim_amount"] < 200000),
        1,
        0
    )

    return df

def main():
    os.makedirs("data/raw", exist_ok=True)
    df = generate_insurance_data()
    df.to_csv(RAW_DATA_PATH, index=False)

    print("Insurance claims dataset generated successfully")
    print("Path:", RAW_DATA_PATH)
    print("Shape:", df.shape)

if __name__ == "__main__":
    main()

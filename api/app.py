from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np

MODEL_PATH = "models/claim_approval_model.pkl"
DECISION_PATH = "models/model_decision.txt"

app = FastAPI(title="AllianzGI Claim Approval API")

# Input schema
class ClaimInput(BaseModel):
    age: int
    policy_tenure: int
    premium_amount: float
    claim_amount: float
    previous_claims: int
    policy_type: int
    claim_type: int
    fraud_score: float

# Load model only if PROMOTED
if not os.path.exists(DECISION_PATH):
    raise RuntimeError("Model decision file missing")

with open(DECISION_PATH, "r") as f:
    decision = f.read().strip()

if decision != "PROMOTED":
    raise RuntimeError("Model not approved for deployment")

model = joblib.load(MODEL_PATH)

@app.get("/")
def health_check():
    return {"status": "API is running", "model_status": decision}

@app.post("/predict")
def predict_claim(data: ClaimInput):
    try:
        features = np.array([[  
            data.age,
            data.policy_tenure,
            data.premium_amount,
            data.claim_amount,
            data.previous_claims,
            data.policy_type,
            data.claim_type,
            data.fraud_score
        ]])

        prediction = model.predict(features)[0]

        return {
            "claim_approved": int(prediction),
            "message": "Approved" if prediction == 1 else "Flagged for review"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

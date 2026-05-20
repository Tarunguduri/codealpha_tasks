import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(title="Credit Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

@app.on_event("startup")
async def load_model():
    global pipeline
    try:
        model_path = os.path.join(PROJECT_ROOT, "outputs", "best_model.pkl")
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            print(f"[Server] Successfully loaded pipeline from {model_path}")
        else:
            print(f"[Server] WARNING: Model file not found at {model_path}")
    except Exception as e:
        print(f"[Server] Error loading model: {e}")

class CreditRequest(BaseModel):
    age: float
    income: float
    debt: float
    loan_amount: float
    credit_history_years: float
    num_late_payments: float
    num_credit_lines: float
    employment_years: float

@app.post("/predict")
async def predict(request: CreditRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    try:
        # Calculate engineered feature
        debt_to_income = request.debt / request.income if request.income > 0 else 0
        
        # Create DataFrame (must match the feature order in training)
        data = pd.DataFrame([{
            "age": request.age,
            "income": request.income,
            "debt": request.debt,
            "loan_amount": request.loan_amount,
            "credit_history_years": request.credit_history_years,
            "num_late_payments": request.num_late_payments,
            "num_credit_lines": request.num_credit_lines,
            "employment_years": request.employment_years,
            "debt_to_income_ratio": debt_to_income
        }])
        
        # Predict probability of default
        prob_default = pipeline.predict_proba(data)[0][1]
        
        # Predict class (1 = Default, 0 = Good)
        prediction = int(pipeline.predict(data)[0])
        
        # Invert probability for a "Credit Score" (higher is better)
        credit_score = int((1.0 - prob_default) * 850)
        
        # Clamp between 300 and 850
        credit_score = max(300, min(850, credit_score))
        
        return {
            "prediction": "DEFAULT" if prediction == 1 else "APPROVED",
            "probability_of_default": float(prob_default),
            "credit_score": credit_score
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)

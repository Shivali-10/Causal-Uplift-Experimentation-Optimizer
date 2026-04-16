import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_uplift_model.pkl")

app = FastAPI(
    title="Causal ROI Optimization API",
    description="Real-time Uplift Prediction Engine for Criteo Dataset",
    version="1.0.0"
)

# Load the model once at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please run research/modeling.py first.")

model = joblib.load(MODEL_PATH)

class VisitorData(BaseModel):
    # Criteo has 12 features f0 to f11
    f0: float = Field(..., example=12.5)
    f1: float = Field(..., example=0.5)
    f2: float = Field(..., example=10.0)
    f3: float = Field(..., example=4.5)
    f4: float = Field(..., example=0.1)
    f5: float = Field(..., example=0.0)
    f6: float = Field(..., example=2.0)
    f7: float = Field(..., example=1.0)
    f8: float = Field(..., example=15.2)
    f9: float = Field(..., example=0.8)
    f10: float = Field(..., example=0.3)
    f11: float = Field(..., example=5.5)

class RecommendationResponse(BaseModel):
    uplift_score: float
    treatment_recommendation: str
    action_label: str

@app.get("/")
def read_root():
    return {"message": "CROP API is online. Use /docs for documentation."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/recommend-treatment", response_model=RecommendationResponse)
def recommend_treatment(visitor: VisitorData):
    try:
        # Convert input to DataFrame for the model
        data = pd.DataFrame([visitor.dict()])
        
        # Predict uplift score
        score = float(model.predict(data)[0])
        
        # Determine Recommendation Recommendation logic:
        # In a real scenario, you'd use a threshold (e.g. median uplift or 0)
        # For Criteo ClassTransformation, scores are often small floats
        if score > 0.01:
            recommendation = "TREAT: Target this visitor with ads."
            label = "TARGET"
        elif score < -0.005:
            recommendation = "AVOID: This visitor is a 'Sleeping Dog'."
            label = "AVOID"
        else:
            recommendation = "HOLD: Not worth the marketing spend."
            label = "HOLD"
            
        return RecommendationResponse(
            uplift_score=score,
            treatment_recommendation=recommendation,
            action_label=label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

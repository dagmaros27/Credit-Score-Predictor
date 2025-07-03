from pydantic_models import RiskRequest, RiskResponse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

# Load the latest model from MLflow registry
MODEL_NAME = "Random_Forest"
STAGE = "Production"
model_uri = f"models:/{MODEL_NAME}/{STAGE}"
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict", response_model=RiskResponse)
async def predict_risk(data: RiskRequest):
    input_df = pd.DataFrame([data.dict()])
    risk_prob = model.predict_proba(input_df)[:, 1][0]
    
    return RiskResponse(risk_probability=risk_prob)

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
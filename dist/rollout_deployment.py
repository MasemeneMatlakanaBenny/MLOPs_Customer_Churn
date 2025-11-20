from typing import Optional
try:
  from fastapi import FastAPI,HTTPException
  from pydantic import BaseModel

except ImportError:
  raise ImportError("modules not found")

class LiveModelPrediction(BaseModel):
  live_model_name:str="blue model"
  live_model_prediction:int

class PredictionInput(BaseModel):
  X1:int



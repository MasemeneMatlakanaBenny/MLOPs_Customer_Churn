from typing import Optional
try:
  from fastapi import FastAPI,HTTPException
  from pydantic import BaseModel
  from schemas import LiveModel,PredictionInput
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.linear_model import LogisticRegression
  
except ImportError:
  raise ImportError("modules not found")

LIVE_MODEL:Optional[LogisticRegression]=None
SHADOW_MODEL:Optional[DecisionTreeClassifier]=None

def load_models(live_model_path,shadow_model_path):
  import joblib
  global LIVE_MODEL,SHADOW_MODEL
  LIVE_MODEL=joblib.load(live_model_path)
  SHADOW_MODEL=joblib.load(shadow_model_path)


app=FastAPI(title="ML Rollout deployment",
            description="Model versioning in production")

@app.on_event("startup event")
def startup_event():
  load_models()


  
  




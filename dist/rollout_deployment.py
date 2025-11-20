from typing import Optional
try:
  from fastapi import FastAPI,HTTPException
  from pydantic import BaseModel
  from schemas import LiveModel,PredictionInput
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.linear_model import LogisticRegression
  
except ImportError:
  raise ImportError("modules not found")

BLUE_MODEL:Optional[LogisticRegression]=None
GREEN_MODEL:Optional[DecisionTreeClassifier]=None

def load_models(live_model_path,shadow_model_path):
  import joblib
  global BLUE_MODEL,GREEN_MODEL
  BLUE_MODEL=joblib.load(live_model_path)
  GREEN_MODEL=joblib.load(shadow_model_path)


app=FastAPI(title="ML Rollout deployment",
            description="Model versioning in production")

@app.on_event("startup event")
def startup_event():
  load_models()

@app.post("/rollout modeling",response_model=LiveModel,tags=["inference"])
async def rollout_deployment(input_data:PredictionInput,response_model):
  features=np.array(input_data.X1)
  blue_prediction:int=BLUE_MODEL.predict(features)
  green_prediction:int=GREEN_MODEL.predict(features)
  
  
  




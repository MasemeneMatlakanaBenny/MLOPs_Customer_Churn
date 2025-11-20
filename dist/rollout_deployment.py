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





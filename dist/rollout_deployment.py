from typing import Optional
try:
  from fastapi import FastAPI,HTTPException
  from pydantic import BaseModel
  from schemas import LiveModel,PredictionInput
  
except ImportError:
  raise ImportError("modules not found")






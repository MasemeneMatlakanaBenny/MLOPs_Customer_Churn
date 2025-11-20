from typing import Optional
try:
  from fastapi import FastAPI,HTTPException
  from pydantic import BaseModel

except ImportError:
  raise ImportError("modules not found")
  

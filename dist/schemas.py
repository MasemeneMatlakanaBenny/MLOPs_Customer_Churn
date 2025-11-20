class LiveModelPrediction(BaseModel):
  live_model_name:str="blue model"
  live_model_prediction:int

class PredictionInput(BaseModel):
  X1:int

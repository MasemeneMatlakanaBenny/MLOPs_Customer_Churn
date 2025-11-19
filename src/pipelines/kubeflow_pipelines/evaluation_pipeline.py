import pandas as pd
import joblib
from configurations import X_train_y_train,model_metrics
from kfp import dsl,compiler
from kfp.dsl import pipeline,component


## load the dataset to use for evaluation:
@component
def load_data():
  X_test,y_test=X_train_y_train(path="data/test_data.csv")
  
  return X_test,y_test

## load the models using joblib
@component
def load_blue_model():
  blue_model=joblib.load("models/blue_model.pkl")
  
  return blue

@component
def load_green_model():
  green_model=joblib.load("models/green_model.pkl")
  
  return green_model

## get the  performance of both models:
@component
def blue_model_eval(X_test,y_test):
  blue_metrics=model_metrics(y_test,X_test,blue_model)

  return blue_metrics

## get the  performance of both models:
@component
def green_model_eval():
  green_metrics=model_metrics(y_test,X_test,green_model)
  
  return green_metrics

## save the metrics of both models as individual pickle files:
@component
def save_model_metrics(blue_metrics,green_metrics):
  joblib.dump(log_metrics,"metrics/blue_metrics.pkl")
  joblib.dump(dt_metrics,"metrics/green_metrics.pkl")

## build two robust model evaluation pipelines :
@pipeline(
  name="model evaluation ml pipeline",
  description="pipeline that is used to evaluate the two ml models in production"
)
def model_eval_pipeline():
  X_test,y_test=load_data()
  blue_model=load_blue_model()
  green_model=load_green_model()
  ## get the metrics:
  blue_metrics=blue_model_eval(X_test=X_test,y_test=y_test)
  green_metrics=green_model_eval(X_test=X_test,y_test=y_test)

  ## save the metrics now:
  save_model_metrics.submit(blue_metrics=blue_metrics,green_metrics=green_metrics)

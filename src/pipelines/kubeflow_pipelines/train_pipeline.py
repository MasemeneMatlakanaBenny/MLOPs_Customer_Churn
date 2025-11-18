from kfp import dsl,compiler
from kfp.dsl import pipeline,component
@component
def data_loading_component():
  """
  Component for loading the X_train and y_train
  """
  from configurations import X_train_y_train
  
  X_train,y_train=X_train_y_train()
  
  return X_train,y_train

@component
def save_model(model,model_path):
  """
  Component that will save any incoming model
  """
  import joblib
  joblib.dump(model,model_art.path)

@component
def blue_model_component(X_train,y_train):
  """
  Component for model training for the baseline model
  """
  from sklearn.linear_model import LogisticRegression
  
  model=LogisticRegression(solver="liblinear")
  model.fit(X_train,y_train)
  
  return model
  
@component
def green_model_component(X_train,y_train):
  """
  Component for model training for the challenger model
  """
  from sklearn.tree import DecisionTreeClassifier
  
  model=DecisionTreeClassifier()
  model.fit(X_train,y_train)
  
  return model


@pipeline(
  name="train_pipeline_or_model_development_pipeline",
  description="A kubeflow pipeline that is used to develop two models simultaneously i.e green model and blue model"
)
def model_development_pipeline():
  """
  The entire model development pipeline

  Load X_train & y_train
     ***
     ***
  Baseline Model Development
     ***
     ***
  Challenger Model Development
  """
  X_train,y_train=data_loading_component()
  blue_model=blue_model_component(X_train=X_train,y_train=y_train)
  green_model=green_model_component(X_train=X_train,y_train=y_train)
  save_model(blue_model,model_path="models/blue_model.pkl")
  save_model(blue_model,model_path="models/blue_model.pkl")
  

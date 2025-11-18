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



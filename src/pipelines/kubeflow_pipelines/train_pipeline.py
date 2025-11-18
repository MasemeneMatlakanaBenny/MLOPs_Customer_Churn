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


from prefect import flow,task
from prefect.task_runners import ConcurrentTaskRunner,DaskTaskRunner


@task
def data_loading():
  """
  A function for loading the dataset
  """
  from configurations import X_train_y_train
  
  X_train,y_train=X_train_y_train()  ## load the X_train and y train:
  return X_train,y_train

@task
def blue_model_training(X_train,y_train):
  """
  A function for model training 
  """
  from sklearn.linear_model import LogisticRegression
  
  model=LogisticRegression(solver="liblinear")
  model.fit(X_train,y_train)
  
  return model

@task
def green_model_training(X_train,y_train):
  """
  A function for model training for the second model 
  """
  from sklearn.tree import DecisionTreeClassifier
  
  model=DecisionTreeClassifier()
  model.fit(X_train,y_train)
  
  return model

@flow(name="Model Development Concurrent Pipeline",task_runner=ConcurrentTaskRunner())
def concurrent_model_development_pipeline():
  import joblib
  
  X_train,y_train=data_loading.submit()
  blue_model=blue_model_training.submit(X_train=X_train,y_train=y_train)
  green_model=green_model_trainig.submit(X_train=X_train,y_train=y_train)
  joblib.dump(blue_model,"models/blue_model.pkl")
  joblib.dump(green_model,"models/green_model.pkl")


@flow(name="Model Development Distributed Pipeline",task_runner=DaskTaskRunner())
def distributed_model_development_pipeline():
  import joblib
  
  X_train,y_train=data_loading.submit()
  blue_model=blue_model_training.submit(X_train=X_train,y_train=y_train)
  green_model=green_model_trainig.submit(X_train=X_train,y_train=y_train)
  joblib.dump(blue_model,"models/blue_model.pkl")
  joblib.dump(green_model,"models/green_model.pkl")

import mlflow.sklearn
from src.configurations_mlflow import set_mlflow_exp,set_mlflow_host,load_blue_model_name,load_green_model_name


## set the mlflow host and experiment within the workflow:
set_mlflow_host()
set_mlflow_exp()

## get the model names:
blue_model_name=load_blue_model_name()
green_model_name=load_green_model_name()


from configurations_mlflow import set_mlflow_exp,set_mlflow_host,load_green_model_name
from configurations_mlflow import test_model_registry,mlflow_client


## set the mlflow experiment and mlflow host within the workflow:
set_mlflow_exp()
set_mlflow_host()

model_name=load_green_model_name()
model_version="1"

## client:
client=mlflow_client()

test_stage_phase=test_model_registry(name=model_name,version=model_version,client_var=client)

print(test_stage_phase)

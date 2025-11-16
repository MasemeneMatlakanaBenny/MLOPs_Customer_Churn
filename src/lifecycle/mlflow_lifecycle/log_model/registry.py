import mlflow
from mlflow.models import infer_signature
from src.configurations import X_train_y_train
from src.configurations_mlflow import set_mlflow_host,set_mlflow_exp,load_log_model

# set mlflow experience and host within the workflow:
set_mlflow_exp()
set_mlflow_host()

## get the X_train and y_train:
X_train,X_train=X_train_y_train()

## get the model name and the model file:
model_name,model_file=load_log_model()

## create the signature that will be logged into when registering the model:
signature=infer_signature(X_train,model_file.predict(X_train))

with mlflow.start_run(run_name="dt_model_run") as run:
    mlflow.sklearn.log_model(sk_model=model_file,
                             registered_model_name=model_name)
    


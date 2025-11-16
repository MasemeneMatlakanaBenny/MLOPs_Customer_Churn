import pandas as pd
import joblib
from configurations import X_train_y_train,model_metrics


## load the models first using joblib
log_model=joblib.load("models/log_model.pkl")

dt_model=joblib.load("models/dt_model.pkl")


## get the  performance of both models:

X_test,y_test=X_train_y_train(path="data/test_data.csv")
dt_metrics=model_metrics(y_test,X_test,dt_model)

log_metrics=model_metrics(y_test,X_test,log_model)

## save the metrics of both models as individual pickle files:

joblib.dump(log_metrics,"metrics/log_metrics.pkl")
joblib.dump(dt_metrics,"metrics/dt_metrics.pkl")

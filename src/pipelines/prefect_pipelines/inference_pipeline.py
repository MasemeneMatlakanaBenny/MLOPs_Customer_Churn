from prefect import task,flow

@task
def model_prediction(input_data,model):
  return model.predict(input_data)

@flow
def inference_pipeline():
  return None # for now
  

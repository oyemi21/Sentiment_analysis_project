import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import logging
import os
from config.constant import model_name,registered_model_name

def get_best_model(experiment_name = "10Alytics_sentiment_analysis_new_run"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        return None
    
    best_model = sorted(
        runs,
        key = lambda x:x.data.metrics.get("f1", 0),
    )[0]

    return best_model

def get_best_f1(experiment_name =  "10Alytics_sentiment_analysis_new_run"):
    best_runs = get_best_model(experiment_name)
    if best_runs is None:
        return None
    return best_runs.data.metrics.get("f1", 0)

def load_registered_model(model_name = registered_model_name):
    dagshub.init(repo_owner='oyemi21', 
                 repo_name='Sentiment_analysis_project', 
                 mlflow=True)
    model_uri = f"models:/{model_name}/latest"
    sentiment_pipeline = mlflow.transformers.load_model(model_uri)
    return sentiment_pipeline


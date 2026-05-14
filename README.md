# Sentiment_analysis_project
This project is all about creating a full sentiment analysis NLP application for a ecommerce platform called ShopEase. 
This was built with a fine-tuned DistilBERT model, FAST API and Streamlit. The project follows an end-to-end MLOps pipeline with experimental tracking via MLflow and DagsHub.

## Project Overview
This project classifies product reviews into three sentiment categories:
  *  Positive
  *  Neutral
  *  Negative

The model supports multilingual reviews using distilbert-base-multilingual-cased as the base model

## MLOps Pipeline
*  Data Cleaning -> Data Ingestion -> Data Preprocessing -> Model Training and Evaluation -> Register best model to MLflow -> FastAPI -> Streamlit app launch
*  Experiment Tracking : MLflow hosted on DagsHub
*  Model Registry: MLflow Model Registry (sentiment_model)

## Experiment Tracking
Experiments are tracked on DagsHub:
https://dagshub.com/oyemi21/Sentiment_analysis_project
Metrics looged per run:
*  f1
*  accuracy
*  loss

Parameters logged:
*  batch size
*  learning rate

## Tech Stack
Python 3.14+
Google Colab
HuggingFace Transformers
DagsHub
MLflow
FastAPI
Streamlit
Uvicorn
Torch
Spacy
Nltk









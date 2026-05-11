import logging
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from pipeline.training import Train_model
from pipeline.prediction import predict_sentiment
from pydantic import BaseModel
import io

logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title = "shopease sentiment system api")

class TextRequest(BaseModel):
    text: str

# load the model as the API starts up
predictor = predict_sentiment()
logging.info("the model has been loaded successfully.")


@app.post("/predict")
def predict_user_sentiment(request: TextRequest):
    try:
        result = predictor.predict(request.text)
        print(result)
        top_label = max(result, key=lambda x: x['score'])
        return {
            "label": top_label['label'],
            "confidence": float(top_label['score'])
        }
    except Exception as e:
        logging.error(f"error occurred during prediction: {e}")
        return {"error": str(e)}


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))

    # check if the required columns 'review' is present
    if 'review' not in df.columns:
        return {"error": "CSV file must contain a 'review' column"}
    
    # predict each review
    result_list =[]
    for idx, row in df.iterrows():
        try:
            review = str(row['review'])

            # call the predict function
            result = predictor.predict(review)

            if result is None or len(result) ==0:
                raise ValueError("Empty result from the model")
            
            top_label = max(result, key= lambda x: x['score'])
            result_row = row.to_dict()
            result_row['sentiment_label'] = top_label['label']
            result_row['sentiment_confidence'] = float(top_label['score'])
            result_list.append(result_row)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"error for review {idx}: {error_details}")
            result_row = row.to_dict()
            result_row['sentiment_label'] = "error"
            result_row['sentiment_confidence'] = 0.0
            result_list.append(result_row)
    return result_list


@app.get("/train")
def train_model():
    try:
        Train_model()
    except Exception as e:
       logging.error(f"error occurred during training: {e}")
       return {"error": str(e)}
    

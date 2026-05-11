import transformers
import numpy as np

from transformers import AutoModelForSequenceClassification, Trainer
from config.constant import training_args, model_name, number_of_labels
from sklearn.metrics import accuracy_score, f1_score
from src.data_preprocessing import Prepare_sentiment_data
from src.model_pusher import ModelPusher

import logging

logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

class ModelTraining:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=number_of_labels)

    def compute_metrics(self, pred):
        prediction = np.argmax(pred.predictions, axis = 1)
        labels = pred.label_ids
        acc = accuracy_score(labels, prediction)
        f1 = f1_score(labels, prediction, average ='weighted')
        return {"accuracy": acc, "f1": f1}
    
    def model_training(self, train_dataset, test_dataset):
        try:
            trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = test_dataset,
            compute_metrics = self.compute_metrics
            )
            trainer.train()
            logging.info("Model is successfully trained...")
            return trainer
        except Exception as e:
            logging.error(f"error occurred while training the model: {e}")

    def model_evaluation(self, trainer):
        results = trainer.evaluate()
        return results


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.data_ingestion import data_ingestion
from config.constant import Clean_data
import re
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)




class DataCleaning:
    def __init__(self):
        self._ensure_nltk()

    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                return spacy.load(model)
            except OSError:
                continue
        nlp_fallback = spacy.blank("xx")
        return nlp_fallback
    
    def _ensure_nltk(self) -> None:
     try:
        _ = stopwords.words("english")
     except LookupError:
        nltk.download("stopwords")
    try:
        word_tokenize("test")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        try:
         nltk.download("punkt_tab")
        except Exception:
            pass
    
    def clean_text(self, text: str) -> str:
        """convert the data to lower case
            remove url and special characters
            remove extra white space
            keeps accented letter and regular expressions
            return cleaned text
        """
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"http\s+|www\s+", " ", text).strip()
        return text
    
    def lemmatize(self, text: str) -> str:
        """groups different form of words so they can be analyzed in a single item
        example running becomes run, ran also becomes run.
        """
        nlp = self._load_nlp()
        doc = nlp(text)
        return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)
    
    def remove_stopwords(self, text: str) -> str:
        """remove stop words like (the, is, in) that don't carry much sentiment
        """
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)
    

def clean_data(data: pd.DataFrame):
    try:
        Cleaner = DataCleaning()
        data['clean_text'] = data['review'].apply(Cleaner.clean_text)
        data['lemma_text'] = data['clean_text'].apply(Cleaner.lemmatize)
        data['final_text'] = data['lemma_text'].apply(Cleaner.remove_stopwords)

        data['label'] = data['rating'].apply(lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2))
        logging.info("dataset has been successfully cleaned")
        data = data[["review", "final_text", "label"]]
        data.to_csv(Clean_data)
        data.head() 
        print(data.head())
        return data
    except Exception as e:
        logging.error(f"error occurred while cleaning the data {e}")


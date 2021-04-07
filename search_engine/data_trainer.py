import os

from sklearn.model_selection import train_test_split

from profile_builder import interest_integrater
from data_builder import DATA_SAVE_DIR
from search_engine.ml_models import NaiveBayesClassifier


class DataTrainer:
    def __init__(self):
        self.ml_model_builder = NaiveBayesClassifier()

    def get_training_test_data(self) -> list:
        interests = interest_integrater.get_interests()
        filepaths = [os.path.join(DATA_SAVE_DIR, each) for each in interests]
        labels = []
        texts = []

        for each in filepaths:
            with open(each, 'r') as file:
                for each_sentence in file.read().splitlines():
                    texts.append(each_sentence)
                    labels.append(each.split('/')[1])
        return train_test_split(texts, labels, test_size=.3, random_state=1)

    def train_documents(self):
        return self.ml_model_builder.get_trained_model(*self.get_training_test_data())

    def predict_interest(self, doc):
        ml_pred = self.ml_model_builder.predict(doc)
        return ml_pred[0]










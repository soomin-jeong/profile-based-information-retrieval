import os
import numpy as np

from data_preprocessor import preprocess_sentence

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from profile_builder import interest_integrater
from data_builder import SAVE_DIR


class MachineLearningModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=preprocess_sentence)
        self.trained_model = None


class NaiveBayesClassifier(MachineLearningModel):
    def __init__(self):
        super().__init__()
        self.trained_model = None

    def train_model(self, x_train, y_train):
        vec_train = self.vectorizer.fit_transform(x_train)
        return MultinomialNB().fit(vec_train, y_train)

    def evaluate(self, x_test, y_test):
        vec_test = self.vectorizer.transform(x_test)
        predicted = self.trained_model.predict(vec_test)
        return np.mean(predicted == y_test)

    def predict(self, doc):
        vectorized_doc = self.vectorizer.transform([doc])
        return self.trained_model.predict(vectorized_doc)


class DataTrainer:
    def __init__(self):
        self.ml_model = NaiveBayesClassifier()

    def get_training_test_data(self) -> list:
        interests = interest_integrater.get_interests()
        filepaths = [os.path.join(SAVE_DIR, each) for each in interests]
        labels = []
        texts = []

        for each in filepaths:
            with open(each, 'r') as file:
                for each_sentence in file.read().splitlines():
                    texts.append(each_sentence)
                    labels.append(each.split('/')[1])
        return train_test_split(texts, labels, test_size=.3, random_state=1)

    def train_documents(self):
        x_train, x_test, y_train, y_test = self.get_training_test_data()
        ml_model = self.ml_model.train_model(x_train, y_train)
        accuracy = self.ml_model.evaluate(x_test, y_test)
        print("Accuracy: ", accuracy)
        return ml_model

    def predict_interest(self, doc):
        ml_pred = self.ml_model.predict(doc)
        return ml_pred










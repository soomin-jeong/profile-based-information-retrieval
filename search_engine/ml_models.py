import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from search_engine.data_preprocessor import preprocess_sentence


class MachineLearningModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=preprocess_sentence)
        self.trained_model = None
        self.model_fp = os.path.join('models', self.model_name)
        self.vec_fp = os.path.join('vecs', self.vectorizer_name)

    @property
    def model_name(self):
        raise NotImplementedError("Subclass should implement this")

    @property
    def vectorizer_name(self):
        raise NotImplementedError("Subclass should implement this")

    def load_model(self):
        return pickle.load(open(self.model_fp, 'rb'))

    def load_vectorizer(self):
        return pickle.load(open(self.vec_fp, 'rb'))

    def train_model(self, x_train_, y_train, x_test, y_test):
        raise NotImplementedError("Subclass should implement this")

    def get_trained_model(self, x_train, y_train, x_test, y_test):
        try:
            return self.load_model()
        except FileNotFoundError:
            return self.train_model(x_train, y_train, x_test, y_test)
        except TypeError:
            return self.train_model(x_train, y_train, x_test, y_test)


class NaiveBayesClassifier(MachineLearningModel):
    def __init__(self):
        super().__init__()

    @property
    def model_name(self):
        return 'nb'

    @property
    def vectorizer_name(self):
        return 'tfidf_vec'

    def train_model(self, x_train, y_train, x_test, y_test):
        vec_train = self.vectorizer.fit_transform(x_train)
        model = MultinomialNB()
        model.fit(vec_train, y_train)
        accuracy = self.evaluate(x_test, y_test)
        print("Accuracy: ", accuracy)
        with open(self.model_fp, "wb") as f, open(self.vec_fp, "wb") as vf:
            pickle.dump(model, f)
            pickle.dump(self.vectorizer, vf)
        return model

    def evaluate(self, x_test, y_test):
        self.vectorizer = self.load_vectorizer()
        vec_test = self.vectorizer.transform(x_test)
        predicted = self.load_model().predict(vec_test)
        return np.mean(predicted == y_test)

    def predict(self, doc):
        self.vectorizer = self.load_vectorizer()
        vectorized_doc = self.vectorizer.transform([doc])
        return self.trained_model.predict(vectorized_doc)

# class DecisionTreeClassifier(MachineLearningModel):
#     def __init__(self):
#         self.model_name = 'dt'
#         self.traoined_model = self.load_model()
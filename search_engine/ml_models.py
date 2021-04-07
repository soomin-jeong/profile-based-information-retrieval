import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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

    def train_model(self, x_train, x_test, y_train, y_test):
        model = Pipeline([('count_vec', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('nb', MultinomialNB())])

        model.fit(x_train, y_train)
        self.trained_model = model

        predicted = model.predict(x_test)
        accuracy = np.mean(predicted == y_test)
        print("Accuracy: ", accuracy)
        with open(self.model_fp, "wb") as f, open(self.vec_fp, "wb") as vf:
            pickle.dump(model, f)
        return model

    def predict(self, doc):
        # self.vectorizer = self.load_vectorizer()
        # vectorized_doc = self.vectorizer.transform([doc])
        model = self.trained_model or self.load_model()
        return model.predict([doc])

# class DecisionTreeClassifier(MachineLearningModel):
#     def __init__(self):
#         self.model_name = 'dt'
#         self.traoined_model = self.load_model()
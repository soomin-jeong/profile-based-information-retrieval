import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class MachineLearningModel:
    def __init__(self):
        self.trained_model = None
        self.model_fp = os.path.join('models', self.model_name)

    @property
    def model_name(self):
        raise NotImplementedError("Subclass should implement this")

    def load_model(self):
        return pickle.load(open(self.model_fp, 'rb'))

    def train_model(self, x_train_, y_train, x_test, y_test):
        raise NotImplementedError("Subclass should implement this")

    def predict(self, doc):
        raise NotImplementedError("Subclass should implement this")

    def get_trained_model(self, x_train, y_train, x_test, y_test):
        try:
            return self.load_model()
        except FileNotFoundError:
            return self.train_model(x_train, y_train, x_test, y_test)
        except TypeError:
            return self.train_model(x_train, y_train, x_test, y_test)

    def save_model(self, model):
        with open(self.model_fp, "wb") as f:
            pickle.dump(model, f)


class NaiveBayesClassifier(MachineLearningModel):
    @property
    def model_name(self):
        return 'NaiveBayes'

    def train_model(self, x_train, x_test, y_train, y_test):
        model = Pipeline([('count_vec', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('nb', MultinomialNB(fit_prior=False))])

        model.fit(x_train, y_train)

        self.trained_model = model
        self.save_model(model)

        predicted = model.predict(x_test)
        accuracy = np.mean(predicted == y_test)
        print("\n[{}] Accuracy: {}".format(self.model_name, accuracy))
        return model

    def predict(self, doc):
        model = self.trained_model or self.load_model()
        return model.predict([doc])


class SVMClassifier(MachineLearningModel):
    @property
    def model_name(self):
        return 'SVM'

    def train_model(self, x_train, x_test, y_train, y_test):
        model = Pipeline([('count_vec', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('svm', SGDClassifier(random_state=1))])

        model.fit(x_train, y_train)
        self.trained_model = model

        predicted = model.predict(x_test)
        accuracy = np.mean(predicted == y_test)
        print("[{}] Accuracy: {}".format(self.model_name, accuracy))
        self.save_model(model)
        return model

    def predict(self, doc):
        model = self.trained_model or self.load_model()
        return model.predict([doc])


class DecisionTree(MachineLearningModel):

    @property
    def model_name(self):
        return 'DecisionTree'

    def train_model(self, x_train, x_test, y_train, y_test):
        model = Pipeline([('count_vec', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('dt', DecisionTreeClassifier(max_depth=8, random_state=4))])

        model.fit(x_train, y_train)
        self.trained_model = model

        predicted = model.predict(x_test)
        accuracy = np.mean(predicted == y_test)
        print("[{}] Accuracy: {}".format(self.model_name, accuracy))
        self.save_model(model)
        return model

    def predict(self, doc):
        model = self.trained_model or self.load_model()
        return model.predict([doc])


class RandomForest(MachineLearningModel):

    @property
    def model_name(self):
        return 'RandomForest'

    def train_model(self, x_train, x_test, y_train, y_test):
        model = Pipeline([('count_vec', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('rf', RandomForestClassifier(max_depth=10, random_state=4))])

        model.fit(x_train, y_train)
        self.trained_model = model

        predicted = model.predict(x_test)
        accuracy = np.mean(predicted == y_test)
        print("[{}] Accuracy: {}".format(self.model_name, accuracy))
        self.save_model(model)
        return model

    def predict(self, doc):
        model = self.trained_model or self.load_model()
        return model.predict([doc])


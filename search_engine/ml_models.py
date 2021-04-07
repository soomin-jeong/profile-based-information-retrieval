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
    def __init__(self, model_name: str, pipeline: Pipeline):
        self.trained_model = None
        self.model_fp = os.path.join('models', self.model_name)
        self.model_name = model_name
        self.pipeline = pipeline

    def load_model(self):
        return pickle.load(open(self.model_fp, 'rb'))

    def train_model(self, x_train, x_test, y_train, y_test):
        model = self.pipeline
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


naive_bayes_pipeline = Pipeline([('count_vec', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('nb', MultinomialNB(fit_prior=False))])
NaiveBayesClassifier = MachineLearningModel('NaiveBayes', naive_bayes_pipeline)

svm_pipeline = Pipeline([('count_vec', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('svm', SGDClassifier(random_state=1))])
SVMClassifier = MachineLearningModel('SVM', svm_pipeline)

dt_pipeline = Pipeline([('count_vec', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('dt', DecisionTreeClassifier(max_depth=8, random_state=4))])
DecisionTree = MachineLearningModel('DecisionTree', dt_pipeline)

rf_pipeline = Pipeline([('count_vec', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('rf', RandomForestClassifier(max_depth=10, random_state=4))])
RandomForest = MachineLearningModel('RandomForest', rf_pipeline)


class MachineLearningModelFactory:
    def __init__(self, ml_models):
        self.ml_models = ml_models

    def train_model(self, x_train, x_test, y_train, y_test):
        for each in self.ml_models:
            each.train_model(x_train, x_test, y_train, y_test)

    def predict(self, doc):
        preds = []
        for each in self.ml_models:
            each_pred = each.predict(doc)
            preds.append(each_pred)
        return max(preds, key=preds.count)


ml_factory = MachineLearningModelFactory([NaiveBayesClassifier,
                                          SVMClassifier,
                                          DecisionTree,
                                          RandomForest])

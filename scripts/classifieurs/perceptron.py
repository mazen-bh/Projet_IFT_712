# Importations nécessaires
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Classe Perceptron
class Perceptron_model(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = None
        self.perceptron_classifier = Perceptron()

    def validation_croisee_gridsearch(self):
        parameters = {
            'alpha': [0.0001, 0.001, 0.01],
            'tol': [1e-3, 1e-4, 1e-5],
            'max_iter': [1000, 2000, 3000],
            'eta0': [0.1, 0.01, 0.001]
        }

        clf = GridSearchCV(self.perceptron_classifier, parameters, cv=5, n_jobs=-1)
        clf.fit(self.x_val, self.y_val)  
    
        self.perceptron_classifier = clf.best_estimator_

        print("Meilleurs hyperparamètres:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.perceptron_classifier.fit(self.x_train, self.y_train)  
        


    def prediction(self):
        return self.perceptron_classifier.predict(self.x_test)
    
    def prediction_proba(self):
        return self.perceptron_classifier.predict_proba(self.x_test)
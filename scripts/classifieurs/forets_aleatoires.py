from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class Forets_aleatoires(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.rf_classifier = RandomForestClassifier()

    def validation_croisee_gridsearch(self):
        parameters = {
            'n_estimators': [50, 100, 200, 300],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 4, 8, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'bootstrap': [True, False],
            'min_impurity_decrease': [0.0, 0.01, 0.02],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }

        clf = GridSearchCV(self.rf_classifier, parameters, cv=5, n_jobs=-1, scoring='accuracy')
        clf.fit(self.x_train, self.y_train)  # Utilisation de x_train et y_train pour GridSearchCV

        self.rf_classifier = clf.best_estimator_

        print("Meilleurs hyperparamètres:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.rf_classifier.fit(self.x_train, self.y_train)  # Entraînement avec x_train et y_train

    def prediction(self):
        return self.rf_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.rf_classifier.predict_proba(self.x_test)

    def evaluation(self):
        y_pred = self.prediction()
        print("Matrice de confusion :")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nRapport de classification :")
        print(classification_report(self.y_test, y_pred))

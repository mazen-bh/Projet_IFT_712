
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve

class Arbre_de_decision(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.criterion = 'gini'
        self.min_samples_split = 2
        self.max_depth = None
        self.min_samples_leaf = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = None 
        self.dt_classifier = RandomForestClassifier(
            n_estimators=100,  
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=None,
        )

    def validation_croisee_gridsearch(self):
        parameters = {
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': list(range(1, 10)),
            'min_samples_split': list(range(2, 20, 2)),
            'max_depth': list(range(1, 20)),
        }

        clf = GridSearchCV(self.dt_classifier, parameters, cv=5)  
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        clf.fit(combined_x, combined_y)

        self.criterion = clf.best_params_["criterion"]
        self.min_samples_split = clf.best_params_["min_samples_split"]
        self.min_samples_leaf = clf.best_params_["min_samples_leaf"]
        self.max_depth = clf.best_params_["max_depth"]

        print("Best hyperparameters:", clf.best_params_)
        return combined_x, combined_y
    
    def garder_meilleur_hyperparameters(self):
        combined_x, combined_y = self.validation_croisee_gridsearch()
        self.dt_classifier = RandomForestClassifier(
            n_estimators=100,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=None,
        )

        self.dt_classifier.fit(combined_x, combined_y)

    def entrainement(self):
        model_rf = RandomForestClassifier(
            n_estimators=100,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=None,
            n_jobs=-1  # Utilisation de tous les cœurs disponibles
        )

        model_rf.fit(self.x_train, self.y_train)

        self.dt_classifier = model_rf



    def prediction(self):
        return self.dt_classifier.predict(self.x_test)

    def predict_proba(self):
        return self.dt_classifier.predict_proba(self.x_test)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

class AdaBoost_model(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.n_estimators = 50
        self.learning_rate = 1.0
        self.base_estimator = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = None
        self.ab_classifier = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            base_estimator=self.base_estimator
        )

    def validation_croisee_gridsearch(self):
        parameters = {
            'n_estimators': [50, 75, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
            'random_state': [None, 42],
        }

        scoring = {
            'Accuracy': 'accuracy',
            'Precision': 'precision_weighted'
        }

        clf = GridSearchCV(self.ab_classifier, parameters, cv=2, scoring=scoring, refit='Accuracy')
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        clf.fit(combined_x, combined_y)

        self.n_estimators = clf.best_params_["n_estimators"]
        self.learning_rate = clf.best_params_["learning_rate"]
        self.base_estimator = clf.best_params_["base_estimator"]

        return combined_x, combined_y

    def garder_meilleur_hyperparameters(self):
        combined_x, combined_y = self.validation_croisee_gridsearch()
        self.ab_classifier = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            base_estimator=self.base_estimator
        )

        self.ab_classifier.fit(combined_x, combined_y)

    def entrainement(self):
        model_ab = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            base_estimator=self.base_estimator
        )

        model_ab.fit(self.x_train, self.y_train)
        self.ab_classifier = model_ab


    def prediction(self):
        return self.ab_classifier.predict(self.x_test)

    def predict_proba(self):
        return self.ab_classifier.predict_proba(self.x_test)

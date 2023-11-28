from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
 
class LogisticRegression_model(object):

    param_grid_default = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, param_grid=None):

        self.param_grid = param_grid or self.param_grid_default
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = None
        self.logistic_regression_classifier = LogisticRegression(
            C=1,  # Set a positive value for C
            penalty='l2',  # Default penalty is 'l2'
            solver='liblinear',  # Default solver is 'liblinear'
            max_iter=100

        )

    def validation_croisee_gridsearch(self):
        parameters = {

            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']

        }

        clf = GridSearchCV(self.logistic_regression_classifier, parameters, cv=2)
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        clf.fit(combined_x, combined_y)

        self.logistic_regression_classifier = LogisticRegression(
            C=clf.best_params_["C"],
            penalty=clf.best_params_["penalty"],
            solver=clf.best_params_["solver"],
            max_iter=100

        )

        print("Meilleurs hyperparamètres:", clf.best_params_)
        return combined_x, combined_y

    def garder_meilleur_hyperparameters(self):
        combined_x, combined_y = self.validation_croisee_gridsearch()
        self.logistic_regression_classifier.fit(combined_x, combined_y)

    def entrainement(self):

        clf = LogisticRegression(
            C=1,  # Set a positive value for C
            penalty='l2',  # Default penalty is 'l2'
            solver='liblinear',  # Default solver is 'liblinear'
            max_iter=100

        )

        clf.fit(self.x_train, self.y_train)
        self.logistic_regression_classifier = clf
        train_sizes, train_scores, test_scores = learning_curve(
            clf, self.x_train, self.y_train, cv=2, scoring="accuracy")
        learning_curve_data = {

            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1),
            "train_loss": np.mean(train_scores, axis=1),
            "val_loss": np.mean(test_scores, axis=1)

        }

        self.learning_curve_data = learning_curve_data

    def prediction(self):
        return self.logistic_regression_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.logistic_regression_classifier.predict_proba(self.x_test)

    def resultats_model(self):

        y_pred = self.logistic_regression_classifier.predict(self.x_test)
        print("Matrice de confusion:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nRapport de classification:")
        print(classification_report(self.y_test, y_pred))

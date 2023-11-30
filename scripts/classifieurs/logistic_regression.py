from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.logistic_regression_classifier = LogisticRegression()

    def validation_croisee_gridsearch(self):
        clf = GridSearchCV(self.logistic_regression_classifier, self.param_grid, cv=5, n_jobs=-1)
        clf.fit(self.x_val, self.y_val)  # Utilisation de x_val et y_val pour la recherche d'hyperparamètres

        # Mise à jour du classificateur avec les meilleurs hyperparamètres trouvés
        self.logistic_regression_classifier = clf.best_estimator_

        print("Meilleurs hyperparamètres:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.logistic_regression_classifier.fit(self.x_train, self.y_train)  # Entraînement avec x_train et y_train

        # Génération de la courbe d'apprentissage
        train_sizes, train_scores, test_scores = learning_curve(
            self.logistic_regression_classifier, self.x_train, self.y_train, cv=5, scoring="accuracy")
        self.learning_curve_data = {
            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1)
        }

    def prediction(self):
        return self.logistic_regression_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.logistic_regression_classifier.predict_proba(self.x_test)

    def resultats_model(self):
        y_pred = self.prediction()
        print("Matrice de confusion:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nRapport de classification:")
        print(classification_report(self.y_test, y_pred))

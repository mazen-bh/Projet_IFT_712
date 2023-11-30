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
        clf.fit(self.x_val, self.y_val)  # Utilisation de x_val et y_val pour la recherche d'hyperparamètres

        # Mise à jour du classificateur avec les meilleurs hyperparamètres trouvés
        self.perceptron_classifier = clf.best_estimator_

        print("Meilleurs hyperparamètres:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.perceptron_classifier.fit(self.x_train, self.y_train)  # Entraînement avec x_train et y_train

        # Génération de la courbe d'apprentissage
        train_sizes, train_scores, test_scores = learning_curve(
            self.perceptron_classifier, self.x_train, self.y_train, cv=5, scoring="accuracy")
        
        learning_curve_data = {
            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1)
        }
        self.learning_curve_data = learning_curve_data

    def prediction(self):
        return self.perceptron_classifier.predict(self.x_test)

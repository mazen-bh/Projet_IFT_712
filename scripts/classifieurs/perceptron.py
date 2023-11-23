# Importations nécessaires
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
 
# Classe Perceptron
class Perceptron_Classificateur(object):
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
 
        clf = GridSearchCV(self.perceptron_classifier, parameters, cv=2)
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        clf.fit(combined_x, combined_y)
 
        # Obtenez les meilleurs paramètres du modèle
        best_alpha = clf.best_params_["alpha"]
        best_tol = clf.best_params_["tol"]
        best_max_iter = clf.best_params_["max_iter"]
        best_eta0 = clf.best_params_["eta0"]
 
        # Utilisez les meilleurs paramètres pour le modèle final
        self.perceptron_classifier = Perceptron(alpha=best_alpha, tol=best_tol, max_iter=best_max_iter, eta0=best_eta0)
 
        return combined_x, combined_y
 
    def garder_meilleur_hyperparameters(self):
        combined_x, combined_y = self.validation_croisee_gridsearch()
        self.perceptron_classifier.fit(combined_x, combined_y)
 
    def entrainement(self):
        self.perceptron_classifier = Perceptron()
        self.perceptron_classifier.fit(self.x_train, self.y_train)
 
        train_sizes, train_scores, test_scores = learning_curve(
            self.perceptron_classifier, self.x_train, self.y_train, cv=2, scoring="accuracy")
 
        learning_curve_data = {
            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1),
            "train_loss": np.mean(train_scores, axis=1),
            "val_loss": np.mean(test_scores, axis=1)
        }
        self.learning_curve_data = learning_curve_data
 
    def prediction(self):
        return self.perceptron_classifier.predict(self.x_test)
 
    def resultats_model(self):
        y_pred = self.perceptron_classifier.predict(self.x_test)
        print("Matrice de confusion:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nRapport de classification:")
        print(classification_report(self.y_test, y_pred))
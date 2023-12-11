#Importation des librairies
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Classe pour modele Forets regression logistique
class LogisticRegression_model(object):
    param_grid_default = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    # Constructeur de la classe
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, param_grid=None):
        self.param_grid = param_grid or self.param_grid_default
        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.learning_curve_data = None
        self.logistic_regression_classifier = LogisticRegression()

    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.logistic_regression_classifier, self.param_grid, cv=5, n_jobs=-1)
        # Entrainement du modele avec les paramétres
        clf.fit(self.x_val, self.y_val)  
        # Mise à jour du classificateur avec les meilleurs hyperparametres
        self.logistic_regression_classifier = clf.best_estimator_
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres:", clf.best_params_)

    # Fonction pour entrainer le modèle
    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.logistic_regression_classifier.fit(self.x_train, self.y_train) 
      
    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.logistic_regression_classifier.predict(self.x_test)
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.logistic_regression_classifier.predict_proba(self.x_test)


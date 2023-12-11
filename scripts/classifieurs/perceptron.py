#Importation des librairies
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Classe pour modele de Perceptron
class Perceptron_model(object):
    # Constructeur de la classe
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes 
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.perceptron_classifier = Perceptron() # Initialisation du classificateur du perceptron

    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        parameters = {
            'alpha': [0.0001, 0.001, 0.01],
            'tol': [1e-3, 1e-4, 1e-5],
            'max_iter': [1000, 2000, 3000],
            'eta0': [0.1, 0.01, 0.001]
        }

        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.perceptron_classifier, parameters, cv=5, n_jobs=-1, scoring='accuracy')
        # Entrainement du modele avec les paramétres
        clf.fit(self.x_val, self.y_val)  
        # Mise à jour du classificateur avec les meilleurs hyperparametres
        self.perceptron_classifier = clf.best_estimator_
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres:", clf.best_params_)

    # Fonction pour entrainer le modèle
    def entrainement(self):
        self.validation_croisee_gridsearch() # Appel de la validation croisée
        self.perceptron_classifier.fit(self.x_train, self.y_train) # Entraînement du modèle
        

    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.perceptron_classifier.predict(self.x_test) # Retourne les prédictions
    
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.perceptron_classifier.predict_proba(self.x_test) # Retourne les probabilités de prédiction

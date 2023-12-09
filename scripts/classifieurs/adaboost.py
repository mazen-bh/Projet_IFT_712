#Importation des librairies
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# Classe pour modele AdaBoost
class AdaBoost_model(object):

    # Constructeur de la classe 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):

        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.ab_classifier = AdaBoostClassifier() # Initialisation du classificateur AdaBoost

    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        dt_params = {
            'max_depth': [1],  
            'min_samples_split': [6, 8], 
            'min_samples_leaf': [3, 4],  
        }

        parameters = {
            'n_estimators': [30, 50],  
            'learning_rate': [0.01, 0.1],
            'base_estimator': [DecisionTreeClassifier(**params) for params in ParameterGrid(dt_params)],
            'algorithm': ['SAMME.R']  
        }

        stratified_k_fold = StratifiedKFold(n_splits=5)
        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.ab_classifier, parameters, cv=stratified_k_fold)
        # Entrainement du modele avec les paramétres
        clf.fit(self.x_train, self.y_train)
        # Mise à jour du classificateur avec les meilleurs hyperparametres
        self.ab_classifier.set_params(**clf.best_params_)
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres:", clf.best_params_)

    # Fonction pour entrainer le modèle
    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.ab_classifier.fit(self.x_train, self.y_train)

    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.ab_classifier.predict(self.x_test)
    
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.ab_classifier.predict_proba(self.x_test)

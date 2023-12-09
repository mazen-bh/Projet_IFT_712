#Importation des librairies
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Classe pour modele Forets aleatoires
class Forets_aleatoires(object):
    # Constructeur de la classe 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.rf_classifier = RandomForestClassifier() # Initialisation du classificateur Forets aleatoires

    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        parameters = {
            'n_estimators': [100, 200],  
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 20],  
            'min_samples_split': [2, 10],  
            'min_samples_leaf': [1, 4],  
            'max_features': ['sqrt'],  
            
        }

        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.rf_classifier, parameters, cv=5, n_jobs=-1, scoring='accuracy')
        # Entrainement du modele avec les paramétres
        clf.fit(self.x_train, self.y_train)
        # Mise à jour du classificateur avec les meilleurs hyperparametres
        self.rf_classifier = clf.best_estimator_
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres:", clf.best_params_)

    # Fonction pour entrainer le modèle
    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.rf_classifier.fit(self.x_train, self.y_train)  

    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.rf_classifier.predict(self.x_test)
    
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.rf_classifier.predict_proba(self.x_test)



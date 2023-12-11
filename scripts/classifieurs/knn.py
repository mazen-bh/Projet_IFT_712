#Importation des librairies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
 
 # Classe pour modele Knn
class Knn(object):
    param_grid_default = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
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
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            p=2
        ) # Initialisation du classificateur KNN

    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        parameters = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.knn_classifier, parameters, cv=2)
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        # Entrainement du modele avec les paramétres
        clf.fit(combined_x, combined_y)
 
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=clf.best_params_["n_neighbors"],
            weights=clf.best_params_["weights"],
            p=clf.best_params_["p"]
        )
        # Mise à jour du classificateur avec les meilleurs hyperparametres
        self.knn_classifier = clf.best_estimator_
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres:", clf.best_params_)
        return combined_x, combined_y

    # Fonction pour entrainer le modèle
    def entrainement(self):
        clf = KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            p=2
        )
        clf.fit(self.x_train, self.y_train)
        self.knn_classifier = clf
    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.knn_classifier.predict(self.x_test)
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.knn_classifier.predict_proba(self.x_test)
 
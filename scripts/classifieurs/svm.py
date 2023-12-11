#Importation des librairies
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
 
# Classe pour modele SVM
class SVM_Classificateur(object):
    param_grid_default = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}
 
    # Constructeur de la classe
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test,param_grid=None):
        self.param_grid = param_grid or self.param_grid_default
        self.nmb_arbre = 0
        self.criterion = 'gini'
        self.min_samples_split = 2
        self.max_depth = None
        self.max_features = 'auto'
        self.x_train = x_train # Données d'entrainement
        self.y_train = y_train # Etiquettes 
        self.x_val = x_val # Données de validation
        self.y_val = y_val # Etiquettes de validation
        self.x_test = x_test # Données de test
        self.y_test = y_test # Etiquettes de test
        self.learning_curve_data = None
        self.svm_classifier = svm.SVC(
            C=self.nmb_arbre,
            gamma='scale',
            kernel='rbf',
            probability=True,
            decision_function_shape='ovr'
   
        )
    # Fonction pour effectuer la validation croisée et la recherche des meilleurs hyperparametres
    # avec gridsearch
    def validation_croisee_gridsearch(self):
        parameters = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }
        # Création d'une instance de GridSearchCV avec 5 folds de validation croisée
        clf = GridSearchCV(self.svm_classifier, parameters, cv=5)
        clf.fit(self.x_val, self.y_val)  

        # Entrainement du modele avec les paramétres
        self.svm_classifier = svm.SVC(
            C=clf.best_params_['C'],
            gamma=clf.best_params_['gamma'],
            kernel=clf.best_params_['kernel'],
            probability=True
        )
        # Affichage des meilleurs hyperparamètres
        print("Meilleurs hyperparamètres:", clf.best_params_)

    # Fonction pour entrainer le modèle
    def entrainement(self):
        clf = svm.SVC(
            C=1,  
            gamma='scale',
            kernel='rbf',
            probability=True
        )
        clf.fit(self.x_train, self.y_train)
        self.svm_classifier = clf
    # Fonction pour  faire des prediction sur les données de test
    def prediction(self):
        return self.svm_classifier.predict(self.x_test)
    
    # Fonction pour obtenir les probabilités de la prédiction
    def prediction_proba(self):
        return self.svm_classifier.predict_proba(self.x_test)

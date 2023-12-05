from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
 
class SVM_Classificateur(object):
    param_grid_default = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}
 
 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test,param_grid=None):
        self.param_grid = param_grid or self.param_grid_default
        self.nmb_arbre = 0
        self.criterion = 'gini'
        self.min_samples_split = 2
        self.max_depth = None
        self.max_features = 'auto'
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = None
        self.svm_classifier = svm.SVC(
            C=self.nmb_arbre,
            gamma='scale',
            kernel='rbf',
            probability=True,
            decision_function_shape='ovr'
   
        )
 
    def validation_croisee_gridsearch(self):
        parameters = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }

        clf = GridSearchCV(self.svm_classifier, parameters, cv=2)
        clf.fit(self.x_val, self.y_val)  # Utiliser les données de validation pour la recherche d'hyperparamètres

        # Mise à jour du classificateur SVM avec les meilleurs hyperparamètres trouvés
        self.svm_classifier = svm.SVC(
            C=clf.best_params_['C'],
            gamma=clf.best_params_['gamma'],
            kernel=clf.best_params_['kernel'],
            probability=True
        )

        print("Meilleurs hyperparamètres:", clf.best_params_)
 
    def entrainement(self):
        clf = svm.SVC(
            C=1,  
            gamma='scale',
            kernel='rbf',
            probability=True
        )
        clf.fit(self.x_train, self.y_train)
        self.svm_classifier = clf
 
    def prediction(self):
        return self.svm_classifier.predict(self.x_test)
 
    def prediction_proba(self):
        return self.svm_classifier.predict_proba(self.x_test)

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class SVM_Classificateur(object):
    param_grid_default = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}
 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, param_grid=None):
        self.param_grid = param_grid or self.param_grid_default
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.svm_classifier = svm.SVC(probability=True)

    def validation_croisee_gridsearch(self):
        clf = GridSearchCV(self.svm_classifier, self.param_grid, cv=5, n_jobs=-1)
        clf.fit(self.x_val, self.y_val)  # Utilisation de x_val et y_val pour la recherche d'hyperparamètres

        # Mise à jour du classificateur avec les meilleurs hyperparamètres trouvés
        self.svm_classifier = clf.best_estimator_

        print("Meilleurs hyperparamètres:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.svm_classifier.fit(self.x_train, self.y_train)  # Entraînement avec x_train et y_train

    def prediction(self):
        return self.svm_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.svm_classifier.predict_proba(self.x_test)


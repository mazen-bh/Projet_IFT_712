from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

class AdaBoost_model(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.ab_classifier = AdaBoostClassifier()

    def validation_croisee_gridsearch(self):
        dt_params = {
            'max_depth': [1],  # Réduction de la profondeur maximale
            'min_samples_split': [6, 8],  # Augmentation de min_samples_split
            'min_samples_leaf': [3, 4],  # Augmentation de min_samples_leaf
            'max_features': [None, 'sqrt']
        }

        parameters = {
            'n_estimators': [30, 50],  # Réduction du nombre d'estimateurs
            'learning_rate': [0.01, 0.1],
            'base_estimator': [DecisionTreeClassifier(**params) for params in ParameterGrid(dt_params)],
            'algorithm': ['SAMME.R']  # Utilisation de SAMME.R
        }

        stratified_k_fold = StratifiedKFold(n_splits=5)

        clf = GridSearchCV(self.ab_classifier, parameters, cv=stratified_k_fold)
        clf.fit(self.x_train, self.y_train)

        self.ab_classifier.set_params(**clf.best_params_)
        print("Meilleurs hyperparamètres:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.ab_classifier.fit(self.x_train, self.y_train)

    def prediction(self):
        return self.ab_classifier.predict(self.x_test)

    def predict_proba(self):
        return self.ab_classifier.predict_proba(self.x_test)

    def evaluer_sur_validation(self):
        predictions_val = self.ab_classifier.predict(self.x_val)
        accuracy_val = np.mean(predictions_val == self.y_val)
        print("Précision sur l'ensemble de validation:", accuracy_val)

# Exemple d'utilisation du modèle
# Assurez-vous de remplacer ces variables par vos propres données
# x_train, y_train, x_val, y_val, x_test, y_test = [Votre ensemble de données ici]
# model = AdaBoost_model(x_train, y_train, x_val, y_val, x_test, y_test)
# model.entrainement()
# model.evaluer_sur_validation()
# predictions = model.prediction()

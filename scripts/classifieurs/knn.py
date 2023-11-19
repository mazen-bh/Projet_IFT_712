from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix

class Knn(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.n_estimators = 50
        self.base_classifier = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = None
        self.bc_classifier = BaggingClassifier(
            base_estimator=None, 
            n_estimators=self.n_estimators
        )
    def recherche_hyper(self):
        p_grid = {'n_estimators': np.arange(5, 101, 5),
                  'max_samples': np.arange(0.1, 1.1, 0.1),
                  'base_estimator__criterion': ['gini', 'entropy'],
                  'base_estimator__max_depth': [None, 10, 20],
                  'base_estimator__min_samples_split': [2, 3, 4],
                  'base_estimator__max_features': ['auto', 'sqrt']}

        cross_v = KFold(n_splits=10, shuffle=True, random_state=42)

        # Recherche d'hyperparamètres
        base_estimator = DecisionTreeClassifier()
        self.classif = RandomizedSearchCV(estimator=BaggingClassifier(base_estimator),
                                           param_distributions=p_grid, n_iter=25, cv=cross_v)
        self.classif.fit(self.x_train, self.y_train)
        best_params = self.classif.best_params_

        return best_params

    def garder_meilleur_hyperparameters(self):
        best_params = self.recherche_hyper()
        base_estimator = DecisionTreeClassifier(criterion=best_params['base_estimator__criterion'],
                                                max_depth=best_params['base_estimator__max_depth'],
                                                min_samples_split=best_params['base_estimator__min_samples_split'],
                                                max_features=best_params['base_estimator__max_features'])
        self.classif = BaggingClassifier(base_estimator=base_estimator,
                                         n_estimators=best_params['n_estimators'],
                                         max_samples=best_params['max_samples'])
        self.classif.fit(self.x_train, self.y_train)

    def entrainement(self):
        model_bc = BaggingClassifier(
            base_estimator=self.base_classifier,
            n_estimators=self.n_estimators
        )

        model_bc.fit(self.x_train, self.y_train)
        self.bc_classifier = model_bc

        train_sizes, train_scores, test_scores = learning_curve(
            model_bc, self.x_train, self.y_train, cv=2, scoring="accuracy")

        learning_curve_data = {
            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1),
            "train_loss": np.mean(train_scores, axis=1),
            "val_loss": np.mean(test_scores, axis=1)
        }
        self.learning_curve_data = learning_curve_data

    def prediction(self):
        return self.classif.predict(self.x_test)

    def prediction_proba(self):
        return self.classif.predict_proba(self.x_test)

    def resultats_model(self):
        y_pred = self.classif.predict(self.x_test)
        print("Matrice de confusion:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nRapport de classification:")
        print(classification_report(self.y_test, y_pred))

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier  # Utilisation d'un classifieur d'arbre de décision comme base_estimator
 
class Bagging_model(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.classif = None
 
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
        if self.classif is None:
            print('Bagging Classifier non entraîné. Utilisez garder_meilleur_hyperparameters() avant d\'appeler entrainement().')
            return
        self.classif.fit(self.x_train, self.y_train)
 
    def prediction(self):
        if self.classif is None:
            print('Bagging Classifier non entraîné. Utilisez garder_meilleur_hyperparameters() avant d\'appeler prediction().')
            return
        return self.classif.predict(self.x_test)
 
    def prediction_proba(self):
        if self.classif is None:
            print('Bagging Classifier non entraîné. Utilisez garder_meilleur_hyperparameters() avant d\'appeler prediction_proba().')
            return
        return self.classif.predict_proba(self.x_test)
 
    def resultats_model(self):
        if self.classif is None:
            print('Bagging Classifier non entraîné. Utilisez garder_meilleur_hyperparameters() avant d\'appeler resultats_model().')
            return
        y_pred = self.classif.predict(self.x_test)
        print("Matrice de confusion:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nRapport de classification:")
        print(classification_report(self.y_test, y_pred))
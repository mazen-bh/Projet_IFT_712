from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
 
class Knn(object):
    param_grid_default = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, param_grid=None):
        self.param_grid = param_grid or self.param_grid_default
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.learning_curve_data = None
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            p=2
        )
 
    def validation_croisee_gridsearch(self):
        parameters = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
 
        clf = GridSearchCV(self.knn_classifier, parameters, cv=2)
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        clf.fit(combined_x, combined_y)
 
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=clf.best_params_["n_neighbors"],
            weights=clf.best_params_["weights"],
            p=clf.best_params_["p"]
        )
 
        print("Meilleurs hyperparamètres:", clf.best_params_)
        return combined_x, combined_y
        combined_x, combined_y = self.validation_croisee_gridsearch()
        self.knn_classifier.fit(combined_x, combined_y)

    def entrainement(self):
        clf = KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            p=2
        )
        clf.fit(self.x_train, self.y_train)
        self.knn_classifier = clf
  
 
 
    def prediction(self):
        return self.knn_classifier.predict(self.x_test)
 
    def prediction_proba(self):
        return self.knn_classifier.predict_proba(self.x_test)
 
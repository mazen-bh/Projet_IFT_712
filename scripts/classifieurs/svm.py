from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

class SVM_Classificateur(object):
 
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
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
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        clf.fit(combined_x, combined_y)
 
        self.nmb_arbre = clf.best_params_["C"]
        self.criterion = clf.best_params_["gamma"]
        self.min_samples_split = clf.best_params_["kernel"]
 
        print("Meilleurs hyperparamètres:", clf.best_params_)
        return combined_x, combined_y
 
    def garder_meilleur_hyperparameters(self):
        combined_x, combined_y = self.validation_croisee_gridsearch()
        self.svm_classifier = svm.SVC(
            C=self.nmb_arbre,
            gamma='scale',
            kernel='rbf',
            probability=True
        )
        self.svm_classifier.fit(combined_x, combined_y)
 
    def entrainement(self):
        clf = svm.SVC(
            C=self.nmb_arbre,
            gamma='scale',
            kernel='rbf',
            probability=True
        )
        clf.fit(self.x_train, self.y_train)
        self.svm_classifier = clf
        train_sizes, train_scores, test_scores = learning_curve(
        clf, self.x_train, self.y_train, cv=2, scoring="accuracy")

        learning_curve_data = {
            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1),
            "train_loss": np.mean(train_scores, axis=1), 
            "val_loss": np.mean(test_scores, axis=1)  
        }
        self.learning_curve_data = learning_curve_data

    def prediction(self):
        return self.svm_classifier.predict(self.x_test)
 
    def prediction_proba(self):
        return self.svm_classifier.predict_proba(self.x_test)
 
    def resultats_model(self):
        y_pred = self.svm_classifier.predict(self.x_test)
        print("Matrice de confusion:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nRapport de classification:")
        print(classification_report(self.y_test, y_pred))
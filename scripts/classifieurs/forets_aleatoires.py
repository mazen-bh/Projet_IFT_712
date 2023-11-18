from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve


class Forets_aleatoires(object):

    def __init__(self, x_train, y_train, x_val, y_val,x_test,y_test):
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
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.nmb_arbre,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            max_features=self.max_features
        )

    def validation_croisee_gridsearch(self):
        parameters = {
            'n_estimators': np.arange(100, 702, 100),
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3, 4],
            'max_depth': [None, 10, 20],
            'max_features': ['auto', 'sqrt']
        }

        clf = GridSearchCV(self.rf_classifier, parameters, cv=2)
        combined_x = pd.concat([self.x_train, self.x_val], ignore_index=True)
        combined_y = self.y_train + self.y_val
        clf.fit(combined_x, combined_y)

        self.nmb_arbre = clf.best_params_["n_estimators"]
        self.criterion = clf.best_params_["criterion"]
        self.min_samples_split = clf.best_params_["min_samples_split"]
        self.max_depth = clf.best_params_["max_depth"]
        self.max_features = clf.best_params_["max_features"]

        print("Best hyperparameters:", clf.best_params_)
        return combined_x, combined_y

    def garder_meilleur_hyperparameters(self):
        combined_x, combined_y = self.validation_croisee_gridsearch() 
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.nmb_arbre,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            max_features=self.max_features
        )

        self.rf_classifier.fit(combined_x, combined_y)

    def entrainement(self):
        modele_rf = RandomForestClassifier(
            n_estimators=self.nmb_arbre,
            criterion=self.criterion,
            max_depth=None,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None, )


        modele_rf.fit(self.x_train, self.y_train)
        self.rf_classifier = modele_rf
        train_sizes, train_scores, test_scores = learning_curve(
        modele_rf, self.x_train, self.y_train, cv=2, scoring="accuracy")

        learning_curve_data = {
            "train_sizes": train_sizes,
            "train_accuracy": np.mean(train_scores, axis=1),
            "val_accuracy": np.mean(test_scores, axis=1),
            "train_loss": np.mean(train_scores, axis=1), 
            "val_loss": np.mean(test_scores, axis=1)  
        }
        self.learning_curve_data = learning_curve_data


    def prediction(self):
        return self.rf_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.rf_classifier.predict_proba(self.x_test)

    def resultats_model(self):
        y_pred = self.rf_classifier.predict(self.x_test)
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))





from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
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

# print recall
# print F1_score
# learning curve
# improve visualization confusion matrix

    # def plot_learning_curve(self,model, title="Learning Curve"):
    #     train_sizes, train_scores, test_scores = learning_curve(
    #         model.rf_classifier, self.x_train, self.y_train, cv=2, n_jobs=-1, 
    #         train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy")

    #     train_scores_mean = np.mean(train_scores, axis=1)
    #     train_scores_std = np.std(train_scores, axis=1)
    #     test_scores_mean = np.mean(test_scores, axis=1)
    #     test_scores_std = np.std(test_scores, axis=1)

    #     plt.figure()
    #     plt.title(title)
    #     plt.xlabel("Training examples")
    #     plt.ylabel("Score")

    #     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    #     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    #     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    #     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    #     plt.legend(loc="best")
    #     plt.grid(True)
    #     plt.show()

    def compute_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            self.rf_classifier, self.x_train, self.y_train, cv=2, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy")

        self.learning_curve_data = {
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "test_scores": test_scores
        }

    def plot_learning_curve(self, title="Learning Curve"):
        if self.learning_curve_data is None:
            self.compute_learning_curve()

        train_sizes = self.learning_curve_data["train_sizes"]
        train_scores = self.learning_curve_data["train_scores"]
        test_scores = self.learning_curve_data["test_scores"]

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

class Bagging(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.bg_classifier = BaggingClassifier()

    def validation_croisee_gridsearch(self):
        parameters = {
            'n_estimators': [30, 50, 100, 200],
            'max_samples': [0.5, 0.7, 0.9, 1.0],
            'max_features': [0.5, 0.7, 0.9, 1.0],
            'base_estimator': [DecisionTreeClassifier(max_depth=1), 
                               DecisionTreeClassifier(max_depth=2)]
        }

        clf = GridSearchCV(self.bg_classifier, parameters, cv=5)
        clf.fit(self.x_train, self.y_train)

        self.bg_classifier.set_params(**clf.best_params_)
        print("Meilleurs hyperparamètres pour Bagging:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.bg_classifier.fit(self.x_train, self.y_train)

    def prediction(self):
        return self.bg_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.bg_classifier.predict_proba(self.x_test)

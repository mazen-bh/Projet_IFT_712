from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

class Arbre_de_decision(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.dt_classifier = DecisionTreeClassifier()

    def validation_croisee_gridsearch(self):
        parameters = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 3, 5, 7],  # Réduire la complexité
            'min_samples_split': [2, 4, 6, 8, 10],  # Augmenter pour plus de généralisation
            'min_samples_leaf': [1, 2, 4, 6, 8],  # Augmenter pour plus de généralisation
            'max_features': [None, 'auto', 'sqrt', 'log2'],
            'class_weight': [None, 'balanced'],
            'min_impurity_decrease': [0.0, 0.01, 0.05],
            'max_leaf_nodes': [None, 10, 20, 30, 40, 50]  # Ajouter plus d'options
        }
        clf = GridSearchCV(self.dt_classifier, parameters, cv=5, n_jobs=-1, scoring='accuracy')
        clf.fit(self.x_train, self.y_train)  # Utilisation de x_train et y_train

        self.dt_classifier = clf.best_estimator_

        print("Meilleurs hyperparamètres:", clf.best_params_)

    def entrainement(self):
        self.validation_croisee_gridsearch()
        self.dt_classifier.fit(self.x_train, self.y_train)  # Entraînement avec x_train et y_train

    def prediction(self):
        return self.dt_classifier.predict(self.x_test)



  

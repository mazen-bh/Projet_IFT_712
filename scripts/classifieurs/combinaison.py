from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class Combinaison(object):
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        # Création des classificateurs individuels avec des hyperparamètres ajustés pour réduire l'overfitting
        self.decision_tree_classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=10, min_samples_leaf=5)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)
        self.svc_classifier = SVC(probability=True, C=0.1, kernel='linear')

    def validation_croisee_gridsearch(self):
        # Hyperparamètres à rechercher pour chaque classificateur individuel
        parametres_decision_tree = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10]
        }

        parametres_knn = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }

        parametres_svc = {
            'C': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf'],
        }

        # Recherche d'hyperparamètres pour chaque classificateur individuel
        grid_decision_tree = GridSearchCV(self.decision_tree_classifier, parametres_decision_tree, cv=5)
        grid_knn = GridSearchCV(self.knn_classifier, parametres_knn, cv=5)
        grid_svc = GridSearchCV(self.svc_classifier, parametres_svc, cv=5)

        # Exécution de la recherche d'hyperparamètres sur l'ensemble de validation
        grid_decision_tree.fit(self.x_val, self.y_val)
        grid_knn.fit(self.x_val, self.y_val)
        grid_svc.fit(self.x_val, self.y_val)

        # Mise à jour des classificateurs individuels avec les meilleurs hyperparamètres
        self.decision_tree_classifier = grid_decision_tree.best_estimator_
        self.knn_classifier = grid_knn.best_estimator_
        self.svc_classifier = grid_svc.best_estimator_

        print("Meilleurs hyperparamètres pour l'arbre de décision:", grid_decision_tree.best_params_)
        print("Meilleurs hyperparamètres pour KNN:", grid_knn.best_params_)
        print("Meilleurs hyperparamètres pour SVM:", grid_svc.best_params_)

    def entrainement(self):
        # Appel de la fonction de recherche d'hyperparamètres
        self.validation_croisee_gridsearch()
        
        # Entraînement des classificateurs individuels
        self.decision_tree_classifier.fit(self.x_train, self.y_train)
        self.knn_classifier.fit(self.x_train, self.y_train)
        self.svc_classifier.fit(self.x_train, self.y_train)

        # Création du modèle d'ensemble (VotingClassifier)
        self.ensemble_classifier = VotingClassifier(
            estimators=[
                ('decision_tree', self.decision_tree_classifier),
                ('knn', self.knn_classifier),
                ('svc', self.svc_classifier)
            ],
            voting='soft'
        )

        # Entraînement du modèle d'ensemble
        self.ensemble_classifier.fit(self.x_train, self.y_train)

    def prediction(self):
        return self.ensemble_classifier.predict(self.x_test)

    def prediction_proba(self):
        return self.ensemble_classifier.predict_proba(self.x_test)
